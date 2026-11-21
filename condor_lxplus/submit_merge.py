#!/usr/bin/env python
"""
Cascade merging of condor submission outputs on condor (HTCondor DAGMan), instead of one
long mergeoutputs.py run per output directory:

  stage 1  jobs merging --hist-files-per-job coffea files / --root-files-per-job ROOT files
           each into intermediate files <outputDir>/.merge_tmp/... (or, where all inputs
           fit into one job, directly into the final file)
  stage 2  jobs merging the intermediate files of an output directory into the final
           <outputDir>/hists_<name>.coffea and <outputDir>/hadd/<syst>/<dataset>.root
           (start when all stage-1 jobs of that directory are done)
  final    one job per output directory: checks the final files (ROOT: TotalEventCount
           entries = number of per-job files), writes the merge manifest and removes
           .merge_tmp

--split-hists: one file per histogram (<outputDir>/hists/<histogram>.coffea, index.json, see
mergeoutputs.py) instead of hists_<name>.coffea: the stage-1 jobs write their chunk as one
file per histogram, --key-groups stage-2 jobs merge the histograms (each job the histograms
of its group, one after the other), the final job checks them and writes hists/index.json.
No job holds more than one chunk or one merged histogram in memory.

The final files and the manifest are those of mergeoutputs.py, so a later interactive
mergeoutputs.py (or E_merge_stag.sh) run finds them complete and, with --delete-inputs,
removes the per-job files. Nothing per-job is deleted by the jobs. Files that are already
merged are skipped (-f: redo).

The jobs use the conda-pack tarball of the active environment in the sandbox directory on
EOS (as submitter.py), read the inputs as root:// URLs and copy the outputs with xrdcp.

Tasks: a file with one line per output directory, "<outputDir> <name> <only>" (only: hists,
arrays, all), or --task <outputDir> <name> <only> (repeatable).

Example
-------
    python condor_lxplus/submit_merge.py --jobDir /afs/cern.ch/work/v/vanderli/btv_jobs/merge_v4 \\
        --task /eos/user/v/vanderli/out/e_DY/MC e_DY_MC all --submit
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mergeoutputs as mo  # noqa: E402
from submitter import default_sandbox_dir, env_tarball, eos_url, get_proxy_path  # noqa: E402


def chunks(seq, n):
    return [seq[i : i + n] for i in range(0, len(seq), n)]


def plan_task(output_dir, name, only, args):
    """stage-1 items, stage-2 items and the finalize spec of one output directory"""
    output_dir = mo.canonical_eos(output_dir.rstrip("/"))
    files = mo.list_files(output_dir)
    hists, arrays, _ = mo.find_inputs(output_dir, files)
    existing = set(files)
    tmp = f"{output_dir}/.merge_tmp"
    s1, s2 = [], []
    spec = {"output_dir": output_dir, "outputs": {}, "checks": {}, "tmp": tmp}
    skipped = 0

    if hists and only in ("all", "hists") and args.split_hists:
        hdir = f"{output_dir}/{mo.HIST_DIR}"
        inputs = sorted(hists.values())
        done = (f"{hdir}/{mo.INDEX}" in existing
                and not (set(inputs) - mo.merged_hist_inputs(output_dir, files)))
        if done and not args.force:
            skipped += 1
        else:
            m = args.hist_files_per_job
            # unique per submission: leftovers of an earlier, failed cascade are not reused
            stamp = time.strftime("%Y%m%d_%H%M%S")
            if len(inputs) <= m:
                s1.append({"type": "coffea_split", "target": hdir, "inputs": inputs,
                           "entries": None})
                spec["split"] = {"hists_dir": hdir, "parts": None}
            else:
                parts = []
                for i, c in enumerate(chunks(inputs, m)):
                    part = f"{tmp}/hparts_{stamp}/part{i}"
                    s1.append({"type": "coffea_parts", "target": part, "inputs": c,
                               "entries": None})
                    parts.append(part)
                meta = f"{tmp}/hmeta_{stamp}"
                for g in range(args.key_groups):
                    s2.append({"type": "coffea_keys", "target": hdir, "parts": parts,
                               "group": g, "ngroups": args.key_groups, "meta": meta,
                               "entries": None})
                spec["split"] = {"hists_dir": hdir, "parts": parts, "meta": meta}
            spec["outputs"][hdir] = inputs

    elif hists and only in ("all", "hists"):
        target = f"{output_dir}/hists_{name}.coffea"
        inputs = sorted(hists.values())
        done = target in existing and not (set(inputs) - mo.merged_hist_inputs(output_dir, files))
        if done and not args.force:
            skipped += 1
        else:
            m = args.hist_files_per_job
            if len(inputs) <= m:
                s1.append({"type": "coffea", "target": target, "inputs": inputs, "entries": None})
            else:
                parts = []
                for i, c in enumerate(chunks(inputs, m)):
                    part = f"{tmp}/hists_{name}_part{i}.coffea"
                    s1.append({"type": "coffea", "target": part, "inputs": c, "entries": None})
                    parts.append(part)
                s2.append({"type": "coffea", "target": target, "inputs": parts, "entries": None})
            spec["outputs"][target] = inputs
            spec["checks"][target] = None

    if arrays and only in ("all", "arrays"):
        m = args.root_files_per_job
        for group, inputs in sorted(arrays.items()):
            target = f"{output_dir}/hadd/{group}.root"
            if target in existing and not args.force and mo.root_complete(target, len(inputs)):
                skipped += 1
                continue
            if len(inputs) <= m:
                s1.append({"type": "root", "target": target, "inputs": inputs,
                           "entries": len(inputs)})
            else:
                parts = []
                for i, c in enumerate(chunks(inputs, m)):
                    part = f"{tmp}/hadd/{group}_part{i}.root"
                    s1.append({"type": "root", "target": part, "inputs": c, "entries": len(c)})
                    parts.append(part)
                s2.append({"type": "root", "target": target, "inputs": parts,
                           "entries": len(inputs)})
            spec["outputs"][target] = inputs
            spec["checks"][target] = len(inputs)
    return s1, s2, spec, skipped, len(hists), sum(map(len, arrays.values()))


def pack(items, max_files, max_items):
    """jobs: coffea items one per job; ROOT items packed up to max_files input files and
    max_items items per job"""
    jobs, cur, n = [], [], 0
    for it in items:
        if it["type"] != "root":
            jobs.append([it])
            continue
        k = len(it["inputs"])
        if cur and (n + k > max_files or len(cur) >= max_items):
            jobs.append(cur)
            cur, n = [], 0
        cur.append(it)
        n += k
    if cur:
        jobs.append(cur)
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default=None, help="file with lines '<outputDir> <name> <only>'")
    p.add_argument("--task", nargs=3, action="append", default=[],
                   metavar=("OUTPUTDIR", "NAME", "ONLY"), help="one task (repeatable)")
    p.add_argument("--jobDir", required=True, help="job directory (DAG, logs; on AFS)")
    p.add_argument("--jobName", default="merge", help="batch name (default: merge)")
    p.add_argument("--jobFlavour", default="workday",
                   choices=["espresso", "microcentury", "longlunch", "workday", "tomorrow",
                            "testmatch", "nextweek"])
    p.add_argument("--hist-files-per-job", type=int, default=20,
                   help="coffea files merged per stage-1 job (default: 20)")
    p.add_argument("--split-hists", action="store_true",
                   help="one file per histogram (hists/<histogram>.coffea) instead of "
                   "hists_<name>.coffea")
    p.add_argument("--key-groups", type=int, default=16,
                   help="--split-hists: stage-2 jobs per output directory, each merging "
                   "the histograms of its group (default: 16)")
    p.add_argument("--root-files-per-job", type=int, default=50,
                   help="ROOT files merged per stage-1 job (default: 50)")
    p.add_argument("--root-items-per-job", type=int, default=10,
                   help="at most this many ROOT outputs per job (default: 10)")
    p.add_argument("--cpus", type=int, default=4,
                   help="cores per job: parallel hadds / coffea readers (default: 4)")
    p.add_argument("--memory", type=int, default=None,
                   help="memory per job in MB (default: 2000 x cpus)")
    p.add_argument("--disk", default="20GB", help="scratch disk per job (default: 20GB)")
    p.add_argument("--hadd-threads", type=int, default=1, help="hadd -j per hadd")
    p.add_argument("--maxJobs", type=int, default=500, help="DAGMan -maxjobs (default: 500)")
    p.add_argument("-f", "--force", action="store_true", help="redo existing merges")
    p.add_argument("--accountingGroup", default=os.environ.get("BTV_CONDOR_ACCOUNTING_GROUP"),
                   help="condor accounting group (default: $BTV_CONDOR_ACCOUNTING_GROUP)")
    p.add_argument("--sandboxDir", default=default_sandbox_dir(),
                   help="EOS directory of the environment tarball (as submitter.py)")
    p.add_argument("--submit", action="store_true", help="submit (else: prepare only)")
    args = p.parse_args()

    tasks = [tuple(t) for t in args.task]
    if args.tasks:
        with open(args.tasks) as f:
            tasks += [tuple(line.split()) for line in f if line.strip() and not line.startswith("#")]
    for t in tasks:
        if len(t) != 3 or t[2] not in ("hists", "arrays", "all"):
            raise SystemExit(f"bad task {t}: '<outputDir> <name> <hists|arrays|all>'")
    if not tasks:
        raise SystemExit("no tasks")
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        raise SystemExit("activate the BTVNanoCommissioning environment (setup_env.sh) first")

    job_dir = os.path.abspath(args.jobDir)
    if os.path.exists(job_dir):
        raise SystemExit(f"{job_dir} exists: remove it or choose another --jobDir")

    # plan
    dag_nodes, dag_deps, specs = [], [], {}
    for k, (outdir, name, only) in enumerate(tasks):
        s1, s2, spec, skipped, n_h, n_a = plan_task(outdir, name, only, args)
        print(f"{outdir}: {n_h} coffea + {n_a} ROOT files, {skipped} merged files exist; "
              f"{len(s1)} stage-1 and {len(s2)} stage-2 merges")
        if not spec["checks"] and not spec.get("split"):
            continue
        pk1 = pack(s1, args.root_files_per_job, args.root_items_per_job)
        n1 = [f"t{k}_s1_{i}" for i in range(len(pk1))]
        pk2 = pack(s2, 10**9, args.root_items_per_job)
        n2 = [f"t{k}_s2_{i}" for i in range(len(pk2))]
        fin = f"t{k}_fin"
        for node, items in list(zip(n1, pk1)) + list(zip(n2, pk2)):
            specs[f"{node}.json"] = items
            dag_nodes.append((node, "items"))
        specs[f"{fin}.json"] = spec
        dag_nodes.append((fin, "finalize"))
        if n2:
            dag_deps.append((n1, n2))
            dag_deps.append((n2, [fin]))
        else:
            dag_deps.append((n1, [fin]))
    if not dag_nodes:
        print("nothing to merge")
        return 0

    os.makedirs(os.path.join(job_dir, "specs"))
    for d in ("log", "out", "err"):
        os.makedirs(os.path.join(job_dir, d))
    shutil.copy(get_proxy_path(), os.path.join(job_dir, "proxy"))
    for f in ("merge_job.sh", "mergeoutputs.py"):
        shutil.copy(os.path.join(HERE, f), os.path.join(job_dir, f))
    os.chmod(os.path.join(job_dir, "merge_job.sh"), 0o755)
    for fname, content in specs.items():
        with open(os.path.join(job_dir, "specs", fname), "w") as f:
            json.dump(content, f, indent=1)

    env_url = env_tarball(prefix, eos_url(args.sandboxDir))
    memory = args.memory or 2000 * args.cpus
    accounting = f'+AccountingGroup = "{args.accountingGroup}"' if args.accountingGroup else ""
    sub = f"""Universe   = vanilla
Executable = {job_dir}/merge_job.sh
Arguments  = $(mode) $(node).json {args.cpus} {args.hadd_threads} {args.cpus} {int(args.force)}

request_cpus   = {args.cpus}
request_memory = {memory}
request_disk   = {args.disk}
+JobFlavour = "{args.jobFlavour}"
{accounting}

Log    = {job_dir}/log/job.log
Output = {job_dir}/out/$(node).out
Error  = {job_dir}/err/$(node).err

should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = {env_url},{job_dir}/mergeoutputs.py,{job_dir}/specs/$(node).json
x509userproxy           = {job_dir}/proxy
on_exit_hold        = (ExitBySignal == False) && (ExitCode == 99)
on_exit_hold_reason = "merge job setup failed (exit code 99), see the err file of the node"

Queue
"""
    with open(os.path.join(job_dir, "merge.sub"), "w") as f:
        f.write(sub)
    lines = []
    for node, mode in dag_nodes:
        lines += [f"JOB {node} merge.sub", f'VARS {node} mode="{mode}" node="{node}"',
                  f"RETRY {node} 2"]
    for parents, children in dag_deps:
        lines.append(f"PARENT {' '.join(parents)} CHILD {' '.join(children)}")
    with open(os.path.join(job_dir, "merge.dag"), "w") as f:
        f.write("\n".join(lines) + "\n")
    n_jobs = len(dag_nodes)
    print(f"{n_jobs} merge jobs (DAG) prepared in {job_dir}")
    cmd = ["condor_submit_dag", "-batch-name", args.jobName, "-maxjobs", str(args.maxJobs),
           "merge.dag"]
    if not args.submit:
        print(f"submit with: cd {job_dir} && {' '.join(cmd)}")
        return 0
    res = subprocess.run(cmd, cwd=job_dir, capture_output=True, text=True)
    with open(os.path.join(job_dir, "condor_submit.log"), "w") as f:
        f.write(res.stdout + res.stderr)
    print(res.stdout.strip())
    if res.returncode != 0:
        raise SystemExit(f"condor_submit_dag failed:\n{res.stderr.strip()}")
    print(f"progress: condor_q -batch {args.jobName}, {job_dir}/merge.dag.dagman.out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
