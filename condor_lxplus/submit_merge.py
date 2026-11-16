#!/usr/bin/env python
"""
Condor jobs for mergeoutputs.py: one job per (output directory, histograms or arrays), so
that the merges of many submissions run in parallel on the batch system instead of one
after the other on lxplus.

The jobs use the conda-pack tarball of the active environment in the sandbox directory on
EOS (as submitter.py; built once per environment version), read the inputs as root:// URLs
and copy the merged files to EOS with xrdcp. They do not delete anything: after all jobs
finished, run mergeoutputs.py (or E_merge_stag.sh) interactively once more, which finds the
merged files complete, and with --delete-inputs to remove the per-job files.

Tasks: a file with one line per job, "<outputDir> <name> <only>" (only: hists, arrays, all),
or --task <outputDir> <name> <only> (repeatable).

Example
-------
    python condor_lxplus/submit_merge.py --jobDir /afs/cern.ch/work/v/vanderli/btv_jobs/merge_v4 \\
        --task /eos/user/v/vanderli/out/e_DY/MC e_DY_MC hists \\
        --task /eos/user/v/vanderli/out/e_DY/MC e_DY_MC arrays --submit
"""

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from submitter import default_sandbox_dir, env_tarball, eos_url, get_proxy_path  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default=None, help="file with lines '<outputDir> <name> <only>'")
    p.add_argument("--task", nargs=3, action="append", default=[],
                   metavar=("OUTPUTDIR", "NAME", "ONLY"), help="one task (repeatable)")
    p.add_argument("--jobDir", required=True, help="job directory (configs, logs; on AFS)")
    p.add_argument("--jobName", default="merge", help="JobBatchName (default: merge)")
    p.add_argument("--jobFlavour", default="workday",
                   choices=["espresso", "microcentury", "longlunch", "workday", "tomorrow",
                            "testmatch", "nextweek"])
    p.add_argument("--cpus", type=int, default=4,
                   help="cores per job: parallel hadd processes / coffea readers (default: 4)")
    p.add_argument("--memory", type=int, default=None,
                   help="memory per job in MB (default: 2000 x cpus)")
    p.add_argument("--disk", default="30GB",
                   help="scratch disk per job: the merged files are written there before the "
                   "copy to EOS (default: 30GB)")
    p.add_argument("--hadd-threads", type=int, default=1, help="hadd -j per hadd")
    p.add_argument("--hist-workers", type=int, default=None,
                   help="processes reading the coffea files (default: --cpus)")
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
    for d in ("", "log", "out", "err"):
        os.makedirs(os.path.join(job_dir, d))
    shutil.copy(get_proxy_path(), os.path.join(job_dir, "proxy"))
    for f in ("merge_job.sh", "mergeoutputs.py"):
        shutil.copy(os.path.join(HERE, f), os.path.join(job_dir, f))
    os.chmod(os.path.join(job_dir, "merge_job.sh"), 0o755)
    with open(os.path.join(job_dir, "tasks.txt"), "w") as f:
        f.write("\n".join(" ".join(t) for t in tasks) + "\n")

    env_url = env_tarball(prefix, eos_url(args.sandboxDir))
    memory = args.memory or 2000 * args.cpus
    workers = args.hist_workers or args.cpus
    accounting = f'+AccountingGroup = "{args.accountingGroup}"' if args.accountingGroup else ""
    jdl = f"""Universe   = vanilla
Executable = {job_dir}/merge_job.sh
Arguments  = $(outdir) $(name) $(only) {args.cpus} {args.hadd_threads} {workers}

request_cpus   = {args.cpus}
request_memory = {memory}
request_disk   = {args.disk}
+JobFlavour = "{args.jobFlavour}"
{accounting}
JobBatchName = {args.jobName}

Log    = {job_dir}/log/job.log_$(Cluster)
Output = {job_dir}/out/job.out_$(Cluster)-$(Process)
Error  = {job_dir}/err/job.err_$(Cluster)-$(Process)

should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = {env_url},{job_dir}/mergeoutputs.py
x509userproxy           = {job_dir}/proxy
on_exit_remove      = (ExitBySignal == False) && (ExitCode == 0)
on_exit_hold        = (ExitBySignal == False) && (ExitCode == 99)
on_exit_hold_reason = "merge job setup failed (exit code 99), see the err file of the job"
max_retries = 2

Queue outdir, name, only from {job_dir}/tasks.txt
"""
    with open(os.path.join(job_dir, "merge.jdl"), "w") as f:
        f.write(jdl)
    print(f"{len(tasks)} merge jobs prepared in {job_dir}")
    if not args.submit:
        print(f"submit with: cd {job_dir} && condor_submit merge.jdl")
        return 0
    res = subprocess.run(["condor_submit", "merge.jdl"], cwd=job_dir, capture_output=True,
                         text=True)
    with open(os.path.join(job_dir, "condor_submit.log"), "w") as f:
        f.write(res.stdout + res.stderr)
    print(res.stdout.strip())
    if res.returncode != 0:
        raise SystemExit(f"condor_submit failed:\n{res.stderr.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
