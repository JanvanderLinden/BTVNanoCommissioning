#!/usr/bin/env python
"""
Condor jobs for scripts/plotdataMC.py: one job per histogram (the per-histogram files of
scripts/split_coffea.py, or of mergeoutputs.py --split-hists: <outputDir>/hists/, several
input directories, e.g. MC, minor_MC and data, split by ','), all jobs of one call in one
batch (JobBatchName).

The jobs use the conda-pack environment and code tarballs of the sandbox directory on EOS
(as submitter.py; built once per environment / code version), read their input file with
xrdcp and copy the plots to the output directory on EOS. Options of plotdataMC.py are given
after `--`, the same for all jobs.

Example
-------
    python condor_lxplus/submit_plots.py --phase stag_e_DY_s \\
        --indir $B/e_DY/plot_split --vars vars_e_DY.txt --outdir $WWW/plots/e_DY \\
        --jobDir /afs/cern.ch/work/v/vanderli/btv_jobs/plots_e_DY --submit \\
        -- --lumi 109815 --split flavor --syst-band
"""

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from submitter import (  # noqa: E402
    code_tarball,
    default_sandbox_dir,
    env_tarball,
    eos_url,
    get_proxy_path,
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", required=True, help="plotdataMC.py -p (workflow name)")
    p.add_argument("--indir", required=True,
                   help="EOS directory (directories, split by ',') of <variable>.coffea")
    p.add_argument("--vars", required=True, help="file with one variable per line")
    p.add_argument("--outdir", required=True, help="EOS directory of the plots")
    p.add_argument("--jobDir", required=True, help="job directory (configs, logs; on AFS)")
    p.add_argument("--jobName", default="plots", help="JobBatchName (default: plots)")
    p.add_argument("--jobFlavour", default="microcentury",
                   choices=["espresso", "microcentury", "longlunch", "workday", "tomorrow",
                            "testmatch", "nextweek"],
                   help="job flavour (default: microcentury, 1 h)")
    p.add_argument("--memory", type=int, default=2000, help="memory per job in MB")
    p.add_argument("--maxRunning", type=int, default=500, help="max_materialize")
    p.add_argument("--accountingGroup", default=os.environ.get("BTV_CONDOR_ACCOUNTING_GROUP"),
                   help="condor accounting group (default: $BTV_CONDOR_ACCOUNTING_GROUP)")
    p.add_argument("--sandboxDir", default=default_sandbox_dir(),
                   help="EOS directory of the tarballs (as submitter.py)")
    p.add_argument("--submit", action="store_true", help="submit (else: prepare only)")
    p.add_argument("plot_opts", nargs=argparse.REMAINDER,
                   help="-- further plotdataMC.py options")
    args = p.parse_args()

    opts = args.plot_opts[1:] if args.plot_opts[:1] == ["--"] else args.plot_opts
    for o in ("-o", "--outdir", "-i", "--input", "-p", "--phase", "-v", "--variable"):
        if o in opts:
            raise SystemExit(f"{o} is set by the jobs, not a plot option")
    with open(args.vars) as f:
        variables = [line.strip() for line in f if line.strip()]
    if not variables:
        raise SystemExit(f"no variables in {args.vars}")
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        raise SystemExit("activate the BTVNanoCommissioning environment (setup_env.sh) first")
    in_urls = [eos_url(d) for d in args.indir.split(",") if d]
    out_url = eos_url(args.outdir)

    job_dir = os.path.abspath(args.jobDir)
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir)  # job directories of earlier plot submissions are replaced
    for d in ("", "log", "out", "err"):
        os.makedirs(os.path.join(job_dir, d))
    shutil.copy(get_proxy_path(), os.path.join(job_dir, "proxy"))
    shutil.copy(os.path.join(HERE, "plot_job.sh"), os.path.join(job_dir, "plot_job.sh"))
    os.chmod(os.path.join(job_dir, "plot_job.sh"), 0o755)
    shutil.copy(os.path.join(BASE, "scripts", "plotdataMC.py"),
                os.path.join(job_dir, "plotdataMC.py"))
    with open(os.path.join(job_dir, "plot_opts.txt"), "w") as f:
        f.write("\n".join(opts) + ("\n" if opts else ""))
    with open(os.path.join(job_dir, "vars.txt"), "w") as f:
        f.write("\n".join(variables) + "\n")
    with open(os.path.join(job_dir, "indirs.txt"), "w") as f:
        f.write("\n".join(in_urls) + "\n")

    sandbox = eos_url(args.sandboxDir)
    env_url = env_tarball(prefix, sandbox)
    code_url = code_tarball(BASE, sandbox)
    host, path = out_url[len("root://"):].split("/", 1)
    subprocess.run(["xrdfs", f"root://{host}", "mkdir", "-p", "/" + path.lstrip("/")],
                   check=True)

    accounting = f'+AccountingGroup = "{args.accountingGroup}"' if args.accountingGroup else ""
    jdl = f"""Universe   = vanilla
Executable = {job_dir}/plot_job.sh
Arguments  = $(var) {args.phase} indirs.txt {out_url}

request_cpus   = 1
request_memory = {args.memory}
request_disk   = 5GB
max_materialize = {args.maxRunning}
+JobFlavour = "{args.jobFlavour}"
{accounting}
JobBatchName = {args.jobName}

Log    = {job_dir}/log/job.log_$(Cluster)
Output = {job_dir}/out/job.out_$(Cluster)-$(var)
Error  = {job_dir}/err/job.err_$(Cluster)-$(var)

should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = {env_url},{code_url},{job_dir}/plotdataMC.py,{job_dir}/plot_opts.txt,{job_dir}/indirs.txt
x509userproxy           = {job_dir}/proxy
on_exit_remove      = (ExitBySignal == False) && (ExitCode == 0)
on_exit_hold        = (ExitBySignal == False) && (ExitCode == 99)
on_exit_hold_reason = "plot job setup failed (exit code 99), see the err file of the job"
max_retries = 2

Queue var from {job_dir}/vars.txt
"""
    with open(os.path.join(job_dir, "plots.jdl"), "w") as f:
        f.write(jdl)
    print(f"{len(variables)} plot jobs prepared in {job_dir}")
    if not args.submit:
        print(f"submit with: cd {job_dir} && condor_submit plots.jdl")
        return 0
    res = subprocess.run(["condor_submit", "plots.jdl"], cwd=job_dir, capture_output=True,
                         text=True)
    with open(os.path.join(job_dir, "condor_submit.log"), "w") as f:
        f.write(res.stdout + res.stderr)
    print(res.stdout.strip())
    if res.returncode != 0:
        raise SystemExit(f"condor_submit failed:\n{res.stderr.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
