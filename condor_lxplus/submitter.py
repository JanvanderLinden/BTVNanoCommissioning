import os
import json
import time
import shutil
import hashlib
import tarfile
import argparse
import tempfile
import subprocess


# From https://github.com/PocketCoffea/PocketCoffea/blob/main/pocket_coffea/utils/network.py
def get_proxy_path() -> str:
    """
    Checks if the VOMS proxy exists and if it is valid
    for at least 1 hour.
    If it exists, returns the path of it"""
    try:
        subprocess.run("voms-proxy-info -exists -valid 0:20", shell=True, check=True)
    except subprocess.CalledProcessError:
        raise Exception(
            "VOMS proxy expirend or non-existing: please run `voms-proxy-init -voms cms -rfc --valid 168:0`"
        )

    # Now get the path of the certificate
    proxy = subprocess.check_output(
        "voms-proxy-info -path", shell=True, text=True
    ).strip()
    return proxy


# ------------------------------------------------------------------------------------------
# Job sandbox: the jobs do not use AFS. The conda environment (conda-pack) and the code
# (runner.py + src/) are packed into tarballs on EOS, content-hashed so that they are only
# rebuilt when the environment or the code changes; the jobs get them with the EOS xrootd
# transfer plugin and unpack them in their local scratch directory (execute.sh).
# ------------------------------------------------------------------------------------------


def eos_url(path):
    """root:// URL of an EOS path (must match checkoutputs.py)."""
    if path.startswith("root://"):
        return path.rstrip("/")
    if path.startswith("/eos/home-"):
        path = f"/eos/user/{path[10]}/{path[12:]}"
    if path.startswith("/eos/user/"):
        return "root://eosuser.cern.ch/" + path.rstrip("/")
    if path.startswith("/eos/"):
        return "root://eoscms.cern.ch/" + path.rstrip("/")
    raise SystemExit(f"ERROR: not an EOS path: '{path}'")


def _split_url(url):
    rest = url[len("root://") :]
    host, path = rest.split("/", 1)
    return f"root://{host}", "/" + path.lstrip("/")


def xrd_exists(url):
    host, path = _split_url(url)
    return subprocess.run(["xrdfs", host, "stat", path], capture_output=True).returncode == 0


def xrdcp_to(local, url):
    for attempt in range(3):
        res = subprocess.run(["xrdcp", "-f", "-p", "-N", local, url], capture_output=True, text=True)
        if res.returncode == 0:
            return
        time.sleep(10 * (attempt + 1))
    raise SystemExit(f"ERROR: xrdcp {local} -> {url} failed:\n{res.stderr}")


def _tmpdir():
    base = os.environ.get("TMPDIR") or f"/tmp/{os.environ.get('USER', 'user')}"
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix="btv_sandbox_", dir=base)


def env_tarball(prefix, sandbox_url):
    """URL of the conda-pack tarball of the environment `prefix` in sandbox_url; built and
    uploaded if no tarball of the same package set exists yet."""
    listing = subprocess.run(
        ["conda", "list", "--explicit", "--md5", "-p", prefix],
        capture_output=True, text=True, check=True,
    ).stdout
    pip = subprocess.run(
        [os.path.join(prefix, "bin", "python"), "-m", "pip", "freeze", "--exclude-editable"],
        capture_output=True, text=True, check=True,
    ).stdout
    digest = hashlib.sha1((listing + pip).encode()).hexdigest()[:12]
    url = f"{sandbox_url}/btv_env_{digest}.tar.gz"
    if xrd_exists(url):
        print(f"Environment tarball: {url} (exists)")
        return url
    if not shutil.which("conda-pack"):
        raise SystemExit(
            "ERROR: conda-pack is needed once per environment version to pack the environment "
            "for the jobs: pip install conda-pack  (in the active environment)"
        )
    tmp = _tmpdir()
    out = os.path.join(tmp, os.path.basename(url))
    print(f"Packing {prefix} with conda-pack (once per environment version, a few minutes)")
    subprocess.run(
        ["conda-pack", "-p", prefix, "-o", out, "--ignore-editable-packages",
         "--ignore-missing-files", "--compress-level", "4", "-j", "-1"],
        check=True,
    )
    print(f"Uploading {os.path.getsize(out) / 1e9:.2f} GB to {url}")
    xrdcp_to(out, url)
    shutil.rmtree(tmp, ignore_errors=True)
    return url


def _code_files(base_dir):
    files = [os.path.join(base_dir, "runner.py")]
    for root, dirs, names in os.walk(os.path.join(base_dir, "src")):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__" and not d.endswith(".egg-info"))
        files += [os.path.join(root, n) for n in sorted(names) if not n.endswith((".pyc", ".pyo"))]
    return files


def code_tarball(base_dir, sandbox_url):
    """URL of the tarball of runner.py and src/ (code, correction and model files) in
    sandbox_url; built and uploaded if no tarball with the same content exists yet."""
    files = _code_files(base_dir)
    h = hashlib.sha1()
    for f in files:
        h.update(os.path.relpath(f, base_dir).encode())
        with open(f, "rb") as fin:
            h.update(hashlib.sha1(fin.read()).digest())
    url = f"{sandbox_url}/btv_code_{h.hexdigest()[:12]}.tar.gz"
    if xrd_exists(url):
        print(f"Code tarball: {url} (exists)")
        return url
    tmp = _tmpdir()
    out = os.path.join(tmp, os.path.basename(url))
    with tarfile.open(out, "w:gz") as tar:
        for f in files:
            tar.add(f, arcname=os.path.relpath(f, base_dir))
    print(f"Uploading code tarball ({os.path.getsize(out) / 1e6:.0f} MB) to {url}")
    xrdcp_to(out, url)
    shutil.rmtree(tmp, ignore_errors=True)
    return url


def default_sandbox_dir():
    user = os.environ.get("USER", "")
    return os.environ.get("BTV_CONDOR_SANDBOX") or f"/eos/user/{user[:1]}/{user}/btv_condor_sandbox"


def get_condor_submitter_parser(parser):
    parser.add_argument(
        "--jobName",
        help="Condor job name to make the job directory",
        required=True,
    )
    parser.add_argument(
        "--condorFileSize",
        type=int,
        default=1,
        help="Number of files to process per condor job",
    )
    parser.add_argument(
        "--outputDir",
        help="Output directory",
        required=True,
    )
    parser.add_argument(
        "--remoteRepo",
        default=None,
        help="If specified, access BTVNanoCommissioning from a remote tarball (downloaded via https), instead of from a transferred sandbox",
    )
    parser.add_argument(
        "--jobDir",
        default=os.environ.get("BTV_CONDOR_JOBDIR", "."),
        help="Directory in which the job directory jobs_<jobName> (configs, logs) is "
        "created (default: $BTV_CONDOR_JOBDIR, else the current directory). Outside /eos "
        "(e.g. on AFS) the jobs are submitted without spooling and the logs are written "
        "there directly.",
    )
    parser.add_argument(
        "--accountingGroup",
        default=os.environ.get("BTV_CONDOR_ACCOUNTING_GROUP"),
        help="HTCondor accounting group (+AccountingGroup), e.g. group_u_CMST3.all "
        "(default: $BTV_CONDOR_ACCOUNTING_GROUP, else none)",
    )
    parser.add_argument(
        "--sandboxDir",
        default=default_sandbox_dir(),
        help="EOS directory of the environment and code tarballs of the jobs "
        "(default: $BTV_CONDOR_SANDBOX, else /eos/user/<u>/<user>/btv_condor_sandbox)",
    )
    parser.add_argument(
        "--maxRunning",
        type=int,
        default=200,
        help="Maximum number of jobs of this submission in the queue at a time "
        "(max_materialize; the others are added as these finish). Default: %(default)s",
    )
    parser.add_argument(
        "--jobFlavour",
        default="workday",
        choices=["espresso", "microcentury", "longlunch", "workday", "tomorrow",
                 "testmatch", "nextweek"],
        help="CERN job flavour = maximum run time (20 min, 1 h, 2 h, 8 h, 1 d, 3 d, 1 w); "
        "a job running longer is killed before it copies its outputs. Default: %(default)s",
    )
    parser.add_argument(
        "--disk",
        default="10GB",
        help="request_disk per job (unpacked environment + outputs). Default: %(default)s",
    )
    parser.add_argument(
        "--noSandbox",
        action="store_true",
        help="Run the jobs from this checkout and conda environment directly (shared "
        "filesystem, e.g. AFS). Do not use for more than a few jobs: hundreds of jobs "
        "reading a conda environment on AFS overload the AFS server.",
    )
    parser.add_argument(
        "--noSpool", action="store_true", help="Submit condor jobs without spooling."
    )
    parser.add_argument(
        "--nCPU",
        default=1,
        help="Number of CPUs to request for condor. RAM of the job scales as 2Gb*nCPU",
    )
    return parser


def get_main_parser():
    parser = argparse.ArgumentParser(description="Arguments for condor submitter")
    ## Inputs
    parser.add_argument(
        "--wf",
        "--workflow",
        dest="workflow",
        help="Which processor to run",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        default=r"hists.coffea",
        help="Output histogram filename (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        "--json",
        dest="samplejson",
        default="dummy_samples.json",
        help="JSON file containing dataset and file locations (default: %(default)s)",
    )
    ## Configuations
    parser.add_argument("--year", default="2023", help="Year")
    parser.add_argument(
        "--campaign",
        default="Summer23",
        choices=[
            "Summer22",
            "Summer22EE",
            "Summer23",
            "Summer23BPix",
            "Summer24",
            "CustomSummer24",
            "CustomSummer24UParTPaper",
            "CustomSummer25",
            "2018-UL",
            "2017-UL",
            "2016preVFP-UL",
            "2016postVFP-UL",
            "CAMPAIGN_prompt_dataMC",
        ],
        help="Dataset campaign, change the corresponding correction files",
    )
    parser.add_argument(
        "--isSyst",
        default=False,
        type=str,
        choices=[
            "False",
            "all",
            "weight_only",
            "JEC_full",
            "JEC_reduced",
            "JEC_reduced_JER_split",
            "JEC_total",
            "JP_MC",
        ],
        help="Run with systematics, all, weights_only(no JERC uncertainties included),JERC_split, None",
    )
    parser.add_argument(
        "--ttbar-reweights",
        dest="ttbar_reweights",
        default="none",
        choices=["none", "hdamp_ml", "full"],
        help=(
            "Forwarded to runner.py --ttbar-reweights: enable additional "
            "ttbar event reweights in correction.py. 'hdamp_ml' applies hdamp "
            "ONNX up/down; 'full' additionally reserves hooks for frag/decay "
            "reweights. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--nlo-reweight",
        dest="nlo_reweight",
        action="store_true",
        help="Forwarded to runner.py --nlo-reweight",
    )
    parser.add_argument(
        "--vjets-stitch",
        dest="vjets_stitch",
        default=None,
        type=lambda p: os.path.abspath(p),
        help="Forwarded to runner.py --vjets-stitch (FxFx W/DY LHE stitching json).",
    )
    parser.add_argument("--isArray", action="store_true", help="Output root files")
    parser.add_argument(
        "--array-systs",
        dest="array_systs",
        default="none",
        choices=["none", "weights", "shifts", "both"],
        help="Forwarded to runner.py --array-systs: how much systematic "
        "content the --isArray trees carry. Default: %(default)s",
    )

    parser.add_argument(
        "--noHist", action="store_true", help="Not output coffea histogram"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Only  process/skip part of the dataset. By input list of file",
    )
    parser.add_argument(
        "--selectionModifier",
        type=str,
        default=None,
        help="selection Modifier",
    )
    parser.add_argument(
        "--voms",
        default=None,
        type=str,
        help="Path to voms proxy, made accessible to worker nodes. By default a copy will be made to $HOME.",
    )

    parser.add_argument(
        "--chunk",
        type=int,
        default=75000,
        metavar="N",
        help="Number of events per process chunk",
    )
    parser.add_argument("--skipbadfiles", action="store_true", help="Skip bad files.")
    parser.add_argument("--submit", action="store_true", help="Submit condor jobs.")
    parser = get_condor_submitter_parser(parser)
    return parser


if __name__ == "__main__":
    parser = get_main_parser()
    args = parser.parse_args()
    print("Running with the following options:")
    print(args)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = current_dir.replace("/condor_lxplus", "")
    # Create job dir
    os.makedirs(args.jobDir, exist_ok=True)
    job_dir = os.path.abspath(os.path.join(args.jobDir, f"jobs_{args.jobName}"))
    # the jobs copy their outputs to --outputDir at the end: absolute path or root:// URL
    if "://" not in args.outputDir:
        args.outputDir = os.path.abspath(args.outputDir)
    if os.path.exists(job_dir):
        user_input = input("Job directory already exists, overwrite? (y/n): ")
        if user_input.lower() == "y":
            shutil.rmtree(job_dir)
        else:
            raise Exception("Job exiting...")
    print(f"Job directory created: {job_dir}")
    os.mkdir(job_dir)
    os.mkdir(job_dir + "/log")
    os.mkdir(job_dir + "/out")
    os.mkdir(job_dir + "/err")

    # Handle voms proxy
    if args.voms is not None:
        if not os.path.exists(args.voms):
            raise Exception(f"Provided voms proxy path {args.voms} does not exist.")
        proxy_path = os.path.join(job_dir, "proxy")
        shutil.copy(args.voms, proxy_path)
        print(f"Copied provided proxy file {args.voms} to local directory.")
    else:
        proxy_file = get_proxy_path()
        proxy_path = os.path.join(job_dir, "proxy")
        shutil.copy(proxy_file.strip(), proxy_path)
        print(f"Copied proxy file {proxy_file} to local directory.")

    # Find conda/mamba environment
    envpath = "/eos/home-m/milee/miniforge3/envs/btv_coffea/bin"
    pathvarlist = [
        i
        for i in os.environ["PATH"].split(":")
        if "envs/btv_coffea" in i or "envs/BTVCOFFEA" in i
    ]
    # the active conda/micromamba environment, whatever its name
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if "/envs/" in conda_prefix:
        pathvarlist.insert(0, os.path.join(conda_prefix, "bin"))
    if len(pathvarlist) == 0:
        print(
            f"You did not source the btv_coffea conda/mamba environment. Proceed with the central conda environment:\n{envpath} ?"
        )
        response = input("(y/yes): ").strip().lower()
        if response == "y" or response == "yes":
            pass
        else:
            print(
                "First source the conda environment with which 'pip install -e .' was run. Then proceed again."
            )
            exit()
    else:
        envpath = pathvarlist[0]
        print(
            f"Using conda installation in\n{envpath}.\nThis has to be the conda installation with which 'pip install -e .' was run. If not, please source the correct environment and run again."
        )

    # Store job submission files

    ## store parser arguments
    with open(os.path.join(job_dir, "arguments.json"), "w") as json_file:
        json.dump(vars(args), json_file, indent=4)

    ## split the sample json
    with open(args.samplejson) as f:
        sample_dict = json.load(f)
    split_sample_dict = {}
    counter = 0
    only = []
    if args.only is not None:
        if "*" in args.only:
            only = [
                k
                for k in sample_dict.keys()
                if k.lstrip("/").startswith(args.only.rstrip("*"))
            ]
        else:
            only.append(args.only)

    for sample_name, files in sample_dict.items():
        if len(only) != 0 and sample_name not in only:
            continue
        for ifile in range(
            (len(files) + args.condorFileSize - 1) // args.condorFileSize
        ):
            split_sample_dict[counter] = {
                sample_name: files[
                    ifile * args.condorFileSize : (ifile + 1) * args.condorFileSize
                ]
            }
            counter += 1

    ## store the split sample json file
    with open(os.path.join(job_dir, "split_samples.json"), "w") as json_file:
        json.dump(split_sample_dict, json_file, indent=4)
    ## store the jobnum list (0..jobnum-1)
    with open(os.path.join(job_dir, "jobnum_list.txt"), "w") as f:
        f.write("\n".join([str(i) for i in range(counter)]))

    ## job inputs: small files from the job directory; with the sandbox, the environment
    ## and code tarballs and the split json from EOS (xrootd transfer plugin), so that the
    ## jobs do not read from AFS
    transfer_inputs = [f"{job_dir}/arguments.json"]
    if args.noSandbox:
        print(
            "WARNING: --noSandbox: the jobs run from this checkout and environment. "
            "Keep the number of jobs small (AFS overload)."
        )
        transfer_inputs.append(f"{job_dir}/split_samples.json")
    else:
        sandbox_url = eos_url(args.sandboxDir)
        transfer_inputs.append(env_tarball(os.path.dirname(envpath.rstrip("/")), sandbox_url))
        transfer_inputs.append(code_tarball(base_dir, sandbox_url))
        # split json next to the outputs, where checkoutputs.py -u expects it
        split_url = eos_url(args.outputDir) + "/.condor_sandbox/split_samples.json"
        xrdcp_to(os.path.join(job_dir, "split_samples.json"), split_url)
        transfer_inputs.append(split_url)

    if "/eos/" in job_dir:
        print(
            "WARNING: Submitting from /eos. Log files will NOT be created.\nTo debug/retrieve logs, use `condor_transfer_data <job id>` after a job fails/terminates."
        )
        spool = "-spool"
    else:
        spool = ""

    if args.noSpool:
        print("Submitting without spooling due to --noSpool option.")
        spool = ""

    # The jobs write their outputs directly to --outputDir. Spool jobs must set
    # transfer_output_files (CERN schedds reject them otherwise): transfer nothing back
    # besides stdout/stderr.
    transfer_output = 'transfer_output_files   = ""' if spool else ""

    ## store the jdl file
    jdl_template = """Universe   = vanilla
Executable = {executable}


Arguments = $(JOBNUM) {base_dir} {outputDir} {envpath} {nthreads}

request_cpus = {nthreads}
request_disk = {disk}
max_materialize = {max_running}

+JobFlavour = "{job_flavour}"
{accounting}
# name shown by condor_q (BATCH_NAME) instead of ID_<cluster>
JobBatchName = {batch_name}

Log        = {log_dir}/job.log_$(Cluster)
Output     = {job_dir}/out/job.out_$(Cluster)-$(Process)
Error      = {job_dir}/err/job.err_$(Cluster)-$(Process)

should_transfer_files   = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_input_files    = {transfer_input_files}
{transfer_output}
x509userproxy           = {proxy}
on_exit_remove   = (ExitBySignal == False) && (ExitCode == 0)
# exit code 99 = setup failure of the job (execute.sh): the same for every attempt, so hold
# the job instead of requeueing it (condor_q -hold -af HoldReason, the err file has the reason)
on_exit_hold        = (ExitBySignal == False) && (ExitCode == 99)
on_exit_hold_reason = "job setup failed (exit code 99), see the err file of the job"
max_retries      = 3
requirements     = (Machine =!= split(LastRemoteHost, "@")[1])

Queue JOBNUM from {jobnum_file}
""".format(
        executable=f"{base_dir}/condor_lxplus/execute.sh",
        base_dir=base_dir,
        outputDir=args.outputDir,
        envpath=envpath,
        nthreads=args.nCPU,
        accounting=f'+AccountingGroup = "{args.accountingGroup}"' if args.accountingGroup else "",
        log_dir=f"{job_dir}/log",
        transfer_input_files=",".join(transfer_inputs),
        disk=args.disk,
        batch_name=args.jobName,
        job_flavour=args.jobFlavour,
        max_running=args.maxRunning,
        jobnum_file=f"{job_dir}/jobnum_list.txt",
        transfer_output=transfer_output,
        job_dir=job_dir,
        proxy=proxy_path,
    )
    with open(os.path.join(job_dir, "submit.jdl"), "w") as f:
        f.write(jdl_template)

    if args.submit:
        cmd = ["condor_submit"] + ([spool] if spool else []) + [f"{job_dir}/submit.jdl"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout.strip())
        if res.returncode != 0:
            # a refused submission must not look like a success (C_submit_stag.sh counts it)
            print(res.stderr.strip())
            raise SystemExit(
                f"ERROR: condor_submit failed ({res.returncode}) for {job_dir}/submit.jdl"
            )
        with open(os.path.join(job_dir, "condor_submit.log"), "w") as f:
            f.write(res.stdout + res.stderr)
    else:
        print(
            f"Setup completed. Now submit the condor jobs by:\n  condor_submit {spool} {job_dir}/submit.jdl"
        )
