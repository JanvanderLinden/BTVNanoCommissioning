import os, sys, json, subprocess
from copy import deepcopy
from collections import defaultdict
import time
import argparse
from rich import print
from BTVNanoCommissioning.utils.xrootdtools import get_xrootd_sites_map, find_other_file
from glob import glob
from alive_progress import alive_bar


def eos_url(path):
    """xrootd URL for a CERN EOS path -- must match condor/submitter.py."""
    if path.startswith("root://"):
        return path.rstrip("/")
    if path.startswith("/eos/user/") or path.startswith("/eos/home-"):
        return "root://eosuser.cern.ch/" + path.rstrip("/")
    if path.startswith("/eos/"):
        return "root://eoscms.cern.ch/" + path.rstrip("/")
    raise SystemExit(f"ERROR: cannot derive an EOS sandbox URL for '{path}'.")


parser = argparse.ArgumentParser()
parser.add_argument("job_dir", type=str, help="Input string")
parser.add_argument(
    "-u",
    "--update-xrootd",
    action="store_true",
    help="Update xrootd paths of failed jobs.",
)
args = parser.parse_args()

sitemap = get_xrootd_sites_map()
jobdir = args.job_dir
updatexrootd = args.update_xrootd

uid = os.getuid()
homedir = os.getenv("HOME")
expected_value = f"{homedir}/x509up_u{uid}"
current_value = os.getenv("X509_USER_PROXY")
if current_value != expected_value:
    print("[red][b]X509_USER_PROXY is NOT set correctly.[/][/]")
    print(f"Please run the following command in your shell:")
    print(f"export X509_USER_PROXY=$HOME/x509up_u`id -u`")
    sys.exit(1)

numlist = [i.split()[0] for i in open(f"{jobdir}/jobnum_list.txt", "r").readlines()]

with open(f"{jobdir}/arguments.json") as f:
    args = json.load(f)
configs = args.get("configs")
if configs:  # one output per config that the job ran
    split = {}
    for f in glob(f"{jobdir}/split/*/split_samples.json"):
        split.update(json.load(open(f)))
    if updatexrootd:
        sys.exit("ERROR: -u is not implemented for --configs job directories")

toresubmit = []

with alive_bar(len(numlist), title="Checking jobs") as bar:
    for n in numlist:
        outdirs = (
            [
                configs[int(i)]["outputDir"]
                for i in {c for d in split[n].values() for cs in d.values() for c in cs}
            ]
            if configs
            else [args["outputDir"]]
        )
        for outdir in outdirs:
            outfile = f"{outdir}/hists_{n}/hists_{n}.coffea"
            if not os.path.isfile(outfile):
                print(f"[red]Job {n}[/] does not have an output histogram file.")
                toresubmit.append(n)
                break
            arraylist = glob(f"{outdir}/arrays_hists_{n}/*/*/*.root")
            if any(os.path.getsize(f) < 10 * 1024 for f in arraylist):  # zombies
                print(f"[red]Job {n}[/] has at least one zombie root output file.")
                toresubmit.append(n)
                break
        bar()

if len(toresubmit) == 0:
    print("[green][b]All jobs complete. Nothing to resubmit![/][/]\n")
    print("[b]To hadd, run:[/]")
    print(f"[yellow]python condor_lxplus/haddoutputs.py {outdir}[/]")
    exit()

if updatexrootd:
    with open(f"{jobdir}/split_samples.json") as f:
        samples = json.load(f)
    newsamples = deepcopy(samples)
    for r in toresubmit:
        sampdict = samples[r]
        for samp, fllist in sampdict.items():
            newfllist = []
            for fl in fllist:
                newfl = find_other_file(fl, sitemap)
                newfllist.append(newfl)
                if newfl == fl:
                    print(
                        f"[red][b]WARNING:[/] Could not find replacement for {fl}.[/]"
                    )
                else:
                    print(f"[yellow]{fl}[/] [cyan]->[/] [green]{newfl}[/].")
            newsamples[r][samp] = newfllist
    with open(os.path.join(jobdir, "split_samples_resubmit.json"), "w") as json_file:
        json.dump(newsamples, json_file, indent=4)
    # resubmit.jdl inherits the root:// URL from submit.jdl with the filename
    # swapped to split_samples_resubmit.json (below), so push this file to the
    # same EOS .condor_sandbox the submitter used.
    sandbox_url = eos_url(outdir) + "/.condor_sandbox"
    subprocess.run(
        [
            "xrdcp",
            "-f",
            "-p",
            os.path.join(jobdir, "split_samples_resubmit.json"),
            f"{sandbox_url}/split_samples_resubmit.json",
        ],
        check=True,
    )


numlist2 = open(f"{jobdir}/jobnum_list_resubmit.txt", "w")
for r in toresubmit:
    numlist2.write(f"{r} {int(r) // 100}\n" if configs else f"{r}\n")
numlist2.close()

jdl = open(f"{jobdir}/submit.jdl", "r").readlines()
jdlnew = open(f"{jobdir}/resubmit.jdl", "w")
for line in jdl:
    towrite = line.replace("jobnum_list.txt", "jobnum_list_resubmit.txt")
    if updatexrootd:
        towrite = towrite.replace("split_samples.json", "split_samples_resubmit.json")
    jdlnew.write(towrite)

print(f"[yellow]Found {len(toresubmit)} missing outputs: {toresubmit}[/]")
print("[b]Resubmit with:[/]")
print(f"condor_submit {jobdir}/resubmit.jdl")
