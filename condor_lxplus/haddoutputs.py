import os, sys, json
from glob import glob

if len(sys.argv) < 2:
    raise ValueError(
        "Syntax: python condor_lxplus/haddoutputs.py output_dir [-f/--force]\n"
        "  -f/--force: overwrite existing merged files instead of leaving them untouched"
        " (hadd's default) -- needed to redo a merge after more jobs have landed."
    )

force = "-f" in sys.argv or "--force" in sys.argv
sys.argv = [a for a in sys.argv if a not in ("-f", "--force")]

outputdir = sys.argv[1]

arraydirs = [i for i in os.listdir(outputdir) if i.startswith("arrays_")]
systlist = list(set([i.split("/")[-1] for i in glob(f"{outputdir}/arrays_*/*")]))

cmdlist = []
for syst in systlist:
    samplist = list(
        set([i.split("/")[-1] for i in glob(f"{outputdir}/arrays_*/{syst}/*")])
    )
    newoutdir = f"{outputdir}/hadd/{syst}"
    os.system("mkdir -p " + newoutdir)
    for samp in samplist:
        # -k: skip corrupt/unreadable/non-existent input files (e.g. a job still running or an
        # EOS/xrootd blip) instead of aborting the whole dataset's merge because of one of them.
        cmd = f"hadd -v 0 -k {'-f ' if force else ''}{newoutdir}/{samp}.root {outputdir}/arrays_*/{syst}/{samp}/*.root"
        cmdlist.append(cmd)

if len(cmdlist) == 0:
    print("Found zero files to hadd. Are you pointing to the right output directory?")
    exit()

haddfile = f"{outputdir}/dohadd.sh"
print()
with open(haddfile, "w") as outfl:
    for cmd in cmdlist:
        outfl.write(cmd + "\n")
        print(cmd)

print("\nNow switch to a root env and run:")
print("bash", haddfile)
print("or")
print(f"parallel :::: {haddfile}")
