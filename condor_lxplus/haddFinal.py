import argparse, os
from glob import glob

# Force hadd's output compression to differ from the input files' (normalizeArrays.py's
# uproot.recreate default is zlib level 1, i.e. setting 101) so hadd always takes its "slower"
# full decompress/recompress merge path instead of fast-cloning basket bytes directly.
# Fast-clone has been observed to silently corrupt jagged/vector branches (e.g. SelJet_phi)
# written by uproot -- hadd's own exit code stays 0 when this happens, so this is not optional.
_HADD_COMPRESSION = "-f102"

_VALIDATE_MACRO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validateMerge.C")

parser = argparse.ArgumentParser(
    description="Merge per-dataset normalized array-output root files (see "
    "condor_lxplus/normalizeArrays.py) into one combined MC.root and one combined data.root "
    "per systematic. This is safe to do at this stage -- but not before normalizeArrays.py -- "
    "because every event already carries its final self-normalized `weight` branch "
    "(xsec * lumi / sumw, times the SF weight components), so merging trees across datasets "
    "no longer needs any dataset-specific bookkeeping (xSecWeight/sumw)."
)
parser.add_argument("output_dir", help="Where to write <output_dir>/<syst>/{MC,data}.root")
parser.add_argument(
    "--mc", nargs="+", required=True, metavar="DIR",
    help="One or more hadd_normalized directories to merge into MC.root, e.g. the MC and "
    "minor_MC job outputs' hadd_normalized directories",
)
parser.add_argument(
    "--data", nargs="*", default=[], metavar="DIR",
    help="Zero or more hadd_normalized directories to merge into data.root (omit to merge MC only)",
)
parser.add_argument(
    "-f", "--force", action="store_true",
    help="overwrite existing merged files instead of leaving them untouched (hadd's default)",
)
args = parser.parse_args()

systlist = sorted(
    set(os.path.basename(s) for d in (args.mc + args.data) for s in glob(f"{d}/*"))
)

if len(systlist) == 0:
    print("Found zero systematics under the given directories -- are you pointing at hadd_normalized dirs?")
    exit()

cmdlist = []
for syst in systlist:
    newoutdir = f"{args.output_dir}/{syst}"
    os.makedirs(newoutdir, exist_ok=True)

    mc_files = [f for d in args.mc for f in glob(f"{d}/{syst}/*.root")]
    data_files = [f for d in args.data for f in glob(f"{d}/{syst}/*.root")]

    if mc_files:
        mc_target = f"{newoutdir}/MC.root"
        hadd_cmd = (
            f"hadd -v 0 -k {_HADD_COMPRESSION} {'-f ' if args.force else ''}{mc_target} "
            + " ".join(mc_files)
        )
        validate_cmd = f'root -l -b -q \'{_VALIDATE_MACRO}("{mc_target}")\''
        # If either hadd or the validation crashes/fails, remove the target instead of
        # leaving a broken or silently-corrupt file where a valid merge is expected.
        cmdlist.append(f"{{ {hadd_cmd} && {validate_cmd} ; }} || rm -f {mc_target}")
    if data_files:
        data_target = f"{newoutdir}/data.root"
        hadd_cmd = (
            f"hadd -v 0 -k {_HADD_COMPRESSION} {'-f ' if args.force else ''}{data_target} "
            + " ".join(data_files)
        )
        validate_cmd = f'root -l -b -q \'{_VALIDATE_MACRO}("{data_target}")\''
        cmdlist.append(f"{{ {hadd_cmd} && {validate_cmd} ; }} || rm -f {data_target}")

if len(cmdlist) == 0:
    print("Found zero files to hadd. Are you pointing to the right directories?")
    exit()

haddfile = f"{args.output_dir}/dohadd_final.sh"
print()
with open(haddfile, "w") as outfl:
    for cmd in cmdlist:
        outfl.write(cmd + "\n")
        print(cmd)

print("\nNow switch to a root env and run:")
print("bash", haddfile)
print("or")
print(f"parallel :::: {haddfile}")
