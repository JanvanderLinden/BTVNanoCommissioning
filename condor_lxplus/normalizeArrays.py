import argparse, glob, os, subprocess, sys

# The actual read+add-branches+write is done by the companion ROOT macro (normalizeArray.C),
# not here: uproot 4.3.7's TTree writer has been observed to silently write malformed baskets
# for some jagged/vector branches on the uproot.open()+uproot.recreate() round trip this script
# used to do, even though the same branches read back cleanly beforehand. The macro's native
# ROOT CloneTree()+Fill() doesn't hit that bug. See normalizeArray.C for the weight branches
# list and the actual per-event logic.
_NORMALIZE_MACRO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "normalizeArray.C"
)

# Integrated luminosity per campaign, in /pb. Kept in sync with SFbc-2D/05_make_templates.sh.
LUMI_PB = {
    "Summer22": 7980,
    "Summer22EE": 26700,
    "Summer23": 18100,
    "Summer23BPix": 9690,
    "Summer24": 110000,
    "Prompt25": 110000,
    "CustomSummer24": 110000,  # private NanoAODv15 production, uses Summer24 corrections/lumi
    "CustomSummer24UParTPaper": 110000,  # 03Sept2026 phys_btag production, same Summer24 corrections/lumi
    "CustomSummer25": 110628,  # CustomSummer24 MC + private 2025 data; 2025 golden, brilcalc normtag_BRIL
}

parser = argparse.ArgumentParser(
    description="Add `lumi` (the integrated luminosity, in pb, used for this normalization) and "
    "`weight` (xsec * lumi / sumw, times the SF weight components already written per event) "
    "branches to hadd'd array-output root files produced by array_writer() (see "
    "condor_lxplus/haddoutputs.py for the hadd step). All existing branches, including the "
    "individual xSecWeight/genWeight/puwei/ftagWeight/ttbar_weight/muWeight/eleWeight "
    "components `weight` is built from, are kept as-is."
)
parser.add_argument(
    "input_dir", help="Directory of hadd'd files, e.g. <outputDir>/hadd"
)
parser.add_argument(
    "output_dir",
    help="Where to write the normalized files, e.g. <outputDir>/hadd_normalized",
)
lumi_group = parser.add_mutually_exclusive_group(required=True)
lumi_group.add_argument(
    "--campaign",
    choices=sorted(LUMI_PB),
    help="Look up the integrated luminosity for this campaign",
)
lumi_group.add_argument("--lumi", type=float, help="Integrated luminosity in /pb")
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Print what would be done, don't write anything",
)


def normalize_file(in_file, out_file, lumi_pb, dry_run):
    if not dry_run:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
    macro_call = (
        f'{_NORMALIZE_MACRO}("{in_file}", "{out_file}", {lumi_pb}, '
        f'{"true" if dry_run else "false"})'
    )
    result = subprocess.run(["root", "-l", "-b", "-q", macro_call])
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    args = parser.parse_args()
    lumi_pb = LUMI_PB[args.campaign] if args.campaign else args.lumi

    files = glob.glob(f"{args.input_dir}/*/*.root")
    if len(files) == 0:
        print(
            f"Found zero .root files in {args.input_dir}/*/*.root -- are you pointing at a `hadd` directory?"
        )
        exit()

    for in_file in files:
        rel = os.path.relpath(in_file, args.input_dir)
        out_file = f"{args.output_dir}/{rel}"
        normalize_file(in_file, out_file, lumi_pb, args.dry_run)
