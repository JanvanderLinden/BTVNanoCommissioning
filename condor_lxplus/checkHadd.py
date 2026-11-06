import argparse, os, sys
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

import uproot

parser = argparse.ArgumentParser(
    description="Verify hadd'd array-output root files (see condor_lxplus/haddoutputs.py): for "
    "every {syst}/{samp} dataset found under <output_dir>/arrays_*/, check that "
    "<output_dir>/hadd/{syst}/{samp}.root exists, can be opened, and merged every expected input "
    "chunk file. Each per-chunk file writes exactly one entry to its TotalEventCount tree "
    "(see array_writer.py), so comparing the merged file's TotalEventCount entry count against "
    "the number of chunk files on disk catches files silently dropped by hadd's -k "
    "skip-corrupt-file flag, or datasets whose condor jobs haven't all landed yet."
)
parser.add_argument("output_dir", help="Same output_dir passed to haddoutputs.py")
parser.add_argument(
    "-j", "--jobs", type=int, default=8, help="Number of files to check in parallel (default 8)"
)
args = parser.parse_args()

outputdir = args.output_dir

systlist = sorted(set(i.split("/")[-1] for i in glob(f"{outputdir}/arrays_*/*")))

datasets = []  # (syst, samp, expected_files)
for syst in systlist:
    samplist = sorted(set(i.split("/")[-1] for i in glob(f"{outputdir}/arrays_*/{syst}/*")))
    for samp in samplist:
        expected_files = glob(f"{outputdir}/arrays_*/{syst}/{samp}/*.root")
        datasets.append((syst, samp, expected_files))

if len(datasets) == 0:
    print(f"Found zero datasets under {outputdir}/arrays_*/ -- are you pointing at the right output_dir?")
    exit()


def check_one(syst, samp, expected_files):
    hadd_file = f"{outputdir}/hadd/{syst}/{samp}.root"
    n_expected = len(expected_files)

    if not os.path.exists(hadd_file):
        return (
            syst, samp, "MISSING",
            f"{hadd_file} does not exist (expected {n_expected} input files to be merged into it)",
        )

    try:
        with uproot.open(hadd_file) as f:
            if "TotalEventCount" not in f:
                return (syst, samp, "CORRUPT", f"{hadd_file} has no TotalEventCount tree")
            n_merged = f["TotalEventCount"].num_entries
    except Exception as e:
        return (syst, samp, "CORRUPT", f"{hadd_file} could not be opened/read: {e}")

    if n_merged != n_expected:
        return (
            syst, samp, "INCOMPLETE",
            f"{hadd_file}: merged {n_merged}/{n_expected} input files "
            "(some condor jobs may still be running, or hadd's -k flag skipped an unreadable file)",
        )

    return (syst, samp, "OK", f"{hadd_file}: merged {n_merged}/{n_expected} input files")


results = []
with ThreadPoolExecutor(max_workers=args.jobs) as pool:
    futures = [pool.submit(check_one, *d) for d in datasets]
    for fut in as_completed(futures):
        results.append(fut.result())

results.sort(key=lambda r: (r[0], r[1]))

counts = {"OK": 0, "MISSING": 0, "CORRUPT": 0, "INCOMPLETE": 0}
for syst, samp, status, msg in results:
    counts[status] += 1
    if status != "OK":
        print(f"[{status}] {syst}/{samp}: {msg}")

print()
print(
    f"Checked {len(results)} datasets: {counts['OK']} OK, {counts['MISSING']} missing, "
    f"{counts['CORRUPT']} corrupt, {counts['INCOMPLETE']} incomplete"
)

if counts["OK"] != len(results):
    sys.exit(1)
