#!/usr/bin/env python
"""
Split coffea output files into one small file per histogram, for parallel plotting
(condor_lxplus/submit_plots.py): the input files are read once, and every plot job reads
only the file of its variable.

    <outdir>/<variable>.coffea   {dataset: {<variable>: hist, sumw, sumw_<variation>}}

The datasets of all input files are added up as in plotdataMC.py. Only histograms that
plotdataMC.py can draw are written: one (or two) axes left besides syst, flav/jflav, osss
and category. Inputs: local paths, /eos paths (read as root:// URLs) or root:// URLs;
outdir: local or /eos (written locally first, then copied with xrdcp).

Example
-------
    python scripts/split_coffea.py "$B/e_DY/*/hists_e_DY_*.coffea" -o $B/e_DY/plot_split \\
        --list vars_e_DY.txt
"""

import argparse
import fnmatch
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

SPECIAL_AXES = {"syst", "flav", "jflav", "osss", "category"}


def xrootd_url(path):
    if path.startswith("/eos/home-"):
        path = f"/eos/user/{path[10]}/{path[12:]}"
    for prefix, host in (
        ("/eos/user/", "eosuser.cern.ch"),
        ("/eos/cms/", "eoscms.cern.ch"),
        ("/eos/project/", "eosproject.cern.ch"),
    ):
        if path.startswith(prefix):
            return f"root://{host}/{path}"
    return path


def load(path):
    import cloudpickle
    import lz4.frame

    url = xrootd_url(path)
    if url.startswith("root://"):
        import fsspec

        with fsspec.open(url, "rb") as raw, lz4.frame.open(raw) as fin:
            return cloudpickle.load(fin)
    with lz4.frame.open(path) as fin:
        return cloudpickle.load(fin)


def expand(patterns):
    files = []
    for p in patterns:
        for q in p.split(","):
            if not q:
                continue
            if "://" in q:
                files.append(q)
                continue
            matched = sorted(glob.glob(q))
            if not matched:
                raise SystemExit(f"no files match {q}")
            files += matched
    return files


def plottable(h):
    import hist

    if not isinstance(h, hist.Hist):
        return False
    left = [a for a in h.axes.name if a not in SPECIAL_AXES]
    return 1 <= len(left) <= 2


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="coffea files (wildcards, or split by ',')")
    p.add_argument("-o", "--outdir", required=True, help="output directory (local or /eos)")
    p.add_argument("-v", "--variables", default="all",
                   help="histograms, split by ',' (wildcards); default all")
    p.add_argument("--list", default=None, help="write the variable names to this file")
    p.add_argument("-j", "--jobs", type=int, default=8, help="parallel copies to EOS")
    args = p.parse_args()

    from coffea.util import save

    files = expand(args.files)
    merged = {}
    for f in files:
        print(f"reading {f}", flush=True)
        out = load(f)
        for ds, d in out.items():
            if not isinstance(d, dict):
                continue
            tgt = merged.setdefault(ds, {})
            for k, v in d.items():
                if k not in tgt:
                    tgt[k] = v
                else:
                    try:
                        tgt[k] = tgt[k] + v
                    except TypeError:
                        pass
        del out

    keys = sorted({k for d in merged.values() for k, v in d.items() if plottable(v)})
    if args.variables != "all":
        pats = [x for x in args.variables.split(",") if x]
        keys = [k for k in keys if any(fnmatch.fnmatch(k, x) for x in pats)]
    if not keys:
        raise SystemExit("no plottable histogram selected")

    eos = args.outdir.startswith(("/eos/", "root://"))
    local_dir = tempfile.mkdtemp(prefix="split_coffea_") if eos else args.outdir
    os.makedirs(local_dir, exist_ok=True)
    for k in keys:
        sub = {}
        for ds, d in merged.items():
            if k not in d:
                continue
            sub[ds] = {kk: v for kk, v in d.items()
                       if kk == k or kk == "sumw" or (isinstance(kk, str) and kk.startswith("sumw_"))}
        save(sub, os.path.join(local_dir, f"{k}.coffea"))
    print(f"{len(keys)} histograms written to {local_dir}")

    if eos:
        url = xrootd_url(args.outdir.rstrip("/"))
        host, path = url[len("root://"):].split("/", 1)
        subprocess.run(["xrdfs", f"root://{host}", "mkdir", "-p", "/" + path.lstrip("/")],
                       check=True)

        def copy(k):
            for _ in range(3):
                res = subprocess.run(["xrdcp", "-f", "-s", os.path.join(local_dir, f"{k}.coffea"),
                                      f"{url}/{k}.coffea"], capture_output=True, text=True)
                if res.returncode == 0:
                    return None
            return f"{k}: {res.stderr.strip()}"

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            errors = [e for e in pool.map(copy, keys) if e]
        shutil.rmtree(local_dir, ignore_errors=True)
        if errors:
            raise SystemExit("copy to EOS failed:\n  " + "\n  ".join(errors))
        print(f"copied to {url}")

    if args.list:
        with open(args.list, "w") as f:
            f.write("\n".join(keys) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
