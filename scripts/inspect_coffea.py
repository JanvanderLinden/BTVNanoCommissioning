#!/usr/bin/env python
"""
List the content of coffea output files (runner.py, condor job outputs, merged
hists_<name>.coffea) for a quick validation:

  per dataset: sumw (and sumw_<variation>), cutflow, number of histograms, non-histogram
               entries
  per histogram (--hists): axes (name, type, size, syst categories), sum of weights and
               of entries (nominal syst), fraction of the sum in under/overflow

Several files are added up as plotdataMC.py does. Paths can be local, /eos (read as
root:// URLs) or root:// URLs; wildcards are expanded for local and /eos paths.

Examples
--------
    python scripts/inspect_coffea.py hists_e_DY_MC.coffea
    python scripts/inspect_coffea.py "$B/e_DY/*/hists_e_DY_*.coffea" --hists -k "stag*" -d "DY*"
    python scripts/inspect_coffea.py out/hists_3/hists_3.coffea --hists --all-systs
"""

import argparse
import fnmatch
import glob
import math

import hist
import numpy as np


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
        if "://" in p:
            files.append(p)
            continue
        matched = sorted(glob.glob(p))
        if not matched:
            raise SystemExit(f"no files match {p}")
        files += matched
    return files


def merge(outputs):
    """{dataset: {key: object}}, summed over files; a file can also hold several
    {file: {dataset: ...}} levels (as plotdataMC.py accepts)."""
    merged = {}

    def add(ds, d):
        tgt = merged.setdefault(ds, {})
        for k, v in d.items():
            if k not in tgt:
                tgt[k] = v
            else:
                try:
                    tgt[k] = tgt[k] + v
                except TypeError:
                    pass

    for out in outputs:
        for ds, d in out.items():
            if isinstance(d, dict) and d and all(isinstance(x, dict) for x in d.values()):
                for ds2, d2 in d.items():
                    add(ds2, d2)
            elif isinstance(d, dict):
                add(ds, d)
    return merged


def fmt(x):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return str(x)
    return f"{x:.6g}"


def axis_desc(ax):
    kind = type(ax).__name__
    if isinstance(ax, (hist.axis.StrCategory, hist.axis.IntCategory)):
        cats = list(ax)
        shown = ",".join(map(str, cats[:6])) + (",..." if len(cats) > 6 else "")
        return f"{ax.name}[{kind} {len(cats)}: {shown}]"
    return f"{ax.name}[{kind} {ax.size}: {ax.edges[0]:g}..{ax.edges[-1]:g}]"


def hist_summary(h, all_systs):
    """{syst: (sum of weights, sum of entries/variances proxy, flow fraction)}"""
    systs = [None]
    if "syst" in h.axes.name:
        cats = list(h.axes["syst"])
        systs = cats if all_systs else (["nominal"] if "nominal" in cats else cats[:1])
    out = {}
    for s in systs:
        hs = h[{"syst": s}] if s is not None else h
        if not isinstance(hs, hist.Hist):  # only a syst axis
            val = hs.value if hasattr(hs, "value") else float(hs)
            out[s] = (val, None, 0.0)
            continue
        tot = hs.sum(flow=True)
        inner = hs.sum(flow=False)
        tv = tot.value if hasattr(tot, "value") else float(tot)
        iv = inner.value if hasattr(inner, "value") else float(inner)
        var = tot.variance if hasattr(tot, "variance") else None
        # effective number of entries (sum w)^2 / sum w^2
        neff = tv * tv / var if var else None
        out[s] = (tv, neff, (tv - iv) / tv if tv else 0.0)
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("files", nargs="+", help="coffea file(s) or wildcards")
    parser.add_argument("-d", "--dataset", default="*", help="dataset pattern (fnmatch)")
    parser.add_argument("-k", "--key", default="*", help="histogram pattern (fnmatch)")
    parser.add_argument("--hists", action="store_true", help="list the histograms")
    parser.add_argument("--all-systs", action="store_true",
                        help="per histogram all syst categories, not only nominal")
    args = parser.parse_args()

    files = expand(args.files)
    print(f"{len(files)} file(s)")
    merged = merge(load(f) for f in files)
    datasets = sorted(d for d in merged if fnmatch.fnmatch(d, args.dataset))
    if not datasets:
        raise SystemExit(f"no dataset matches '{args.dataset}' (have: {sorted(merged)[:10]})")

    for ds in datasets:
        d = merged[ds]
        hists = {k: v for k, v in d.items() if isinstance(v, hist.Hist)}
        other = {k: v for k, v in d.items() if not isinstance(v, hist.Hist)}
        print(f"\n=== {ds}: {len(hists)} histograms")
        if "sumw" in other:
            print(f"  sumw = {fmt(other['sumw'])}")
        variations = sorted(k for k in other if k.startswith("sumw_"))
        if variations:
            print(f"  sumw_<variation>: {len(variations)} ({', '.join(v[5:] for v in variations[:6])}"
                  f"{', ...' if len(variations) > 6 else ''})")
        rest = sorted(k for k in other if k != "sumw" and not k.startswith("sumw_"))
        if rest:
            print(f"  other entries: {', '.join(rest[:10])}{', ...' if len(rest) > 10 else ''}")
        if "cutflow" in hists and "step" in hists["cutflow"].axes.name:
            cf = hists["cutflow"]
            steps = list(cf.axes["step"])
            vals = [cf[{"step": s}] for s in steps]
            vals = [v.value if hasattr(v, "value") else float(v) for v in vals]
            print("  cutflow: " + ", ".join(f"{s} {fmt(v)}" for s, v in zip(steps, vals)))
        if "syst" in (next(iter(hists.values())).axes.name if hists else ()):
            systs = list(next(iter(hists.values())).axes["syst"])
            print(f"  syst categories ({len(systs)}): {', '.join(systs[:8])}"
                  f"{', ...' if len(systs) > 8 else ''}")
        if not args.hists:
            continue
        keys = sorted(k for k in hists if fnmatch.fnmatch(k, args.key))
        empty = []
        for k in keys:
            h = hists[k]
            summ = hist_summary(h, args.all_systs)
            if all(v[0] == 0 for v in summ.values()):
                empty.append(k)
                continue
            print(f"  {k}: " + " x ".join(axis_desc(a) for a in h.axes if a.name != "syst"))
            for s, (sw, neff, flow) in summ.items():
                tag = f"[{s}] " if s is not None else ""
                print(f"      {tag}sum w {fmt(sw)}, eff. entries {fmt(neff)}"
                      + (f", {100 * flow:.1f}% in under/overflow" if flow > 1e-3 else ""))
        if empty:
            print(f"  empty ({len(empty)}): {', '.join(empty[:15])}{', ...' if len(empty) > 15 else ''}")


if __name__ == "__main__":
    main()
