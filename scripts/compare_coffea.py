"""Phase 0 regression check (docs/modernization_plan.md).

Bin-by-bin diff of two directory trees produced by `runner.py`: `.coffea` histogram
files and, if `--isArray` was used, the per-jet ROOT ntuples from `array_writer()`.
Walks <baseline_dir> for `*.coffea`/`*.root` files, finds the file at the same
relative path under <current_dir>, and reports every difference found. Exits
non-zero if anything differs, so it can gate a merge.

Usage:
    python scripts/compare_coffea.py <baseline_dir> <current_dir> [--rtol 1e-9] [--atol 1e-12]
"""

import argparse
import itertools
import sys
from pathlib import Path

import awkward as ak
import hist
import numpy as np
import uproot
from coffea.util import load


def _cat_axes(h):
    return [
        ax.name
        for ax in h.axes
        if isinstance(ax, (hist.axis.StrCategory, hist.axis.IntCategory))
    ]


def compare_hist(h1, h2, rtol, atol):
    """Compare two hist.Hist objects. Categorical axes (syst, osss, ...) are matched
    by category label rather than position, since StrCategory(growth=True) axes can
    end up in a different fill order between two otherwise-identical runs."""
    names1 = {ax.name for ax in h1.axes}
    names2 = {ax.name for ax in h2.axes}
    if names1 != names2:
        return [f"axes differ: baseline={sorted(names1)} current={sorted(names2)}"]

    msgs = []
    cat_axes = sorted(set(_cat_axes(h1)) | set(_cat_axes(h2)))
    # IntCategory needs an explicit hist.loc() wrapper for value-based indexing --
    # a bare int in a dict-based __getitem__ is interpreted as a *bin position*,
    # not a category value, and silently selects the wrong bin.
    int_cat_axes = {
        ax.name
        for hh in (h1, h2)
        for ax in hh.axes
        if isinstance(ax, hist.axis.IntCategory)
    }
    cats1 = {a: set(h1.axes[a]) for a in cat_axes}
    cats2 = {a: set(h2.axes[a]) for a in cat_axes}
    for a in cat_axes:
        only1, only2 = cats1[a] - cats2[a], cats2[a] - cats1[a]
        if only1:
            msgs.append(f"axis '{a}': categories only in baseline: {sorted(only1)}")
        if only2:
            msgs.append(f"axis '{a}': categories only in current: {sorted(only2)}")

    common = {a: sorted(cats1[a] & cats2[a]) for a in cat_axes}
    combos = [
        dict(zip(cat_axes, vals))
        for vals in itertools.product(*[common[a] for a in cat_axes])
    ] or [{}]
    for combo in combos:
        indexer = {
            k: (hist.loc(v) if k in int_cat_axes else v) for k, v in combo.items()
        }
        sub1 = h1[indexer] if indexer else h1
        sub2 = h2[indexer] if indexer else h2
        if not hasattr(sub1, "values"):
            # Every axis was categorical and got indexed away -- boost_histogram
            # collapses that to a scalar WeightedSum accumulator, not a Hist.
            if not np.isclose(sub1.value, sub2.value, rtol=rtol, atol=atol):
                msgs.append(
                    f"combo {combo}: value differs (baseline={sub1.value:.6g} current={sub2.value:.6g})"
                )
            if not np.isclose(sub1.variance, sub2.variance, rtol=rtol, atol=atol):
                msgs.append(
                    f"combo {combo}: variance differs (baseline={sub1.variance:.6g} current={sub2.variance:.6g})"
                )
            continue
        v1, v2 = sub1.values(flow=True), sub2.values(flow=True)
        if v1.shape != v2.shape:
            msgs.append(f"combo {combo}: shape mismatch {v1.shape} vs {v2.shape}")
            continue
        if not np.allclose(v1, v2, rtol=rtol, atol=atol, equal_nan=True):
            msgs.append(
                f"combo {combo}: values differ (max abs diff {np.max(np.abs(v1 - v2)):.3g})"
            )
        w1, w2 = sub1.variances(flow=True), sub2.variances(flow=True)
        if not np.allclose(w1, w2, rtol=rtol, atol=atol, equal_nan=True):
            msgs.append(
                f"combo {combo}: variances differ (max abs diff {np.max(np.abs(w1 - w2)):.3g})"
            )
    return msgs


def compare_scalar(v1, v2, rtol, atol):
    try:
        a1, a2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
        if a1.shape != a2.shape or not np.allclose(
            a1, a2, rtol=rtol, atol=atol, equal_nan=True
        ):
            return f"differ: baseline={v1!r} current={v2!r}"
    except (TypeError, ValueError):
        if v1 != v2:
            return f"differ: baseline={v1!r} current={v2!r}"
    return None


def compare_coffea_file(baseline_path, current_path, rtol, atol):
    problems = []
    base = load(str(baseline_path))
    curr = load(str(current_path))
    for ds in sorted(set(base) | set(curr)):
        if ds not in base or ds not in curr:
            only = "current" if ds not in base else "baseline"
            problems.append(f"[{ds}] dataset only present in {only}")
            continue
        if not hasattr(base[ds], "keys") or not hasattr(curr[ds], "keys"):
            # BTA_producer's accumulator is {dataset: <event count>} -- a bare
            # scalar per dataset, not a {key: hist/array} dict like every
            # other workflow.
            msg = compare_scalar(base[ds], curr[ds], rtol, atol)
            if msg:
                problems.append(f"[{ds}] {msg}")
            continue
        for key in sorted(set(base[ds]) | set(curr[ds])):
            if key not in base[ds] or key not in curr[ds]:
                only = "current" if key not in base[ds] else "baseline"
                problems.append(f"[{ds}][{key}] key only present in {only}")
                continue
            v1, v2 = base[ds][key], curr[ds][key]
            if hasattr(v1, "axes") and hasattr(v2, "axes"):
                problems += [
                    f"[{ds}][{key}] {m}" for m in compare_hist(v1, v2, rtol, atol)
                ]
            elif key == "fname":
                # Which input files contributed -- provenance, not physics content.
                if set(v1) != set(v2):
                    problems.append(
                        f"[{ds}][fname] input file set changed (informational): "
                        f"baseline={sorted(v1)} current={sorted(v2)}"
                    )
            elif key in ("run", "lumi"):
                if not np.array_equal(np.sort(v1.value), np.sort(v2.value)):
                    problems.append(f"[{ds}][{key}] processed run/lumi set changed")
            else:
                msg = compare_scalar(v1, v2, rtol, atol)
                if msg:
                    problems.append(f"[{ds}][{key}] {msg}")
    return problems


def _flat_values(arr):
    """Flatten a branch (possibly jagged) to a plain numpy array for comparison."""
    try:
        counts = ak.num(arr, axis=1)
    except ValueError:
        counts = None
    flat = ak.to_numpy(ak.flatten(arr, axis=None))
    return counts, flat


def compare_root_file(baseline_path, current_path, rtol, atol):
    problems = []
    with uproot.open(baseline_path) as fb, uproot.open(current_path) as fc:
        keys_b = {k.split(";")[0] for k in fb.keys()}
        keys_c = {k.split(";")[0] for k in fc.keys()}
        for key in sorted(keys_b | keys_c):
            if key not in keys_b or key not in keys_c:
                only = "current" if key not in keys_b else "baseline"
                problems.append(f"[{key}] tree/object only present in {only}")
                continue
            tb, tc = fb[key], fc[key]
            if not hasattr(tb, "num_entries"):
                # not a TTree -- e.g. a histogram, or (BTA_producer's output
                # nests its tree in a "btagana" TDirectory) a directory, which
                # also has .keys() and would otherwise be misread as branches.
                continue
            branches = set(tb.keys()) | set(tc.keys())
            for br in sorted(branches):
                if br not in tb.keys() or br not in tc.keys():
                    only = "current" if br not in tb.keys() else "baseline"
                    problems.append(f"[{key}/{br}] branch only present in {only}")
                    continue
                ab, ac = tb[br].array(library="ak"), tc[br].array(library="ak")
                cb, fb_flat = _flat_values(ab)
                cc, fc_flat = _flat_values(ac)
                if (cb is None) != (cc is None) or (
                    cb is not None and not ak.all(cb == cc)
                ):
                    problems.append(f"[{key}/{br}] per-event jet/object counts differ")
                    continue
                if fb_flat.shape != fc_flat.shape:
                    problems.append(
                        f"[{key}/{br}] shape mismatch {fb_flat.shape} vs {fc_flat.shape}"
                    )
                    continue
                try:
                    same = np.allclose(
                        fb_flat, fc_flat, rtol=rtol, atol=atol, equal_nan=True
                    )
                except TypeError:
                    same = np.array_equal(fb_flat, fc_flat)
                if not same:
                    diff = (
                        np.max(np.abs(fb_flat - fc_flat))
                        if np.issubdtype(fb_flat.dtype, np.number)
                        else "n/a"
                    )
                    problems.append(f"[{key}/{br}] values differ (max abs diff {diff})")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline_dir")
    ap.add_argument("current_dir")
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--atol", type=float, default=1e-12)
    args = ap.parse_args()

    baseline_dir = Path(args.baseline_dir)
    current_dir = Path(args.current_dir)

    all_problems = {}
    for pattern, compare_fn in (
        ("*.coffea", compare_coffea_file),
        ("*.root", compare_root_file),
    ):
        for baseline_file in sorted(baseline_dir.rglob(pattern)):
            rel = baseline_file.relative_to(baseline_dir)
            current_file = current_dir / rel
            if not current_file.exists():
                all_problems[str(rel)] = [f"missing from {current_dir}"]
                continue
            problems = compare_fn(baseline_file, current_file, args.rtol, args.atol)
            if problems:
                all_problems[str(rel)] = problems

        for current_file in sorted(current_dir.rglob(pattern)):
            rel = current_file.relative_to(current_dir)
            if not (baseline_dir / rel).exists():
                all_problems[str(rel)] = [f"missing from {baseline_dir}"]

    if not all_problems:
        print("OK: no differences found.")
        return 0

    print("DIFFERENCES FOUND:\n")
    for rel, problems in all_problems.items():
        print(f"{rel}:")
        for p in problems:
            print(f"  - {p}")
        print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
