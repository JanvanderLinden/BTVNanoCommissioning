#!/usr/bin/env python
"""
Merge the outputs of one condor submission (condor_lxplus/submitter.py --outputDir) into a few
files, so that the per-job files can be removed from EOS:

    <outputDir>/hists_<n>/hists_<n>.coffea      -> <outputDir>/hists_<name>.coffea
    <outputDir>/arrays_*/<syst>/<dataset>/*.root -> <outputDir>/hadd/<syst>/<dataset>.root

<name> is the base name of outputDir (--name). The array layout under hadd/ is the one of
haddoutputs.py, so checkHadd.py, normalizeArrays.py (input <outputDir>/hadd) and haddFinal.py
work on the result as before.

EOS handling (lxplus storage guidance):
- the files are listed once on the EOS server (`eos find`), not by globbing the /eos mount;
- inputs are read as root:// URLs;
- merged files are written to a local scratch directory (--tmpdir) and copied to EOS once
  with xrdcp; the per-job inputs are removed only with --delete-inputs, after every merged
  file was checked, with `eos rm -r` (server side).

Checks: the merged coffea file holds every histogram of every input file; every merged ROOT
file has as many TotalEventCount entries as it has input files (one per input file, see
array_writer.py). Nothing is copied or deleted if a check fails.

hadd: from the environment, else from the ROOT installation set up by --root-setup (an LCG
view setup script, sourced only in the hadd subprocess). The merge uses -f102 (full
recompression, see haddFinal.py: hadd's fast cloning has corrupted jagged branches written by
uproot) and no -k, so unreadable input files make the merge fail instead of being skipped.

Example
-------
    python condor_lxplus/mergeoutputs.py /eos/user/v/vanderli/s/btv/outputs/stag_dy_e -j 8
    # after looking at the summary:
    python condor_lxplus/mergeoutputs.py /eos/user/v/vanderli/s/btv/outputs/stag_dy_e -j 8 \\
        --delete-inputs
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

EOS_INSTANCES = (
    ("/eos/user/", "root://eosuser.cern.ch"),
    ("/eos/cms/", "root://eoscms.cern.ch"),
    ("/eos/project/", "root://eosproject.cern.ch"),
)
DEFAULT_ROOT_SETUP = "/cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc13-opt/setup.sh"
HADD_COMPRESSION = "-f102"


# ------------------------------------------------------------------------------------------
# EOS helpers
# ------------------------------------------------------------------------------------------


def canonical_eos(path):
    """/eos/home-<u>/<user>/... -> /eos/user/<u>/<user>/...; other paths unchanged."""
    path = os.path.abspath(path)
    if path.startswith("/eos/home-"):
        return f"/eos/user/{path[10]}/{path[12:]}"
    return path


def eos_instance(path):
    for prefix, mgm in EOS_INSTANCES:
        if path.startswith(prefix):
            return mgm
    return None


def xrootd_url(path):
    mgm = eos_instance(path)
    return f"{mgm}/{path}" if mgm else path


def run(cmd, env=None, check=True):
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if check and res.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed ({res.returncode}):\n{res.stderr.strip()}")
    return res


def eos_env(path):
    return dict(os.environ, EOS_MGM_URL=eos_instance(path))


def list_files(top):
    """All files below `top`: `eos find` on EOS (one server-side call), os.walk otherwise."""
    if eos_instance(top) and shutil.which("eos"):
        res = run(["eos", "find", "-f", top], env=eos_env(top))
        out = []
        for line in res.stdout.splitlines():
            line = line.strip()
            if line.startswith("path="):
                line = line[len("path=") :].split(" ")[0]
            if line:
                out.append(canonical_eos(line))
        return out
    return [os.path.join(d, f) for d, _, files in os.walk(top) for f in files]


def copy_to(local, dest):
    """Copy a local file to `dest` (xrdcp with 3 attempts for EOS, else cp)."""
    if eos_instance(dest):
        for attempt in range(3):
            res = run(["xrdcp", "-f", "-p", "-N", local, xrootd_url(dest)], check=False)
            if res.returncode == 0:
                return
            time.sleep(30 * (attempt + 1))
        raise RuntimeError(f"xrdcp {local} -> {dest} failed:\n{res.stderr.strip()}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copyfile(local, dest)


def remove_dir(path):
    if eos_instance(path) and shutil.which("eos"):
        run(["eos", "rm", "-r", path], env=eos_env(path))
    else:
        shutil.rmtree(path)


# ------------------------------------------------------------------------------------------
# inputs
# ------------------------------------------------------------------------------------------


def find_inputs(output_dir):
    """hist files {job dir: file}, array files {group "<syst>/<dataset>": [files]} and the
    array directories arrays_* (for deletion)."""
    hists, arrays, array_dirs = {}, defaultdict(list), set()
    for f in list_files(output_dir):
        rel = os.path.relpath(f, output_dir)
        parts = rel.split(os.sep)
        top = parts[0]
        if top.startswith("hists_") and len(parts) == 2 and f.endswith(".coffea"):
            hists[top] = f
        elif top.startswith("arrays_") and len(parts) >= 3 and f.endswith(".root"):
            arrays["/".join(parts[1:-1])].append(f)
            array_dirs.add(top)
    for files in arrays.values():
        files.sort()
    return hists, dict(arrays), sorted(array_dirs)


# ------------------------------------------------------------------------------------------
# merging
# ------------------------------------------------------------------------------------------


def load_coffea(path):
    import cloudpickle
    import lz4.frame

    if eos_instance(path):
        import fsspec

        with fsspec.open(xrootd_url(path), "rb") as raw, lz4.frame.open(raw) as fin:
            return cloudpickle.load(fin)
    with lz4.frame.open(path) as fin:
        return cloudpickle.load(fin)


def count_hists(out):
    """Number of (sample, key) entries, to check that nothing was dropped."""
    return {s: set(v) for s, v in out.items() if isinstance(v, dict)}


def merge_hists(files, local_out):
    from coffea.processor import accumulate
    from coffea.util import save

    merged, keys = None, defaultdict(set)
    for i, f in enumerate(sorted(files)):
        out = load_coffea(f)
        for s, k in count_hists(out).items():
            keys[s] |= k
        merged = out if merged is None else accumulate([merged, out])
        if (i + 1) % 50 == 0:
            print(f"  hists: {i + 1}/{len(files)} files")
    for s, k in keys.items():
        missing = k - set(merged.get(s, {}))
        if missing:
            raise RuntimeError(f"merged histograms of {s} miss {sorted(missing)[:5]}...")
    save(merged, local_out)
    return len(keys)


def hadd_command(target, inputs, root_setup):
    cmd = ["hadd", "-v", "0", HADD_COMPRESSION, "-f", target] + [xrootd_url(f) for f in inputs]
    if shutil.which("hadd"):
        return cmd
    if not root_setup or not os.path.isfile(root_setup):
        raise RuntimeError(
            "hadd not found: run in an environment with ROOT or give --root-setup "
            "(e.g. an LCG view setup.sh)"
        )
    quoted = " ".join(shlex.quote(c) for c in cmd)
    return ["bash", "-c", f"source {root_setup} >/dev/null 2>&1 && {quoted}"]


def check_merged_root(path, n_inputs):
    import uproot

    with uproot.open(xrootd_url(path)) as f:
        if "TotalEventCount" not in f:
            raise RuntimeError(f"{path}: no TotalEventCount tree")
        n = f["TotalEventCount"].num_entries
    if n != n_inputs:
        raise RuntimeError(f"{path}: TotalEventCount has {n} entries, expected {n_inputs}")


def merge_group(group, inputs, tmpdir, root_setup):
    local = os.path.join(tmpdir, "hadd", group + ".root")
    os.makedirs(os.path.dirname(local), exist_ok=True)
    res = subprocess.run(hadd_command(local, inputs, root_setup), capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"hadd {group} failed:\n{(res.stderr or res.stdout).strip()[-2000:]}")
    check_merged_root(local, len(inputs))
    return local


def merged_hist_inputs(output_dir, files):
    """Histogram inputs listed in the merge manifests of output_dir."""
    listed = set()
    for f in files:
        if os.path.basename(f).startswith("merge_manifest_") and f.endswith(".json"):
            if eos_instance(f):
                import fsspec

                with fsspec.open(xrootd_url(f), "r") as fin:
                    manifest = json.load(fin)
            else:
                with open(f) as fin:
                    manifest = json.load(fin)
            for target, inputs in manifest.get("outputs", {}).items():
                if target.endswith(".coffea"):
                    listed.update(inputs)
    return listed


# ------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("output_dir", help="--outputDir of the submission")
    parser.add_argument("--name", default=None, help="merged histogram file hists_<name>.coffea "
                        "(default: base name of output_dir)")
    parser.add_argument("-j", "--jobs", type=int, default=4, help="parallel hadd processes")
    parser.add_argument("--tmpdir", default=None,
                        help="local scratch directory (default: $TMPDIR or /tmp/$USER)")
    parser.add_argument("--root-setup", default=DEFAULT_ROOT_SETUP,
                        help="script setting up ROOT for hadd if hadd is not on the PATH")
    parser.add_argument("--only", choices=["hists", "arrays"], default=None,
                        help="merge only the histograms or only the arrays")
    parser.add_argument("-f", "--force", action="store_true",
                        help="redo merges whose output already exists")
    parser.add_argument("--delete-inputs", action="store_true",
                        help="after all checks passed, remove the merged per-job inputs "
                        "(hists_<n>/, arrays_*/) from output_dir")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="do not ask before deleting")
    args = parser.parse_args()

    output_dir = canonical_eos(args.output_dir.rstrip("/"))
    name = args.name or os.path.basename(output_dir)
    hists, arrays, array_dirs = find_inputs(output_dir)
    print(f"{output_dir}: {len(hists)} histogram files, "
          f"{sum(map(len, arrays.values()))} array files in {len(arrays)} datasets")
    if not hists and not arrays:
        print("nothing to merge (no hists_*/*.coffea or arrays_*/... files)")
        return 1

    all_files = list_files(output_dir)
    existing = set(all_files) if not args.force else set()
    base_tmp = args.tmpdir or os.environ.get("TMPDIR") or f"/tmp/{os.environ.get('USER', 'user')}"
    os.makedirs(base_tmp, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="mergeoutputs_", dir=base_tmp)
    manifest = {"output_dir": output_dir, "outputs": {}}
    errors = []

    try:
        # ---------------------------------------------------------------- histograms
        hist_target = f"{output_dir}/hists_{name}.coffea"
        if hists and args.only != "arrays":
            if hist_target in existing:
                # merged earlier: all current histogram files must be in it
                missing = set(hists.values()) - merged_hist_inputs(output_dir, all_files)
                if missing:
                    errors.append(
                        f"{hist_target} exists but does not include {len(missing)} histogram "
                        "files (jobs finished after the merge?): rerun with -f"
                    )
                else:
                    print(f"exists, complete, skipped (-f to redo): {hist_target}")
            else:
                local = os.path.join(tmpdir, os.path.basename(hist_target))
                try:
                    n_samples = merge_hists(list(hists.values()), local)
                    copy_to(local, hist_target)
                    manifest["outputs"][hist_target] = sorted(hists.values())
                    print(f"merged {len(hists)} histogram files ({n_samples} samples) -> "
                          f"{hist_target}")
                except Exception as err:
                    errors.append(f"histograms: {err}")

        # ---------------------------------------------------------------- arrays
        if arrays and args.only != "hists":
            todo = {}
            for group, files in arrays.items():
                target = f"{output_dir}/hadd/{group}.root"
                if target in existing:
                    # merged earlier: it must hold all current input files
                    try:
                        check_merged_root(target, len(files))
                        print(f"exists, complete, skipped (-f to redo): {target}")
                    except Exception as err:
                        errors.append(f"{group}: {err} (rerun with -f)")
                else:
                    todo[group] = files
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                futures = {
                    pool.submit(merge_group, g, f, tmpdir, args.root_setup): g
                    for g, f in todo.items()
                }
                for fut in as_completed(futures):
                    group = futures[fut]
                    target = f"{output_dir}/hadd/{group}.root"
                    try:
                        local = fut.result()
                        copy_to(local, target)
                        os.remove(local)
                        manifest["outputs"][target] = arrays[group]
                        print(f"merged {len(arrays[group])} files -> {target}")
                    except Exception as err:
                        errors.append(f"{group}: {err}")

        # ---------------------------------------------------------------- manifest
        if manifest["outputs"]:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            local = os.path.join(tmpdir, f"merge_manifest_{stamp}.json")
            with open(local, "w") as f:
                json.dump(manifest, f, indent=1)
            copy_to(local, f"{output_dir}/merge_manifest_{stamp}.json")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if errors:
        print("\nERRORS (nothing deleted):")
        for e in errors:
            print("  " + e)
        return 1

    # -------------------------------------------------------------------- cleanup
    to_delete = []
    if args.only != "arrays":
        to_delete += [f"{output_dir}/{d}" for d in sorted(hists)]
    if args.only != "hists":
        to_delete += [f"{output_dir}/{d}" for d in array_dirs]
    if not args.delete_inputs:
        print(f"\nAll merges OK. To remove the {len(to_delete)} per-job directories "
              "(hists_<n>/, arrays_*/), rerun with --delete-inputs.")
        return 0
    if not args.yes:
        answer = input(f"\nRemove {len(to_delete)} per-job directories from {output_dir}? [y/N] ")
        if answer.strip().lower() != "y":
            print("nothing deleted")
            return 0
    for d in to_delete:
        remove_dir(d)
    print(f"removed {len(to_delete)} per-job directories")
    return 0


if __name__ == "__main__":
    sys.exit(main())
