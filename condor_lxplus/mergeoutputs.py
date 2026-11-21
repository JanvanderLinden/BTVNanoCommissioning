#!/usr/bin/env python
"""
Merge the outputs of one condor submission (condor_lxplus/submitter.py --outputDir) into a few
files, so that the per-job files can be removed from EOS:

    <outputDir>/hists_<n>/hists_<n>.coffea      -> <outputDir>/hists_<name>.coffea
    <outputDir>/arrays_*/<syst>/<dataset>/*.root -> <outputDir>/hadd/<syst>/<dataset>.root

<name> is the base name of outputDir (--name). The array layout under hadd/ is the one of
haddoutputs.py, so checkHadd.py, normalizeArrays.py (input <outputDir>/hadd) and haddFinal.py
work on the result as before.

--split-hists: one file per histogram instead of hists_<name>.coffea, so that no step has to
hold all histograms in memory:

    <outputDir>/hists/<histogram>.coffea   {dataset: {<histogram>: hist, sumw, sumw_<v>}}
    <outputDir>/hists/_other.coffea        {dataset: {all entries that are not histograms}}
    <outputDir>/hists/index.json           {"keys": {<histogram>: {file, axes, datasets}}}

The coffea files are merged in chunks of --hist-files-per-chunk files (each chunk written as
one file per histogram to the scratch directory), then every histogram on its own: the
memory use is that of one chunk or of one merged histogram, whichever is larger.

EOS handling (lxplus storage guidance):
- the files are listed once on the EOS server (`eos find`), not by globbing the /eos mount;
- inputs are read as root:// URLs;
- merged files are written to a local scratch directory (--tmpdir) and copied to EOS once
  with xrdcp; the per-job inputs are removed only with --delete-inputs, after every merged
  file was checked, with `eos rm -r` (server side).

Checks: the merged coffea file(s) hold every histogram of every input file; every merged ROOT
file has as many TotalEventCount entries as it has input files (one per input file, see
array_writer.py). Nothing is copied or deleted if a check fails.

hadd: from the environment, else from the ROOT installation set up by --root-setup (an LCG
view setup script, sourced only in the hadd subprocess). The merge uses -f102 (full
recompression, see haddFinal.py: hadd's fast cloning has corrupted jagged branches written by
uproot) and no -k, so unreadable input files make the merge fail instead of being skipped.

Speed: the coffea files are added in place (no copy of the growing sum per file) and can be
read by several processes (--hist-workers); the array groups are merged by --jobs parallel
hadd processes, each with --hadd-threads (hadd -j). On condor: submit_merge.py, a cascade of
jobs that run this script with --items (merge given files into given targets) and
--finalize (check the targets, write the manifest, remove the intermediate files).

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
import zlib
from collections import defaultdict
import fnmatch
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

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


def eos_find_files(top):
    """All files below `top`, one server-side `eos find` (at most 100k files per call)."""
    res = run(["eos", "find", "-f", top], env=eos_env(top), check=False)
    if res.returncode != 0:
        raise RuntimeError(
            f"eos find -f {top} failed ({res.returncode}): {res.stderr.strip()}"
            + (" -- more than 100k files in one job directory?" if res.returncode == 7 else "")
        )
    out = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if line.startswith("path="):
            line = line[len("path=") :].split(" ")[0]
        if line:
            out.append(canonical_eos(line))
    return out


def eos_ls(top):
    """(directories, files) directly in `top`, from `eos ls -l`."""
    res = run(["eos", "ls", "-l", top], env=eos_env(top))
    dirs, files = [], []
    for line in res.stdout.splitlines():
        fields = line.split()
        if len(fields) < 2 or fields[-1] in (".", ".."):
            continue
        (dirs if line.startswith("d") else files).append(f"{top}/{fields[-1]}")
    return dirs, files


def list_files(top, workers=16):
    """All files below `top`. On EOS: `eos ls` of `top`, then one `eos find` per
    subdirectory in parallel (eos find returns at most 100k files per call); elsewhere
    os.walk."""
    if eos_instance(top) and shutil.which("eos"):
        dirs, files = eos_ls(top)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for found in pool.map(eos_find_files, dirs):
                files.extend(found)
        return [canonical_eos(f) for f in files]
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
    elif eos_instance(path):
        # no eos client (batch nodes): files and directories one by one with xrdfs
        host, rest = xrootd_url(path)[len("root://"):].split("/", 1)
        top = "/" + rest.lstrip("/")
        server = f"root://{host}"
        listing = run(["xrdfs", server, "ls", "-l", "-R", top]).stdout.splitlines()
        files, dirs = [], []
        for line in listing:
            entry = line.split()[-1] if line.strip() else ""
            if not entry:
                continue
            (dirs if line.startswith("d") else files).append(entry)
        for f in files:
            run(["xrdfs", server, "rm", f])
        for d in sorted(dirs, key=len, reverse=True) + [top]:
            run(["xrdfs", server, "rmdir", d])
    else:
        shutil.rmtree(path)


def remote_rm(paths, workers=8):
    """Remove files (xrdfs rm on EOS: no eos client on the batch nodes)."""
    def rm(path):
        if eos_instance(path):
            host, rest = xrootd_url(path)[len("root://"):].split("/", 1)
            run(["xrdfs", f"root://{host}", "rm", "/" + rest.lstrip("/")], check=False)
        elif os.path.exists(path):
            os.remove(path)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(rm, paths))


def copy_many(pairs, workers=8):
    """copy_to for [(local, dest)] in parallel threads"""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda lp: copy_to(*lp), pairs))


def read_json(path):
    if eos_instance(path):
        import fsspec

        with fsspec.open(xrootd_url(path), "r") as fin:
            return json.load(fin)
    with open(path) as fin:
        return json.load(fin)


def write_json(obj, dest, tmpdir):
    local = os.path.join(tmpdir, f"json_{os.getpid()}_{time.time_ns()}.json")
    with open(local, "w") as f:
        json.dump(obj, f, indent=1)
    copy_to(local, dest)
    os.remove(local)


# ------------------------------------------------------------------------------------------
# inputs
# ------------------------------------------------------------------------------------------


def find_inputs(output_dir, all_files):
    """hist files {job dir: file}, array files {group "<syst>/<dataset>": [files]} and the
    array directories arrays_* (for deletion), from the file list of output_dir."""
    hists, arrays, array_dirs = {}, defaultdict(list), set()
    for f in all_files:
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


def _merge_chunk(files, local_out=None, label=""):
    """Sum of the coffea outputs `files`, added in place into the first one (coffea
    accumulate with an accumulator: no copy of the growing sum per file). Returns the sum
    and the (sample, key) entries of the inputs; with local_out, the sum is saved there
    instead (worker processes) and (local_out, keys) is returned."""
    from coffea.processor import accumulate
    from coffea.util import save

    merged, keys = None, defaultdict(set)
    t0 = time.time()
    for i, f in enumerate(files):
        out = load_coffea(f)
        for s, k in count_hists(out).items():
            keys[s] |= k
        merged = out if merged is None else accumulate([out], merged)
        if (i + 1) % 25 == 0:
            print(f"  hists{label}: {i + 1}/{len(files)} files ({time.time() - t0:.0f} s)",
                  flush=True)
    if local_out is None:
        return merged, keys
    save(merged, local_out)
    return local_out, keys


def merge_hists(files, local_out, workers=1, tmpdir=None):
    """Merge the coffea files into local_out (one file). Returns the number of samples."""
    from coffea.util import save

    merged, keys = merge_in_memory(files, workers, tmpdir)
    save(merged, local_out)
    return len(keys)


def merge_in_memory(files, workers=1, tmpdir=None):
    """Sum of the coffea files and the (sample, key) entries of the inputs. With workers > 1
    the files are split into `workers` chunks, merged in parallel processes (reading and
    decompressing is the slow part) and the partial sums are added at the end; the memory
    use grows accordingly."""
    from coffea.processor import accumulate

    files = sorted(files)
    workers = max(1, min(workers, len(files) // 2 or 1))
    keys = defaultdict(set)
    if workers == 1:
        merged, keys = _merge_chunk(files)
    else:
        chunks = [files[i::workers] for i in range(workers)]
        parts = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_merge_chunk, c, os.path.join(tmpdir, f"partial_{i}.coffea"),
                            f" [{i + 1}/{workers}]")
                for i, c in enumerate(chunks)
            ]
            for fut in futures:
                path, k = fut.result()
                parts.append(path)
                for s, kk in k.items():
                    keys[s] |= kk
        merged = None
        for path in parts:
            out = load_coffea(path)
            merged = out if merged is None else accumulate([out], merged)
            os.remove(path)
    for s, k in keys.items():
        missing = k - set(merged.get(s, {}))
        if missing:
            raise RuntimeError(f"merged histograms of {s} miss {sorted(missing)[:5]}...")
    return merged, keys


# ------------------------------------------------------------------------------------------
# one file per histogram (--split-hists)
# ------------------------------------------------------------------------------------------

HIST_DIR = "hists"
OTHER = "_other"
INDEX = "index.json"


def key_file(key):
    """file name of a histogram key"""
    return key.replace("/", "__") + ".coffea"


def key_group(key, ngroups):
    """stage-2 job of a key (stable over processes, unlike hash())"""
    return zlib.crc32(key.encode()) % ngroups


def split_output(out):
    """{key: {dataset: {key: hist, sumw, sumw_<v>}}} of the histograms and
    {dataset: {other entries}} of everything else (sums of weights, lumi lists, ...), and
    the sums of weights {dataset: {sumw, sumw_<v>}} (numbers). The sums in the histogram
    files are replaced by the totals over all parts when the parts are merged
    (merge_key_files): a part without a histogram of a dataset still counts for its sums."""
    import hist

    per_key, other, sums = defaultdict(dict), {}, {}
    for ds, d in out.items():
        if not isinstance(d, dict):
            other[ds] = d
            continue
        sums[ds] = {k: v for k, v in d.items()
                    if isinstance(k, str) and (k == "sumw" or k.startswith("sumw_"))}
        for k, v in d.items():
            if isinstance(v, hist.Hist):
                if key_file(k) == key_file(OTHER):
                    raise RuntimeError(f"histogram name {k} is reserved")
                per_key[k][ds] = dict(sums[ds], **{k: v})
            else:
                other.setdefault(ds, {})[k] = v
    return per_key, other, sums


def _number(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def total_sums(infos):
    """{dataset: {sumw, sumw_<v>}} summed over the key lists of all parts"""
    tot = defaultdict(dict)
    for info in infos:
        for ds, sw in info.get("sums", {}).items():
            for k, v in sw.items():
                tot[ds][k] = tot[ds].get(k, 0.0) + v
    return tot


def hist_datasets(d, key):
    """datasets of a per-histogram output that have the histogram"""
    return sorted(ds for ds, x in d.items() if isinstance(x, dict) and key in x)


def save_split(merged, local_dir):
    """Write `merged` as one file per histogram (+ _other) into local_dir. Returns the key
    list {"hists": {key: [datasets]}, "other": {dataset: [keys]}, "files": {key: file}}."""
    from coffea.util import save

    os.makedirs(local_dir, exist_ok=True)
    per_key, other, sums = split_output(merged)
    info = {"hists": {}, "other": {}, "files": {}, "axes": {}, "sums": {}}
    for ds, sw in sums.items():
        num = {k: _number(v) for k, v in sw.items()}
        info["sums"][ds] = {k: v for k, v in num.items() if v is not None}
    for k, d in per_key.items():
        save(d, os.path.join(local_dir, key_file(k)))
        info["hists"][k] = hist_datasets(d, k)
        info["files"][k] = key_file(k)
        info["axes"][k] = list(d[info["hists"][k][0]][k].axes.name)
    save(other, os.path.join(local_dir, key_file(OTHER)))
    info["files"][OTHER] = key_file(OTHER)
    info["other"] = {ds: sorted(map(str, d)) if isinstance(d, dict) else []
                     for ds, d in other.items()}
    return info


def merge_key_files(key, paths, local_out, sums=None):
    """Sum of the files of one histogram key -> local_out, with the sums of weights of the
    datasets set to `sums` (totals over all parts). Returns {axes, datasets}."""
    import hist
    from coffea.processor import accumulate
    from coffea.util import save

    merged = None
    for p in paths:
        out = load_coffea(p)
        merged = out if merged is None else accumulate([out], merged)
    if key != OTHER:
        for ds, d in merged.items():
            if isinstance(d, dict) and ds in (sums or {}):
                d.update(sums[ds])
    save(merged, local_out)
    if key == OTHER:
        return {"axes": None, "datasets": sorted(merged)}
    dss = hist_datasets(merged, key)
    axes = list(merged[dss[0]][key].axes.name) if dss else None
    if axes is None or not isinstance(merged[dss[0]][key], hist.Hist):
        raise RuntimeError(f"{local_out}: no histogram {key}")
    return {"axes": axes, "datasets": dss}


def _chunk_to_parts(files, part_dir):
    """worker: merge a chunk of coffea files, write it per histogram into part_dir"""
    merged, keys = merge_in_memory(files)
    info = save_split(merged, part_dir)
    info["inputs"] = {s: sorted(map(str, k)) for s, k in keys.items()}
    return info


def _merge_key(key, paths, local_out, sums):
    return key, merge_key_files(key, paths, local_out, sums)


def check_split(input_keys, index):
    """every (dataset, key) entry of the inputs is in the per-histogram files"""
    have = defaultdict(set)
    for k, meta in index["keys"].items():
        for ds in meta["datasets"]:
            have[ds].add(k)
    for ds, keys in index.get("other", {}).items():
        have[ds] |= set(keys)
    for ds, keys in input_keys.items():
        missing = set(keys) - have[ds]
        if missing:
            raise RuntimeError(f"merged histograms of {ds} miss {sorted(missing)[:5]}...")


def merge_hists_split(files, local_dir, chunk_files=20, workers=1, tmpdir=None):
    """Merge the coffea files into one file per histogram in local_dir (+ index.json):
    chunks of chunk_files files, each written per histogram to the scratch directory
    (`workers` chunks in parallel), then each histogram from its chunk files. Memory: one
    chunk or one merged histogram per worker."""
    files = sorted(files)
    chunks = [files[i : i + chunk_files] for i in range(0, len(files), chunk_files)]
    parts_dir = os.path.join(tmpdir, "split_parts")
    infos = []
    workers = max(1, min(workers, len(chunks)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_chunk_to_parts, c, os.path.join(parts_dir, f"part{i}"))
                for i, c in enumerate(chunks)]
        for i, fut in enumerate(futs):
            infos.append((os.path.join(parts_dir, f"part{i}"), fut.result()))
            print(f"  hists: chunk {i + 1}/{len(chunks)} done", flush=True)
    index, input_keys = merge_parts(infos, local_dir, workers)
    check_split(input_keys, index)
    shutil.rmtree(parts_dir, ignore_errors=True)
    with open(os.path.join(local_dir, INDEX), "w") as f:
        json.dump(index, f, indent=1)
    return index


def merge_parts(infos, local_dir, workers=1, keep_parts=False):
    """per histogram key: merge the files of the parts [(part dir, key list)] into
    local_dir/<key file>; returns (index, input keys)"""
    os.makedirs(local_dir, exist_ok=True)
    keys = sorted({k for _, info in infos for k in info["files"] if k != OTHER})
    input_keys = defaultdict(set)
    other = defaultdict(set)
    for _, info in infos:
        for ds, k in info.get("inputs", {}).items():
            input_keys[ds] |= set(k)
        for ds, k in info["other"].items():
            other[ds] |= set(k)
    todo = [(k, [os.path.join(d, info["files"][k]) for d, info in infos if k in info["files"]])
            for k in keys]
    todo.append((OTHER, [os.path.join(d, info["files"][OTHER]) for d, info in infos]))
    index = {"keys": {}, "other": {ds: sorted(k) for ds, k in other.items()},
             "other_file": key_file(OTHER)}
    sums = total_sums([info for _, info in infos])
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = [pool.submit(_merge_key, k, paths, os.path.join(local_dir, key_file(k)),
                            {ds: sums[ds] for ds in sums})
                for k, paths in todo]
        for fut in as_completed(futs):
            k, meta = fut.result()
            if k != OTHER:
                index["keys"][k] = dict(meta, file=key_file(k))
    if not keep_parts:
        for _, paths in todo:
            for p in paths:
                if os.path.exists(p):
                    os.remove(p)
    return index, {ds: sorted(k) for ds, k in input_keys.items()}


def hadd_command(target, inputs, root_setup, threads=1):
    cmd = ["hadd", "-v", "0", HADD_COMPRESSION]
    if threads > 1 and len(inputs) > 2 * threads:
        # hadd's own parallel merging (sub-processes on chunks of the inputs, then one merge)
        cmd += ["-j", str(threads)]
    cmd += ["-f", target] + [xrootd_url(f) for f in inputs]
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


def merge_group(group, inputs, tmpdir, root_setup, threads=1):
    local = os.path.join(tmpdir, "hadd", group + ".root")
    os.makedirs(os.path.dirname(local), exist_ok=True)
    res = subprocess.run(hadd_command(local, inputs, root_setup, threads),
                         capture_output=True, text=True,
                         cwd=os.path.dirname(local))  # hadd -j writes its chunks to the cwd
    if res.returncode != 0:
        raise RuntimeError(f"hadd {group} failed:\n{(res.stderr or res.stdout).strip()[-2000:]}")
    check_merged_root(local, len(inputs))
    return local


def hadd_to(local, inputs, root_setup, threads=1, expected=None):
    """hadd inputs -> local, check of the TotalEventCount entries (expected: number of
    per-job files merged into it, default len(inputs))"""
    os.makedirs(os.path.dirname(local), exist_ok=True)
    res = subprocess.run(hadd_command(local, inputs, root_setup, threads),
                         capture_output=True, text=True, cwd=os.path.dirname(local))
    if res.returncode != 0:
        raise RuntimeError(f"hadd -> {local} failed:\n{(res.stderr or res.stdout).strip()[-2000:]}")
    check_merged_root(local, expected if expected is not None else len(inputs))
    return local


def root_complete(target, expected):
    try:
        check_merged_root(target, expected)
        return True
    except Exception:
        return False


def remote_exists(path):
    if not eos_instance(path):
        return os.path.exists(path)
    url = xrootd_url(path)
    host, rest = url[len("root://"):].split("/", 1)
    return run(["xrdfs", f"root://{host}", "stat", "/" + rest.lstrip("/")], check=False).returncode == 0


def run_items(items, args, tmpdir):
    """Cascade merge step (submit_merge.py): every item {type: coffea|root, target, inputs,
    entries} merges its inputs into its target. ROOT items run in parallel (--jobs), coffea
    items one after the other (memory). Targets that exist and are complete (ROOT:
    TotalEventCount entries) are skipped unless -f, so a retried job continues."""
    errors = []

    def do_root(i, it):
        if not args.force and root_complete(it["target"], it["entries"]):
            return f"exists, complete: {it['target']}"
        local = os.path.join(tmpdir, "items", f"{i}.root")
        hadd_to(local, it["inputs"], args.root_setup, args.hadd_threads, it["entries"])
        copy_to(local, it["target"])
        os.remove(local)
        return f"merged {len(it['inputs'])} files -> {it['target']}"

    roots = [(i, it) for i, it in enumerate(items) if it["type"] == "root"]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(do_root, i, it): it for i, it in roots}
        for fut in as_completed(futures):
            try:
                print(fut.result(), flush=True)
            except Exception as err:
                errors.append(f"{futures[fut]['target']}: {err}")
    for i, it in enumerate(items):
        if it["type"] == "root":
            continue
        try:
            if it["type"] == "coffea":
                local = os.path.join(tmpdir, f"item_{i}.coffea")
                merge_hists(it["inputs"], local, args.hist_workers, tmpdir)
                copy_to(local, it["target"])
                os.remove(local)
            elif it["type"] == "coffea_split":
                # all inputs in one job: directly the final per-histogram files
                merged, keys = merge_in_memory(it["inputs"], args.hist_workers, tmpdir)
                write_split(merged, keys, os.path.join(tmpdir, f"item_{i}"), it["target"])
            elif it["type"] == "coffea_resplit":
                # an existing single merged file (hists_<name>.coffea) -> per-histogram files
                merged = load_coffea(it["inputs"][0])
                write_split(merged, count_hists(merged), os.path.join(tmpdir, f"item_{i}"),
                            it["target"])
            elif it["type"] == "coffea_parts":
                # stage 1: a chunk of the inputs, one file per histogram + keys.json
                local = os.path.join(tmpdir, f"item_{i}")
                merged, keys = merge_in_memory(it["inputs"], args.hist_workers, tmpdir)
                info = save_split(merged, local)
                del merged
                info["inputs"] = {s: sorted(map(str, k)) for s, k in keys.items()}
                with open(os.path.join(local, "keys.json"), "w") as f:
                    json.dump(info, f, indent=1)
                copy_many([(os.path.join(local, f), f"{it['target']}/{f}")
                           for f in sorted(os.listdir(local)) if f != "keys.json"])
                # last: keys.json marks the part as complete
                copy_to(os.path.join(local, "keys.json"), f"{it['target']}/keys.json")
                shutil.rmtree(local, ignore_errors=True)
            elif it["type"] == "coffea_keys":
                merge_key_group(it, tmpdir)
            print(f"merged {len(it.get('inputs', it.get('parts', [])))} files -> "
                  f"{it['target']}", flush=True)
        except Exception as err:
            errors.append(f"{it['target']}: {err}")
    return errors


def write_split(merged, keys, local, target):
    """`merged` (all histograms) -> one file per histogram + index.json in target; `keys`:
    the (dataset, key) entries that have to be there. `merged` is emptied (memory)."""
    info = save_split(merged, local)
    merged.clear()
    index = {"keys": {}, "other": info["other"], "other_file": key_file(OTHER)}
    for k, dss in info["hists"].items():
        index["keys"][k] = {"file": info["files"][k], "datasets": dss, "axes": info["axes"][k]}
    check_split({s: sorted(map(str, k)) for s, k in keys.items()}, index)
    with open(os.path.join(local, INDEX), "w") as f:
        json.dump(index, f, indent=1)
    copy_many([(os.path.join(local, f), f"{target}/{f}")
               for f in sorted(os.listdir(local)) if f != INDEX])
    # last: index.json marks the directory as complete
    copy_to(os.path.join(local, INDEX), f"{target}/{INDEX}")
    shutil.rmtree(local, ignore_errors=True)


def merge_key_group(it, tmpdir):
    """stage 2 (--split-hists): the histogram keys of group it["group"] (key_group) from
    the stage-1 parts into it["target"]/<key file>, one key at a time; per key a small json
    {axes, datasets} in it["meta"] (for the index of the final step), written after the
    merged file: a retried job skips the keys with a json. The part files of a key are
    removed once it is done (the rest of the parts directory goes in the final step)."""
    infos = [(d, read_json(f"{d}/keys.json")) for d in it["parts"]]
    keys = sorted({k for _, info in infos for k in info["files"]})
    mine = [k for k in keys if key_group(k, it["ngroups"]) == it["group"]]
    sums = total_sums([info for _, info in infos])
    for k in mine:
        paths = [f"{d}/{info['files'][k]}" for d, info in infos if k in info["files"]]
        target = f"{it['target']}/{key_file(k)}"
        meta_path = f"{it['meta']}/{key_file(k)}.json"
        if remote_exists(meta_path):
            print(f"  {k}: done earlier, skipped", flush=True)
            continue
        t0 = time.time()
        local = os.path.join(tmpdir, key_file(k))
        meta = merge_key_files(k, paths, local, sums)
        copy_to(local, target)
        os.remove(local)
        write_json(meta, meta_path, tmpdir)
        remote_rm(paths)
        print(f"  {k}: {len(paths)} parts merged ({time.time() - t0:.0f} s)", flush=True)


def finalize(spec, tmpdir):
    """Last cascade step: check every final target, write the manifest (targets -> per-job
    inputs, as a merge with this script), remove the intermediate directory."""
    errors = []
    split = spec.get("split")
    if split:
        try:
            if split.get("parts"):
                build_split_index(split, tmpdir)
            elif not remote_exists(f"{split['hists_dir']}/{INDEX}"):
                errors.append(f"{split['hists_dir']}/{INDEX}: missing")
        except Exception as err:
            errors.append(f"{split['hists_dir']}: {err}")
    for target, entries in spec["checks"].items():
        if entries is None:
            if not remote_exists(target):
                errors.append(f"{target}: missing")
        elif not root_complete(target, entries):
            errors.append(f"{target}: missing or incomplete (expected {entries} input files)")
    if errors:
        return errors
    stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_cascade_{os.getpid()}"
    local = os.path.join(tmpdir, f"merge_manifest_{stamp}.json")
    with open(local, "w") as f:
        json.dump({"output_dir": spec["output_dir"], "outputs": spec["outputs"]}, f, indent=1)
    copy_to(local, f"{spec['output_dir']}/merge_manifest_{stamp}.json")
    if spec.get("tmp") and remote_exists(spec["tmp"]):
        remove_dir(spec["tmp"])
        print(f"removed {spec['tmp']}")
    print(f"{len(spec['checks'])} merged files complete, manifest written")
    return []


def build_split_index(split, tmpdir):
    """final step of a split merge: check the per-histogram files against the key lists of
    the stage-1 parts and write hists/index.json"""
    hdir = split["hists_dir"]
    infos = [read_json(f"{d}/keys.json") for d in split["parts"]]
    keys = sorted({k for info in infos for k in info["files"] if k != OTHER})
    index = {"keys": {}, "other": defaultdict(set), "other_file": key_file(OTHER)}
    input_keys = defaultdict(set)
    for info in infos:
        for ds, k in info["other"].items():
            index["other"][ds] |= set(k)
        for ds, k in info.get("inputs", {}).items():
            input_keys[ds] |= set(k)
    index["other"] = {ds: sorted(k) for ds, k in index["other"].items()}
    missing = []
    for k in keys + [OTHER]:
        target = f"{hdir}/{key_file(k)}"
        if not remote_exists(target):
            missing.append(k)
            continue
        if k != OTHER:
            meta = read_json(f"{split['meta']}/{key_file(k)}.json")
            index["keys"][k] = dict(meta, file=key_file(k))
    if missing:
        raise RuntimeError(f"{len(missing)} merged histogram files missing: {missing[:5]}")
    check_split(input_keys, index)
    write_json(index, f"{hdir}/{INDEX}", tmpdir)
    print(f"{hdir}/{INDEX}: {len(index['keys'])} histograms")


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
                if target.endswith(".coffea") or target.endswith(f"/{HIST_DIR}"):
                    listed.update(inputs)
    return listed


# ------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("output_dir", nargs="?", help="--outputDir of the submission")
    parser.add_argument("--items", default=None,
                        help="cascade merge step: json list of {type, target, inputs, entries}")
    parser.add_argument("--finalize", default=None,
                        help="cascade final step: json {output_dir, outputs, checks, tmp}")
    parser.add_argument("--name", default=None, help="merged histogram file hists_<name>.coffea "
                        "(default: base name of output_dir)")
    parser.add_argument("-j", "--jobs", type=int, default=4, help="parallel hadd processes")
    parser.add_argument("--hadd-threads", type=int, default=1,
                        help="hadd -j N per hadd (for datasets with many files; total load "
                        "--jobs x N processes)")
    parser.add_argument("--hist-workers", type=int, default=1,
                        help="parallel processes reading the coffea files (memory: one "
                        "partial sum per process)")
    parser.add_argument("--split-hists", action="store_true",
                        help="one file per histogram in <output_dir>/hists/ instead of "
                        "hists_<name>.coffea (memory: one chunk or one histogram at a time)")
    parser.add_argument("--hist-files-per-chunk", type=int, default=20,
                        help="--split-hists: coffea files merged per chunk (default: 20)")
    parser.add_argument("--groups", default=None,
                        help="merge only the array groups <syst>/<dataset> matching these "
                        "patterns (split by ',', fnmatch), e.g. 'nominal/TT*'")
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

    if args.items or args.finalize:
        base_tmp = args.tmpdir or os.environ.get("TMPDIR") or f"/tmp/{os.environ.get('USER', 'user')}"
        os.makedirs(base_tmp, exist_ok=True)
        tmpdir = tempfile.mkdtemp(prefix="mergeoutputs_", dir=base_tmp)
        try:
            with open(args.items or args.finalize) as f:
                spec = json.load(f)
            errors = run_items(spec, args, tmpdir) if args.items else finalize(spec, tmpdir)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        if errors:
            print("\nERRORS:")
            for e in errors:
                print("  " + e)
            return 1
        return 0
    if not args.output_dir:
        parser.error("output_dir is required (unless --items / --finalize)")

    output_dir = canonical_eos(args.output_dir.rstrip("/"))
    name = args.name or os.path.basename(output_dir)
    all_files = list_files(output_dir)
    hists, arrays, array_dirs = find_inputs(output_dir, all_files)
    if args.groups:
        pats = [p for p in args.groups.split(",") if p]
        arrays = {g: f for g, f in arrays.items() if any(fnmatch.fnmatch(g, p) for p in pats)}
        if args.delete_inputs:
            raise SystemExit("--groups and --delete-inputs: delete only after merging all groups")
    print(f"{output_dir}: {len(hists)} histogram files, "
          f"{sum(map(len, arrays.values()))} array files in {len(arrays)} datasets")
    if not hists and not arrays:
        print("nothing to merge (no hists_*/*.coffea or arrays_*/... files)")
        return 1

    existing = set(all_files) if not args.force else set()
    base_tmp = args.tmpdir or os.environ.get("TMPDIR") or f"/tmp/{os.environ.get('USER', 'user')}"
    os.makedirs(base_tmp, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="mergeoutputs_", dir=base_tmp)
    manifest = {"output_dir": output_dir, "outputs": {}}
    errors = []

    try:
        # ---------------------------------------------------------------- histograms
        hist_target = f"{output_dir}/hists_{name}.coffea"
        split_index = f"{output_dir}/{HIST_DIR}/{INDEX}"
        if hists and args.only != "arrays" and args.split_hists:
            hdir = f"{output_dir}/{HIST_DIR}"
            if split_index in existing:
                missing = set(hists.values()) - merged_hist_inputs(output_dir, all_files)
                if missing:
                    errors.append(
                        f"{hdir} exists but does not include {len(missing)} histogram files "
                        "(jobs finished after the merge?): rerun with -f"
                    )
                else:
                    print(f"exists, complete, skipped (-f to redo): {hdir}/")
            else:
                local = os.path.join(tmpdir, HIST_DIR)
                try:
                    index = merge_hists_split(list(hists.values()), local,
                                              args.hist_files_per_chunk, args.hist_workers,
                                              tmpdir)
                    copy_many([(os.path.join(local, f), f"{hdir}/{f}")
                               for f in sorted(os.listdir(local)) if f != INDEX])
                    copy_to(os.path.join(local, INDEX), split_index)
                    shutil.rmtree(local, ignore_errors=True)
                    manifest["outputs"][hdir] = sorted(hists.values())
                    print(f"merged {len(hists)} histogram files -> {len(index['keys'])} "
                          f"histogram files in {hdir}/")
                except Exception as err:
                    errors.append(f"histograms: {err}")
        elif hists and args.only != "arrays":
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
                    n_samples = merge_hists(list(hists.values()), local,
                                            args.hist_workers, tmpdir)
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
                    pool.submit(merge_group, g, f, tmpdir, args.root_setup,
                                args.hadd_threads): g
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
            # unique name: several merges of one output directory can run at the same time
            # (condor merge jobs for the histograms and the arrays)
            stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{args.only or 'all'}_{os.getpid()}"
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
