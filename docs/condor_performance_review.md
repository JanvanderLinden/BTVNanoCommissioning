# Condor Submission Infrastructure Performance Review — BTVNanoCommissioning

Scope: `condor/execute.sh`, `condor/submitter.py`, `condor_lxplus/execute.sh`,
`condor_lxplus/submitter.py`, `scripts/fetch.py`. This is the dedicated follow-up review flagged
in `docs/modernization_plan.md`'s "Post-plan additions" section — the four source reviews
(`architecture_review.md`, `performance_review.md`, `io_review.md`, `simplify_review.md`) all stop
at `runner.py`/`correction.py`/workflow code and never looked at the condor submission layer.
Runtime/CPU/memory concerns only; formatting and pure robustness (error handling, exit codes)
are out of scope except where they compound a performance issue.

Two items were already found and fixed via live triage on the `2D_*_Wc_sf` condor batches
(2026-08-03): `condor/execute.sh`'s unbounded micromamba-installer `wget` (commit `922004c`) and
`scripts/fetch.py` routing CERN-local files through the global XRootD redirector (commit
`5650752`). Not re-reported here as open.

Two more issues (`fetch.py` re-querying DAS per candidate site, and an exhaustive per-URL-variant
XRootD probe) were already caught by `io_review.md` items #3/#4 and are tracked there and in
`modernization_plan.md`'s "below the line" section — also not re-reported as new, though finding
#3 below refines what's still live in that same code path since `fetch.py` has moved on since that
review was written.

---

### 1. `condor/execute.sh` rebuilds the entire Python/conda environment from scratch on every job

**Location:** `condor/execute.sh:36-83` — micromamba bootstrap (`:37-52`), base env creation with
two separate `micromamba install` solves (`:62-65`), then `pip install -e .` plus a standalone
`pip install psutil` (`:79-82`).

**Problem:** Every condor job that runs via `condor/submitter.py` (the `remoteRepo` path, i.e. any
job that doesn't rely purely on the tarball's own tooling) re-runs, from a bare worker node: a
`conda-forge` dependency solve for `python=3.10`+`xrootd`, a second solve for
`setuptools=70.1.1`, then `pip install -e .` which pulls the full `coffea==0.7.31` dependency
tree (`awkward`, `dask`, `uproot`, `cloudpickle`, `numpy`, ...) plus `correctionlib`/`onnxruntime`
from PyPI, then a separate `pip install psutil`. None of this depends on the job's dataset or
sample — it is identical, static work repeated once per job. `condor_lxplus/execute.sh` avoids
this entirely by using a pre-existing conda env mounted from EOS (`ENVPATH`, passed in as
`condor_lxplus/submitter.py`'s `envpath`) — the two variants solve the same "how do I get a Python
env on the worker" problem very differently, and `condor/execute.sh`'s way is the expensive one.

**Expected benefit:** Large and directly billed as wall-clock per job. A `conda-forge` solve plus
a `coffea`-tree PyPI install is commonly 2-6 minutes even when every package is cached upstream,
and worse under network contention (many jobs hitting `conda-forge`/PyPI concurrently, which is
exactly what a several-hundred-job batch does). For a batch of hundreds of jobs this is easily
tens of hours of aggregate worker-node time spent re-doing identical setup, none of which
overlaps with actual event processing.

**Confidence:** High that the work is repeated identically per job with no dependency on job
input; medium on the exact minutes-per-job figure since it depends on `conda-forge`/PyPI caching
and worker-node network quality on the day.

**Implementation effort:** Medium-high for the full fix, low for a partial one:
- Cheap, low-risk partial fix: merge the two `micromamba install` calls (`:63`, `:65`) into one
  (`micromamba install python=3.10 setuptools=70.1.1 -c conda-forge xrootd --yes`), cutting one
  full dependency solve. Also merge `pip install -e .` and `pip install psutil` into one `pip
  install -e . psutil` call.
- Real fix: build once, use everywhere — e.g. a prebuilt conda-pack/tarball of the environment (or
  a CVMFS/EOS-mounted env, mirroring what `condor_lxplus/execute.sh` already does) staged as
  `transfer_input_files` or accessed read-only from the worker, so `execute.sh` only activates an
  existing env instead of building one. This is the same "pay once, not per job" pattern
  `docs/modernization_plan.md` item 5 recommends for the correction payload, applied to the
  environment instead.

**Risk:** Low for the merge-the-calls partial fix (same packages, same versions, just fewer solve
invocations). Medium for the prebuilt-env approach — must guarantee the shipped env is
bit-for-bit what `pip install -e .` would have produced (same `coffea`/`correctionlib` pins), or a
worker could silently run against a different dependency version than a job submitted before/after
an env refresh.

**Files:** `condor/execute.sh`.

---

### 2. `scripts/fetch.py` deletes its own site-map cache at the end of every run, so back-to-back invocations never benefit from it

**Location:** `fetch.py:2136-2140` (unconditional `os.remove(".sites_map.json")` at the end of
`main()`), versus the cache-validity check in `get_xrootd_sites_map()` (`fetch.py:196-206`, valid
for up to an hour) that this deletion defeats. Caller pattern: `scripts/suball.py:189,193,226`
invokes `python scripts/fetch.py ...` once per workflow in a loop, as a fresh subprocess each time.

**Problem:** `get_xrootd_sites_map()` is designed to amortize an expensive operation — walking
every `T*` directory under `/cvmfs/cms.cern.ch/SITECONF/` and JSON-parsing each site's
`storage.json` to build the XRootD prefix map — across multiple calls within an hour, via the
`.sites_map.json` file cache. But `main()` unconditionally deletes that file at the end of every
successful `fetch.py` invocation (not gated on error, not an opt-in flag). Since `suball.py` calls
`fetch.py` as a separate process once per workflow in a single driver run (3 call sites, each
inside a `for wf in scheme[...]` loop), every single one of those invocations re-does the full
SITECONF walk + per-site JSON parse from scratch — the cache never survives to be reused, even
though the invocations happen seconds to minutes apart, squarely inside the 60-minute validity
window it was built for.

**Expected benefit:** Moderate, scales with number of `T*` sites (commonly several dozen to ~100+
site directories on a current CVMFS SITECONF tree) and number of workflows processed per
`suball.py` run. Each avoided rebuild saves one full directory listing + N JSON file opens/parses;
across a multi-workflow commissioning run this is the difference between doing that walk once vs.
once per workflow.

**Confidence:** High — the deletion is unconditional and runs every time `main()` reaches the end,
and `suball.py`'s call pattern (fresh subprocess per workflow, no shared cache directory override)
is confirmed by reading both files.

**Implementation effort:** Low — delete the `os.remove(".sites_map.json")` block at
`fetch.py:2136-2140` entirely and let the existing mtime-based validity check in
`get_xrootd_sites_map()` govern staleness, or at minimum stop calling it from the normal `main()`
exit path.

**Risk:** Low. The existing 60-minute staleness check already handles the "site map went stale"
case; removing the self-deletion doesn't change correctness, only how often the expensive rebuild
happens. One thing to confirm: the cache is process-invocation-scoped by convention (lives in
whatever CWD `fetch.py` is run from) — if two unrelated `fetch.py` runs from different CWDs are
expected to never share state, this is a non-issue; if `suball.py` always runs from the same repo
root (it does, per the call sites above), the cache correctly stays local to that one driver run.

**Files:** `scripts/fetch.py`.

---

### 3. `check_site_responsiveness()` opens/stats every file in the dataset to confirm a site is reachable, even when the site succeeds

**Location:** `fetch.py:1011-1068` (the `dataset_files` branch of `check_site_responsiveness()`),
looping `for idx, test_file in enumerate(dataset_files)` and calling `access_xrootd_file()`
(`fetch.py:325-406`, each attempt with up to a 30s per-variant timeout) per file, called from
`getFilesFromDas()` at `fetch.py:1459-1461` with `dataset_files=flist` (the full file list).

**Problem:** This is related to `io_review.md` item #4 ("full per-file XRootD open+stat probe...
per site"), but the specific mechanism that review described — exhaustively checking every URL
variant even after a successful one is found — is no longer present in the current
`access_xrootd_file()` (it now returns on the first successful open/stat, `fetch.py:381,400`).
What remains, and is not fully captured by that earlier description, is the outer loop: to decide
"is this site usable at all," the function tests literally every file in the dataset one at a
time, sequentially, rather than a small representative sample. For a fully responsive site this
is still O(#files) sequential XRootD round trips before `fetch.py` accepts it and moves on — the
expensive case isn't just "site is down," it's "site is fine but the dataset has thousands of
files," which is the common case, not the exceptional one.

**Expected benefit:** Moderate-to-large for datasets with many files. Sampling e.g. the first
5-10 files (or a fixed cap) instead of the full list turns an O(#files) sequential network probe
into O(1) per site tried, which matters most exactly when things are working (large dataset,
responsive site) — the case that runs on every `fetch.py` invocation, not just the failure path.

**Confidence:** Medium-high that this is real overhead on every successful run; the magnitude
depends on per-file latency (which varies by site/network conditions) and dataset file counts,
neither profiled here.

**Implementation effort:** Low — cap the loop at `fetch.py:1030` to a fixed sample size (e.g.
`dataset_files[:10]`) instead of iterating the full list, keeping the existing fail-fast behavior
(first failure rejects the site) and the "first successful file's URL format is cached"
optimization already present (`fetch.py:1050-1056`).

**Risk:** Low-medium. A small sample could occasionally accept a site where a few files out of
thousands are actually missing/corrupt — but that's exactly what `--skipvalidation`'s downstream
`remove_bad_files()` pass and `--skipbadfiles`/`--retries` at run time already exist to catch
(per `user.md:210`'s documented mitigation), not a new failure mode this introduces.

**Files:** `scripts/fetch.py`.

---

### 4. `fetch.py`'s file validation opens every file twice: once via `gfal-ls`, once via `uproot.open`

**Location:** `fetch.py:1741-1756` (`validate()`), called via `p_tqdm.p_map` from
`remove_bad_files()` (`fetch.py:1759-1796`), which runs by default on every `fetch.py` invocation
unless `--skipvalidation` is passed.

**Problem:** For every file being validated, `validate()` first does `os.popen(f"gfal-ls
{file}").read()` purely to check non-emptiness (existence), then separately does `uproot.open(file)`
(with its own retry loop) to get the entry count. Both are network round trips to the same file;
`uproot.open` failing is already a valid "file not accessible" signal on its own (it's what the
retry loop below it already handles), so the `gfal-ls` pre-check doubles the per-file network cost
for every file in every sample, on every non-`--skipvalidation` run — this is the `fetch.py`-side
analog of `io_review.md` item #1 (double file-open at job start), but in the dataset-preparation
step rather than `runner.py`.

**Expected benefit:** Moderate, scales with total file count across all samples being fetched
(datasets are commonly hundreds to thousands of files). Dropping the `gfal-ls` pre-check and
relying on `uproot.open`'s existing failure/retry handling halves the network round trips in this
step.

**Confidence:** Medium — `gfal-ls` and `uproot.open` do overlap in what they detect (existence),
but `gfal-ls` might catch some non-ROOT-readable-but-technically-present edge case `uproot.open`
handles differently (e.g. a distinct error message in the output file); worth a quick check of
whether the `NotFound:` vs `FailedRetries:` distinction in the output file is actually consumed
downstream before dropping it.

**Implementation effort:** Low — remove the `gfal-ls` check and let the existing `uproot.open`
retry loop (`fetch.py:1746-1756`) be the sole liveness check, folding both outcomes into
`FailedRetries:`/success.

**Risk:** Low-medium — purely a dataset-preparation step (`--skipvalidation` already exists as an
escape hatch for anyone who wants to skip this entirely), zero physics risk, but changes the
wording/granularity of `_file_status.txt`'s failure categorization if anything downstream parses
`NotFound:` specifically.

**Files:** `scripts/fetch.py`.

---

### 5. `condor/execute.sh` spawns `jq` twice per argument key while parsing `arguments.json`

**Location:** `condor/execute.sh:27-30`:
```bash
for key in workflow output samplejson year campaign isSyst ttbar_reweights isArray noHist overwrite voms chunk skipbadfiles outputDir remoteRepo; do
    echo $(jq -r ".$key" $WORKDIR/arguments.json)
    ARGS[key]=$(jq -r ".$key" $WORKDIR/arguments.json)
done
```

**Problem:** For each of the 14 keys, `jq` is invoked once to print the value for the log and a
second time, identically, to populate `ARGS[$key]`. That's 28 `jq` subprocess spawns and file
re-reads per job to extract 14 values, when one read per key (or one `jq` call emitting all 14
fields at once) would do. `condor_lxplus/execute.sh:27-29` does the equivalent parsing with a
single `jq` call per key (no duplicate echo), so this is specific to `condor/execute.sh`.

**Expected benefit:** Small in absolute terms (each `jq` invocation is a few milliseconds), but
free to fix and multiplies by every job in a batch — hundreds of jobs each doing 14 redundant
subprocess spawns adds up to a measurable, entirely avoidable chunk of aggregate CPU/process
overhead across a large batch, for zero benefit (the echo is debug output already visible via
`-x` tracing).

**Confidence:** High — the duplication is directly visible in the script; no profiling needed.

**Implementation effort:** Low — capture once per key (`val=$(jq -r ".$key" ...); echo "$val";
ARGS[$key]="$val"`), or drop the `echo` line entirely since `bash -x` already traces the
assignment's command substitution to the log.

**Risk:** Very low. Setup-script only, no physics impact, output log content essentially
unchanged (the traced `+ ARGS[key]=...` line from `-x` already shows the value).

**Files:** `condor/execute.sh`.

---

## Not flagged / already tracked elsewhere

- **`fetch.py` re-querying DAS per candidate site** and the **originally-described exhaustive
  per-URL-variant probe** — `io_review.md` items #3 and #4, and listed in
  `modernization_plan.md`'s "below the line" section. Finding #3 above refines what's still live
  in that code path rather than duplicating the tracking.
- **`make_tarfile()` duplicated across `runner.py`, `condor/submitter.py`,
  `condor_lxplus/submitter.py`, `scripts/maketar.py` with the nest-guard only in one copy** —
  already tracked in `modernization_plan.md`'s "below the line" section; a one-time, per-submission
  cost (not per job), so it doesn't change the ranking here.
- **`condor_lxplus/submitter.py`'s `get_proxy_path()`** runs two separate `voms-proxy-info`
  subprocess calls — real but a one-time, per-submission cost, not per job; not worth a dedicated
  entry at condor-batch scale.
- **`condor_lxplus/execute.sh`** was otherwise clean for this pass: it reuses a pre-existing conda
  env (no bootstrap cost), and its `jq` parsing loop doesn't have the double-invocation bug in
  finding #5.
- **`remove_bad_files()`'s per-sample `p_map` pool** (`fetch.py:1759-1796`) creates a new
  multiprocessing pool per sample rather than one pool for the whole fetch; real but the pool
  startup cost is small relative to the per-file network validation work it wraps, and is
  dominated by finding #4 above once that's fixed.

---

## Summary table

| # | Location | Fires | Expected benefit | Confidence | Effort | Risk |
|---|----------|-------|-------------------|------------|--------|------|
| 1 | `condor/execute.sh` full env bootstrap per job | Every job | Large (tens of hours aggregate on a big batch) | High | Medium-high (full fix) / Low (merge-calls partial fix) | Low (partial) / Medium (full) |
| 2 | `fetch.py` self-deletes site-map cache | Every `fetch.py` invocation | Moderate | High | Low | Low |
| 3 | `check_site_responsiveness()` tests every file | Every site checked, every invocation | Moderate-large on big datasets | Medium-high | Low | Low-medium |
| 4 | `validate()` double per-file network check | Every file, unless `--skipvalidation` | Moderate | Medium | Low | Low-medium |
| 5 | Duplicate `jq` calls in `condor/execute.sh` | Every job × 14 keys | Small (adds up across a batch) | High | Low | Very low |

Recommend starting with #2 and #5 (trivial, zero-risk, purely wasted work), then #1's partial fix
(merge the micromamba/pip calls — low risk, still meaningful), then #3 and #4 (low effort, small
behavior-visibility tradeoffs worth a quick sanity check against `_file_status.txt` consumers
before merging). #1's full fix (prebuilt/shared environment) is the largest win on this page but is
its own project — sequence it whenever `condor/execute.sh` next needs touching anyway, not as a
quick patch.
