# Condor Submission Infrastructure — I/O Review

Scope: `condor/execute.sh`, `condor/submitter.py`, `condor_lxplus/execute.sh`,
`condor_lxplus/submitter.py`, `scripts/fetch.py` — network calls, redirector choices, per-job
installs/opens, and file-transfer overhead in the condor submission pipeline. Same format as
`docs/io_review.md`, which stops at `runner.py`/`correction.py`/workflow code and explicitly
excludes this area; `docs/condor_review.md` already covers architecture/simplify/performance
findings for these same files (including two of the items below, `P1`/data-locality, in less
I/O-specific framing) — this document is the dedicated I/O-only pass `docs/modernization_plan.md`'s
"Post-plan additions" section asked for.

**Not re-reported as open:** the micromamba `wget` hang on unreachable nodes
(`condor/execute.sh:40`, fixed in commit `922004c` by adding `--timeout`/`--tries`) and `fetch.py`
routing every file through the global XRootD redirector instead of preferring CERN-local EOS
(fixed in commit `5650752`). Both were found via live job triage before this review and are
confirmed fixed in the current tree.

---

### 1. `pip install -e .` (plus a separate `pip install psutil`) re-downloads the entire heavy PyPI dependency stack from scratch on every job

**Location:** `condor/execute.sh:79` (`pip install -e .`) and `:82` (`pip install psutil`), run
inside the freshly-bootstrapped micromamba env (`:62-65`) after code extraction. `setup.cfg`'s
`install_requires` is `p_tqdm`, `vector`, `coffea==0.7.31`, `correctionlib==2.7.0`, `onnxruntime` —
the micromamba step just above only installs `python=3.10`, `xrootd`, and `setuptools` from
conda-forge (already flagged as a redundant per-job install in `docs/condor_review.md`'s P2), so
**none** of the actual analysis stack (`coffea` and its transitive deps — `awkward`, `numpy`,
`uproot`, `dask`, `numba` — plus `correctionlib`, `onnxruntime`, `vector`) is covered by that
proposal. `onnxruntime` wheels alone are on the order of 100+ MB.

**Expected impact:** This is a second, larger instance of the same "rebuild from network sources
every job" problem P2 already identified for the conda-forge layer — the PyPI layer here is
plausibly the heavier of the two given `onnxruntime`/`coffea`'s transitive footprint. For a
submission burst of hundreds of jobs, this is hundreds of concurrent PyPI resolves/downloads of an
identical, static dependency set. The separate `pip install psutil` call adds one more full
resolve+network round-trip for a single package that could ride along with the first call.

**Possible solution:** If `docs/condor_review.md`'s P2 (pack the `btv_coffea` env once via
`conda-pack`/`micromamba pack`, stage on EOS, fetch via `xrdcp`) is implemented, install
`BTVNanoCommissioning`'s own dependencies (and `psutil`) into that same environment *before*
packing it, so the packed artifact already satisfies `pip install -e .` and the per-job step
becomes a no-op/version-check rather than a full install. Note this explicitly: packing only the
conda-forge layer and leaving `pip install -e .` untouched would fix P2 without touching this
issue at all. Independent of that larger change, folding `psutil` into `install_requires` (or
`pip install -e . psutil` in one call) removes the second round-trip cheaply.

**Files:** `condor/execute.sh`, `setup.cfg`.

---

### 2. `fetch.py`'s site-responsiveness probe tests every file in the dataset, serially, before declaring a candidate site responsive

**Location:** `check_site_responsiveness()` (`fetch.py:997-1068`), called with `dataset_files=flist`
— the **full** file list — from the candidate-site loop at `fetch.py:1459-1461`. When
`dataset_files` is non-empty, the function loops over all of them (`fetch.py:1030-1063`), calling
`access_xrootd_file()` (open+stat, each with its own timeout) per file, and only stops early if a
file *fails* (fail-fast on inaccessibility) — a fully responsive site still pays the cost of every
single file before being accepted.

**Expected impact:** For datasets with hundreds-to-thousands of files, this is O(N) serial XRootD
round-trips just to confirm "is this site up," repeated for each candidate site tried in
completion order until one passes. This sits on the critical path of every `fetch.py` invocation
for a not-already-cached dataset and can turn site selection into a multi-minute step. Unaffected
by the two already-applied fixes: those removed a duplicate DAS query and stopped exhaustively
probing URL *variants* per file, but neither reduced the *number of files* tested.

**Possible solution:** Cap the sample passed to `check_site_responsiveness` (e.g. `flist[:5]`) at
the call site instead of the full `flist` — a fixed small sample is a statistically adequate
responsiveness probe, and `check_site_completion()` (the primary completeness gate, run first)
already answers "does this site actually have the files," so responsiveness only needs to answer
"is the redirector alive right now." (Same finding as `docs/condor_review.md`'s P1; repeated here
because it is squarely an I/O issue and this document's dedicated scope.)

**Files:** `scripts/fetch.py`.

---

### 3. `condor/execute.sh`'s `git clone` fallback pulls full repository history on every job, unlike the tarball path

**Location:** `condor/execute.sh:71-76`. When `--remoteRepo` is set, `condor/submitter.py` skips
building `BTVNanoCommissioning.tar.gz` entirely (`submitter.py:226-264`) and does not transfer it
(`submitter.py:371`), so on the worker `$WORKDIR/BTVNanoCommissioning.tar.gz` never exists and
`execute.sh` always takes the `git clone ${ARGS[remoteRepo]} .` branch (line 73) — there is no
`--depth`/`--single-branch` on this clone.

**Expected impact:** The tarball path deliberately strips `.git` (`submitter.py:253-263`'s
`exclude_dirs` comment records `.git` alone as ~666 MB, cutting the sandbox from 588 MB to
~38 MB) specifically to avoid shipping repo history to every job. The `git clone` fallback,
covering the same "get the code onto the worker" job, gets none of that treatment — every job run
with `--remoteRepo` clones the complete history over the network instead of the ~38 MB the tarball
path achieves. For a submission burst, this multiplies a large, avoidable transfer by the job
count.

**Possible solution:** Add `--depth 1` (and `--branch <ref>` if a specific ref is needed) to the
`git clone` call, matching the tarball path's intent of shipping only what's needed to run.

**Files:** `condor/execute.sh`.

---

### 4. `condor_lxplus/` runs the whole job — interpreter, code, correction-file reads — directly off an EOS/AFS network mount for the entire job duration

**Location:** `condor_lxplus/execute.sh:17` (`cd $COMMDIR`, putting the job's working directory
inside the shared mount), `:21` (`PATH="$ENVPATH:$PATH"`, with `condor_lxplus/submitter.py:187`
defaulting `ENVPATH` to `/eos/home-m/milee/miniforge3/envs/btv_coffea/bin`), `:18`
(`X509_USER_PROXY=$COMMDIR/proxy`, also read off the mount).

**Expected impact:** Every Python interpreter start and import for the whole job — plus every read
of `correction.py`'s SF payload files under `src/BTVNanoCommissioning/data/` — goes over the
network mount rather than a locally-staged copy, for as long as the job runs (not a one-time
startup cost like the other items here). `condor_lxplus/README.md` documents this as the
intentional tradeoff of the lighter-weight variant ("good for up to a few hundred files"), so it is
a deliberate design choice, not a bug — flagged here because it is the largest-scope instance of
"unbounded per-job network reads" in this codebase, and because `condor/`'s tarball+local-env
design already avoids it. (Same finding as `docs/condor_review.md`'s data-locality note; repeated
here because it is the single biggest item in this document's specific scope.)

**Possible solution:** Not a quick fix — bringing `condor_lxplus` in line with `condor/`'s
architecture (locally-staged code + env, condor-delegated proxy) would mostly collapse the two
variants into one design. Worth an explicit decision, not an opportunistic patch; see
`docs/condor_review.md`'s recommendation to treat this deliberately.

**Files:** `condor_lxplus/execute.sh`, `condor_lxplus/submitter.py`.

---

### 5. Both `execute.sh` variants re-open and re-parse the same small JSON file once per key instead of once

**Location:** `condor/execute.sh:27-30` and `condor_lxplus/execute.sh:27-29` — both loop over a
list of ~14 argument keys, spawning a separate `jq -r ".$key" $WORKDIR/arguments.json` subprocess
(and a fresh file open/parse) per key.

**Expected impact:** Low — `arguments.json` is a few KB, so each reopen is cheap and effectively
free after the first read hits the local page cache. Included only because it is a literal
instance of "repeated metadata reads without caching" against a bounded, well-defined file; it
does not scale with dataset size or job count in the way items 1-4 do.

**Possible solution:** One `jq` call producing a shell-sourceable `KEY=value` block (e.g.
`jq -r 'to_entries[] | "\(.key)=\(.value)"' arguments.json`), or read the JSON once in the
`python -c` invocation already used a few lines later (`condor/execute.sh:86`) and export what's
needed from there. Cosmetic relative to the rest of this document; not worth doing in isolation.

**Files:** `condor/execute.sh`, `condor_lxplus/execute.sh`.

---

## Confirmed sound (no fix needed)

- **`condor/submitter.py`'s tarball `exclude_dirs`** (`:253-263`) already strips `.git`,
  `metadata`, `SFbc-2D`, `docs`, `notebooks`, `testfile` — cutting the sandbox from 588 MB to
  ~38 MB per job. Good existing practice; item 3 above is exactly the gap where the `git clone`
  fallback doesn't get the same treatment.
- **`fetch.py`'s `das_queries`/`site_queries` fallback loops** (`fetch.py:1312-1362`) `break` as
  soon as one query succeeds — no redundant re-querying beyond what's needed to find a working DAS
  query form.
- **Stage-out in `condor/execute.sh`** (`:132-160`) is one `xrdcp`/`cp` call per output glob
  (`hists_*`, `arrays_*`), not per-file — reasonable batching, not a per-file transfer problem.
- **`condor/submitter.py`'s `use_x509userproxy = true`** (`:337`) delegates the proxy through
  HTCondor rather than reading it off a shared mount each job — contrast with item 4's
  mount-based proxy read in `condor_lxplus`.
- **`get_dasgoclient_command()`** (`fetch.py:441-469`) does a few local `os.path.exists` checks to
  locate the binary — no network calls, and it runs once per invocation rather than once per file
  or per site.

---

## Summary table

| # | Issue | Location | Frequency | Impact |
|---|-------|----------|-----------|--------|
| 1 | `pip install -e .` + `pip install psutil` re-download full PyPI stack | `condor/execute.sh:79,82` | Per job | High (network I/O at submission-burst scale) |
| 2 | Site-responsiveness probe tests every file in the dataset | `fetch.py:997-1068`, `:1459-1461` | #files per candidate site tried | Moderate-high (dataset-discovery time) |
| 3 | `git clone` fallback has no `--depth`, ships full history | `condor/execute.sh:71-76` | Per job (when `--remoteRepo` used) | Moderate (network transfer size) |
| 4 | Whole job reads code/env/corrections off EOS/AFS mount | `condor_lxplus/execute.sh:17,21,18` | For the job's full duration | Moderate-high, but a deliberate design tradeoff |
| 5 | `arguments.json` reopened once per key via separate `jq` calls | `condor/execute.sh:27-30`, `condor_lxplus/execute.sh:27-29` | ~14x per job | Low |

Recommend starting with #1 (compounds directly with `docs/condor_review.md`'s P2 — solve both
layers together when building the packed env) and #3 (one-line `--depth 1` fix), then #2 (one-line
sample cap, already scoped in `docs/condor_review.md` as P1). #4 is the highest-scope item but is
a deliberate `condor_lxplus` design tradeoff requiring an explicit decision, not a quick patch.
#5 is cosmetic and not worth doing in isolation.
