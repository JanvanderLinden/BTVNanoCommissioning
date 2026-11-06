# I/O Performance Review — BTVNanoCommissioning

Scope: file/branch access patterns, metadata reads, caching, and serialization for NanoAOD
columnar processing over XRootD/EOS at distributed scale (dask/condor/parsl). CPU/vectorization
concerns are out of scope (covered in `docs/performance_review.md`).

---

### 1. Every input file is opened twice at job start: once for validation, once for processing

**Location:** `runner.py:69` (`validate_dataset_structure()`, called unconditionally at `runner.py:546-548` unless `--skip-structure-validation`) opens every file with `uproot.open`, reads `.keys()` and `len(events)`; then `processor.run_uproot_job(...)` (`runner.py:655`, `1007`, `1122`, `1159`) has coffea's `NanoEventsFactory` reopen the same files for real chunking/processing.

**Expected impact:** Doubles XRootD/EOS TTree-header round-trips for every file in the fileset before any event processing starts. For a job with thousands of files over WAN/XRootD, this is a real, avoidable chunk of wall-clock startup latency, paid on every non-`--skip-structure-validation` run. (A separate `--validate` invocation adds a third full scan, but that path exits immediately so isn't stacked with the other two in one run.)

**Possible solution:** Either fold branch-existence/entry-count checks into the first coffea-managed open (coffea's own file-opening already reads this metadata), or make `--skip-structure-validation` cheap to default-enable for trusted/re-run datasets. Lowest-risk option: reuse the `uproot.open` handle/metadata from `validate_dataset_structure` instead of discarding it, if coffea's API allows passing pre-opened file objects/metadata into `run_uproot_job`.

---

### 2. `JPCalibHandler` reopens its ROOT calibration file every chunk, unlike every other correction object

**Location:** `correction.py:3920-3955` (`JPCalibHandler.__init__`, `uproot.open` at lines 3929/3947), constructed inside `process_shift()` at `BTA_producer.py:106` (called per shift from `process()` at line 102) rather than once in `__init__` like every other SF loader.

**Expected impact:** Reopens the same small ROOT template file, plus redoes `np.cumsum`/`np.sum` template preprocessing, on every chunk × every shift for BTA workflows — the only correction object in the codebase that isn't loaded once per processor instantiation. At scale (N chunks × M shifts per worker), this is N×M redundant opens of a file that's static per `(campaign, filename)`.

**Expected impact scale:** Moderate — small file, but non-trivial call count; inconsistent with the caching pattern used everywhere else in `correction.py`.

**Possible solution:** Move construction to `BTA_producer.py.__init__` (matching `load_SF()`'s pattern), or add a module-level cache dict keyed by `(campaign, filename)` mirroring `_TTBAR_REWEIGHT_CACHE` (`correction.py:38`) / `_jec_run_edge_cache` (`correction.py:529`).

---

### 3. `fetch.py` re-queries DAS for the same file list once per candidate site

**Location:** `fetch.py:1338-1349` fetches the full file list (`flist`) for a dataset once; then for every site in the candidate-site loop (`fetch.py:1421-1442`), `check_site_completion()` (`fetch.py:963-1018`) issues a **fresh** `dasgoclient -query="file dataset=... | grep file.name"` (lines 970-977) to recompute the same total-file count already held in `flist`.

**Expected impact:** N redundant DAS queries per dataset (N = number of candidate sites tried), each a full network round-trip to DAS. For datasets checked against many sites, this adds unnecessary dataset-discovery latency, though it's a one-time cost per `fetch.py` invocation, not per condor task.

**Possible solution:** Pass `len(flist)` (already computed) into `check_site_completion()` instead of re-deriving it via a fresh `dasgoclient` call.

---

### 4. Full per-file XRootD open+stat probe across all URL variants, per site, at dataset-discovery time

**Location:** `fetch.py:1054-1060` → `access_xrootd_file()` (`fetch.py:325-416`), called with `check_all_variants=True`. Per the function body (lines 366-414), when `check_all_variants=True` it does not early-return on the first successful URL variant — it tries `open` then `stat` for every remaining variant per file (each with a 30-40s timeout), even after finding a working one.

**Expected impact:** For a site that turns out unresponsive, this is a full (#files × #URL variants) open/stat sweep, each with a multi-second timeout, before `fetch.py` moves on to the next candidate site (`break` at line 1478 only fires after a site is confirmed responsive). This can dominate wall time of dataset preparation for datasets with several candidate sites, especially if higher-completion sites are checked first and turn out slow/unresponsive.

**Possible solution:** Early-return on first successful variant per file when the goal is just "is this site responsive," rather than exhaustively checking all variants; reserve `check_all_variants=True` for cases that actually need the full variant list (e.g. building the final usable URL).

---

### 5. Correction payload (correctionlib `CorrectionSet`s, JEC factory, lookup evaluators) is re-serialized to every chunk task, not loaded once per worker

**Location:** `runner.py:531-540` constructs `processor_instance` once (with `self.SF_map` fully populated by `load_SF()`, `correction.py:55-493` — includes multiple `correctionlib.CorrectionSet` objects, a `cloudpickle`-serialized JEC factory per `helpers/func.py:127-151`, and `coffea.lookup_tools` evaluators). Coffea's `run_uproot_job` with `dask_executor`/`parsl_executor` (`runner.py:1007-1036`, `1122-1172`) closes over this same `processor_instance` for every chunk it dispatches, so the whole correction payload is pickled and shipped to the scheduler/workers on every task submission.

**Expected impact:** For jobs with thousands of chunks, this repeatedly serializes/transfers the same static, potentially multi-MB correction payload (multiple `CorrectionSet`s, JEC factory) over the network per task rather than once per worker process. This is network I/O overhead scaling with chunk count rather than worker count — the dominant "unnecessary serialization" cost identified in this review.

**Possible solution:** Load-once-per-worker pattern (dask worker plugin, or a lazily-populated module-level singleton keyed by `(year, campaign)` that each worker initializes on first task and reuses thereafter), so only small per-chunk metadata is shipped per task instead of the full correction payload. Nontrivial change to the coffea 0.7.x processor-instance execution model used throughout `runner.py` — worth scoping as a larger effort rather than a quick fix, and validating no correction values silently change under lazy per-worker loading (priority #1: never change physics results silently).

---

### 6. `normalizeArrays.py` fully materializes post-hadd ROOT files in memory

**Location:** `condor_lxplus/normalizeArrays.py:49` (`f["TotalEventCount"]["TotalEventCount"].array(library="np")`) and `:58` (`f["Events"].arrays(library="ak")`) — reads the entire "Events" TTree of an already-hadd'd (merged across all chunks of a dataset) file into memory in one call, with branches rewritten unchanged afterward (`normalizeArrays.py:83`).

**Expected impact:** Post-hadd files are the largest single ROOT files in the whole pipeline (merged across all chunks of a dataset), so this is the one place reviewed where "materialize entire file into memory" happens on a file not bounded by `--chunk`. Impact scales with dataset size — could be a real memory/time cost for large merged datasets, though it's a one-off post-processing step, not per-chunk.

**Possible solution:** Switch to `uproot.iterate`/chunked read-modify-write if branch rewriting can be done incrementally, or confirm current EOS/condor node memory limits comfortably cover the largest expected hadd'd file size before treating this as urgent.

---

## Confirmed sound (no fix needed)

- **`load_SF()`** (`correction.py:55`) is called exactly once per workflow `__init__` (e.g. `example.py:59`), not per chunk — correct, since `processor_instance` is built once per job on the driver before executor dispatch.
- **ONNX/fragmentation-decay caches** (`_TTBAR_REWEIGHT_CACHE`, `correction.py:38`) and **JEC run-edge cache** (`_jec_run_edge_cache`, `correction.py:529`) correctly memoize expensive loads within a worker process across the multiple chunks that land on it — the only caveat is that in a distributed executor each worker process gets its own copy (cache hit rate is "per worker," not "per job"), which is expected/reasonable, not a bug.
- **`missing_branch()`/`array_writer.py`'s `uproot_writeable`**: branch selection is already column-selective (filters by `include`/`arraySchema` before triggering materialization via `ak.packed`/`ak.fill_none`) — no unnecessary branch materialization found here.
- **Per-chunk output write** (`array_writer.py:280-287`, single `fout["Events"] = events_out` write) is consistent with the bounded per-chunk processing model (`--chunk` default 200,000 events) — not an eager-loading concern at that scale.
- **condor_lxplus pipeline scripts' EOS globbing** (`checkoutputs.py`, `haddoutputs.py`/`checkHadd.py`) each independently glob the output tree, but these are separate hand-run scripts at different pipeline stages, not duplicated within a single run — expected workflow structure, not a bug.

---

## Summary table

| # | Issue | Location | Frequency | Impact |
|---|-------|----------|-----------|--------|
| 1 | Double file-open (validate + process) | `runner.py:69` + coffea job | 2× per file, per job run | Moderate-high (startup latency at scale) |
| 2 | `JPCalibHandler` reopens ROOT file, no cache | `correction.py:3920-3955`, `BTA_producer.py:106` | Per chunk × per shift | Moderate |
| 3 | Duplicate DAS file-list query per site | `fetch.py:977` vs `:1345` | N per dataset (N=#sites) | Low-moderate |
| 4 | Full XRootD probe, all variants, per site | `fetch.py:1054-1060`, `:366-414` | #files × #variants per site tried | Moderate (dataset-discovery time) |
| 5 | Correction payload re-serialized per chunk task | `runner.py:531-540` + dask/parsl executors | Per chunk task, not per worker | High (network I/O at scale) |
| 6 | Full in-memory read of post-hadd file | `normalizeArrays.py:49,58` | Once per hadd'd file | Scales with dataset size |

Recommend starting with #1 and #3 (localized, low-risk fixes to `runner.py`/`fetch.py`), then #2 (small, self-contained cache addition). #5 is the highest-impact item but requires restructuring the coffea executor/processor-instance model — scope as a dedicated effort with careful validation, not a quick patch.
