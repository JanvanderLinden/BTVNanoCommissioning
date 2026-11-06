# Condor Submission Infrastructure — Architecture Review

Based on a structural investigation of `condor/execute.sh`, `condor/submitter.py`,
`condor_lxplus/execute.sh`, `condor_lxplus/submitter.py`, and `scripts/fetch.py`. This is the
dedicated architecture-only pass over the condor submission stack flagged as a follow-up in
`docs/modernization_plan.md`'s "Post-plan additions" section, after the four earlier reviews
(`architecture_review.md`, `performance_review.md`, `io_review.md`, `simplify_review.md`) stopped
at `runner.py`/`correction.py`/workflow code and explicitly skipped this area.

Two live bugs surfaced by job triage on 2026-08-03 (`condor/execute.sh`'s unbounded micromamba
`wget`, and `fetch.py` not preferring a local EOS path) were already found and fixed
(`922004c`, `5650752`) — not re-litigated here.

A prior pass, `docs/condor_review.md`, already covers this same file set from a mixed
architecture/simplify/performance angle (its `S1`–`S4`, `P1`–`P3`, `A1`–`A3`). Rather than
repeat that document, this review: (a) carries forward the two items from it that are
architecture-flavored and still unaddressed (`A2`, `A3` below, renumbered), and (b) adds four new
findings that pass didn't cover, found by reading `scripts/fetch.py`'s module-scope code, its
executor dispatch in `main()`, and the relationship between `fetch.py` and
`src/BTVNanoCommissioning/utils/xrootdtools.py`.

`condor_lxplus/` is intentionally a lighter-weight, independently-runnable variant of `condor/`
per `CLAUDE.md` and `condor_lxplus/README.md` — nothing below recommends merging the two
directories.

---

### 1. `scripts/fetch.py` imports the canonical `get_xrootd_sites_map`, then immediately shadows it with its own copy

**Problem:** `scripts/fetch.py:12` does
`from BTVNanoCommissioning.utils.xrootdtools import get_xrootd_sites_map`, but `scripts/fetch.py:190`
then defines a module-level function with the exact same name. Python's module namespace means the
second definition wins: every call site in `fetch.py` (`fetch.py:1386`) invokes the *local* copy,
never the imported one — the import at line 12 is dead code that misleads a reader into thinking
`fetch.py` delegates to the shared module. `condor_lxplus/checkoutputs.py:7` imports and uses the
same function from the canonical `src/BTVNanoCommissioning/utils/xrootdtools.py:7`, so there are now
two independently-maintained copies of the SITECONF-parsing/`.sites_map.json`-caching logic, and
they have already started to drift: `fetch.py`'s copy gates its `"Loading SITECONF info"` print
behind `args.verbose` and uses `site["rse"] == None`; the canonical copy prints unconditionally and
uses `site.get("rse", None) is None`. Functionally equivalent today, but a real signal that the two
copies are no longer being edited together.

**Why it matters:** The CERN-local-EOS-preference logic just added (`5650752`) happens to live in
`getFilesFromDas()`'s call site rather than inside `get_xrootd_sites_map()` itself, so it wasn't
affected by this split — but the next fix to the site-map-building logic itself (a new SITECONF
protocol shape, a cache-TTL change, a new site-type filter) has no way to land in both copies short
of the author remembering to edit two files, and `condor_lxplus/checkoutputs.py` would silently keep
using the stale behavior.

**Proposed direction:** Delete the local redefinition in `fetch.py` and rely on the imported
`utils.xrootdtools.get_xrootd_sites_map`. If `fetch.py`'s `--verbose`-gated print is wanted, that's a
small, deliberate change to the shared function's signature (e.g. a `verbose=False` kwarg), not a
reason to keep a second copy.

**Risk:** Low — the two implementations are already functionally equivalent; swapping the call is a
mechanical deletion, easy to diff before/after.

**Priority:** High — this is exactly the kind of same-logic-in-two-places risk the framework's own
rules flag as a physics-adjacent concern for hot-path code, and here it's cheap to fix outright
rather than just document.

---

### 2. `--executor futures` and `--executor iterative` can assign different dataset names to the same input

**Problem:** `getFilesFromDas()` (`scripts/fetch.py:1231`) derives the sample-json key (`dsname`)
for a data dataset from a general regex (`data_run_pattern = re.search(r"[/_](Run20\d\d[A-Z])",
dataset)`, `fetch.py:1279`). `process_single_dataset()` (`fetch.py:1568`), which only the
`--executor futures` path in `main()` (`fetch.py:2032-2087`) calls, wraps that same call and then
applies an additional, `MuonEG`-specific correction (`fetch.py:1616-1633`): if the dsname it got back
still doesn't contain the expected run-period letter, or is missing a version suffix, it rewrites
`dsname` before handing it back. `--executor iterative` calls `getFilesFromDas()` directly
(`fetch.py:2030`) and never goes through this correction path.

**Why it matters:** For the exact dataset list that triggers this correction (evidently a case
`getFilesFromDas()`'s own dsname logic gets wrong for `MuonEG`), running the same `fetch.py`
invocation with `--executor iterative` vs. `--executor futures` produces a different key in the
output `metadata/<campaign>/*.json` — the artifact that `runner.py`, `condor/submitter.py`, and
`condor_lxplus/submitter.py` all key their per-sample bookkeeping and output filenames on. That's a
silent, executor-dependent change in dataset naming, the same failure shape (different behavior
depending on an operationally-invisible code path) as the two already-fixed live-triage bugs, just
one layer higher in the stack (metadata correctness instead of job liveness).

**Proposed direction:** Move the `MuonEG` dsname correction into `getFilesFromDas()` itself, right
next to the `data_run_pattern` logic it's patching, so both executors see it. If the correction is
instead a sign that the general regex needs fixing (rather than a special case), fix the regex
directly and drop the `process_single_dataset`-local patch.

**Risk:** Low to relocate (pure code motion); regenerate one already-affected `MuonEG` metadata JSON
under both executors and diff the keys to confirm they match before/after, per this repo's
preference for validating over reference output rather than just CI's `--limit 1`.

**Priority:** High — silently-different metadata keys from the same input, gated only by which
executor flag was passed, is a correctness footgun for anyone re-running `fetch.py` with a different
`--executor` than last time and expecting the same `metadata/*.json` output.

---

### 3. `scripts/fetch.py` runs `argparse.parse_args()` and campaign-resolution logic as import side effects

**Problem:** `scripts/fetch.py:151` (`args = parser.parse_args()`) and `scripts/fetch.py:245-264`
(the `if args.DAS_campaign == "auto": ... raise ValueError(...)` block) both sit at module scope,
not inside `if __name__ == "__main__":` (that guard only wraps the final `main(args)` call at
`fetch.py:2143-2145`). `--campaign` is `required=True` (`fetch.py:107-113`), so merely importing
`scripts/fetch.py` — as opposed to running it — parses `sys.argv` and can call `sys.exit()` or raise
`ValueError` before a single function in the file is reachable.

**Why it matters:** Every one of `fetch.py`'s ~20 functions (`getFilesFromDas`,
`check_site_responsiveness`, `get_dasgoclient_command`, etc.) reads like reusable dataset-discovery
utility code, but none of it can actually be imported and reused from another script or a test
without first satisfying `fetch.py`'s own CLI contract. This is plausibly *why* finding 1's
duplication exists in the first place: `condor_lxplus/checkoutputs.py` needed
`get_xrootd_sites_map()` and importing it from `fetch.py` directly wasn't a safe option, so a second,
independently-importable copy was created in `utils/xrootdtools.py` instead of factoring the
original out of `fetch.py`.

**Proposed direction:** Move `parser = argparse.ArgumentParser(...)`/`args = parser.parse_args()` and
the `DAS_campaign == "auto"` resolution block under `if __name__ == "__main__":`, passing `args`
into the functions that need it (most already take `args` as a parameter; the handful that read the
module-global `args` directly — e.g. `get_xrootd_sites_map`, `determine_execution_mode`,
`check_site_completion` — would need it threaded through explicitly). This is a larger, mechanical
refactor rather than a one-line fix, so it's reasonable to defer until someone actually needs to
import from this file, but it's worth flagging now since it's the root cause enabling finding 1.

**Risk:** Medium — touches every function that currently reads the module-global `args` instead of
a passed-in parameter; needs a careful pass to avoid missing one (which would surface immediately as
a `NameError` at call time, not silently).

**Priority:** Medium — no live bug today (nothing currently imports `fetch.py` as a module), but it's
the structural reason finding 1's duplication exists and will recur for the next thing someone wants
to reuse from this file.

---

### 4. File-validity is checked twice, by two independent, differently-implemented mechanisms

**Problem:** `check_site_responsiveness()` (`scripts/fetch.py:997`), when called with
`dataset_files=flist` (its only call site, `getFilesFromDas()` at `fetch.py:1459-1461`, always passes
the full file list), performs an `open`+`stat` probe over XRootD for every file in the dataset before
concluding a candidate site is usable. Separately and unconditionally afterward (unless
`--skipvalidation`), `main()` calls `remove_bad_files()` (`fetch.py:1759`), which re-checks every file
in the final, site-resolved list again via `validate()` (`fetch.py:1741`) — a completely different
mechanism (`gfal-ls` existence check, then up to 5 retries of `uproot.open(...)["Events"].num_entries`).
Both exist to answer the same question ("is this file actually readable?") for overlapping (often
identical) file lists, own no shared result, and have no reconciliation if they disagree — e.g. a file
that passes the XRootD open/stat probe but that `uproot.open` can't read (corrupt `Events` tree) is
only caught by the second pass, while a file that `gfal-ls`/`uproot` consider fine could still have
failed the first pass's XRootD probe, silently steering site selection away from a site that was
actually fine for that particular file.

**Why it matters:** Two independently-owned validation passes for the same underlying question is
the "duplicated responsibility" pattern this framework's own rules flag as a maintenance and
correctness risk elsewhere (see `architecture_review.md` item 1) — here it manifests as doubled
per-file network I/O (every file opened/stat'd via XRootD, then opened again via `uproot`) with no
guarantee the two checks' notions of "good file" ever converge, and no single place a future author
can look to understand "how does fetch.py decide a file is usable."

**Proposed direction:** Not a call to merge the mechanisms wholesale (XRootD open/stat is answering
"is this site responsive," `uproot.open` is answering "is this ROOT file's `Events` tree readable" —
legitimately different checks). But the two should at least be named/documented as intentionally
distinct, non-overlapping concerns (site liveness vs. file integrity), and `check_site_responsiveness`'s
per-file loop should stop being the accidental first line of file-integrity validation it currently
is by side effect (its actual job, per its name, is picking a site).

**Risk:** Low to document; medium if attempting to remove one of the two checks outright without
confirming the other alone catches everything the removed one did (both have caught different failure
classes on real jobs, per the comments describing the fail-fast/retry behavior in each).

**Priority:** Medium — no known live bug, but it's the same shape of silent-inconsistency risk as
finding 2, just at the file level instead of the dataset-key level.

---

### 5. (Carried forward, still open) `condor_lxplus/submitter.py` is missing the `--outputDir` locality check `condor/submitter.py` has

**Problem:** `condor/submitter.py:205-211` rejects a relative `--outputDir` at submit time, because
`condor/execute.sh` only transfers `.success` back through HTCondor — the real histograms/arrays are
`cp`/`xrdcp`'d directly from the worker node, so a relative path silently lands in the worker's
ephemeral scratch and is lost. `condor_lxplus/submitter.py` takes the same `--outputDir` argument and
`condor_lxplus/execute.sh` has the identical "only `.success` comes back, output leaves the worker
directly" shape, but has no equivalent check. Confirmed still absent as of this review (no
absolute-path/`root://` validation anywhere in `condor_lxplus/submitter.py`).

**Why it matters:** The exact silent-data-loss mode the check in `condor/submitter.py` exists to
prevent is still reachable through its sibling script.

**Proposed direction:** Port the same check into `condor_lxplus/submitter.py`, immediately after job-dir
creation.

**Risk:** Low — purely additive, rejects only submissions that would already lose their output.

**Priority:** Medium-high. (First identified in `docs/condor_review.md` as `A2`; repeated here because
it remains unaddressed and is squarely an architecture — not simplify/performance — finding.)

---

## Not flagged as new problems (already tracked, still accurate)

- **`condor_review.md`'s `A3`** (`check_site_responsiveness()` conflates redirector-shape parsing with
  the probe-and-decide loop, `fetch.py:1096-1140`) — re-checked, still accurate, still low priority/
  maintainability-only. Not repeated in full here.
- **`condor_review.md`'s `S1`–`S4`, `P1`–`P3`** — simplify/performance findings, out of scope for this
  architecture-only pass by design. `S1` (dead `make_tarfile()` in `condor_lxplus/submitter.py`) has
  since been fixed (`5eb4ace`); the rest are unchanged as of this review.
- **Three independently-maintained argparse definitions** (`runner.py`/`suball.py` vs.
  `condor_lxplus/submitter.py`) — already tracked as `architecture_review.md` item 6; `condor/submitter.py`
  is a fourth instance of the same pattern, not a new category of problem.
- **`condor_lxplus/` reading code/env/proxy live off an eos/afs mount instead of localizing everything
  to the worker** — already covered in `condor_review.md`'s "Data-locality note" as a deliberate,
  documented design tradeoff (`condor_lxplus/README.md`), not re-litigated here.
