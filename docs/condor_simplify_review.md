# Condor Submission Infrastructure — Simplification Review

Scope: unnecessary complexity only — dead code, duplicate code, excessive wrappers, unnecessary
classes/config, complicated control flow — in `condor/execute.sh`, `condor/submitter.py`,
`condor_lxplus/execute.sh`, `condor_lxplus/submitter.py`, and `scripts/fetch.py`. Same format as
`docs/simplify_review.md`, which stops at `runner.py`/`correction.py`/workflow code and explicitly
excludes this area; this is the dedicated condor-submission pass `docs/modernization_plan.md`'s
"Post-plan additions" section asked for. Not architecture (`docs/condor_review.md`'s A1-A3) or
performance/I/O (`docs/condor_performance_review.md`, `docs/condor_io_review.md`) — those already
exist for this same file set. All findings below were confirmed by reading the full control flow
and grepping usage repo-wide before being reported as unused/unreachable/duplicated.

`condor_lxplus/` is intentionally a lighter-weight, independently-runnable variant of `condor/`
(`condor_lxplus/README.md`) — none of the findings below propose merging or consolidating the two.

**Already handled, not re-reported:** the 4 duplicated `make_tarfile()` copies (`runner.py:155`,
`condor/submitter.py:19`, `condor_lxplus/submitter.py`, `scripts/maketar.py`) — the
`condor_lxplus/submitter.py` copy was fully dead and deleted (commit `5eb4ace`); the other 3 got a
stale-tarball-nesting guard. Also not re-reported: `condor/execute.sh`'s micromamba `wget` timeout
(commit `922004c`) and `fetch.py`'s EOS-local site preference (commit `5650752`), both fixed via
live triage.

**Already tracked in `docs/condor_review.md`'s "Simplify findings" (S1-S4), not re-derived here:**
`condor_lxplus/submitter.py`'s vestigial `--remoteRepo` plumbing, its use of bare `input()` instead
of `_prompt_yn()`, the 3x copy-pasted global-redirector dict in `fetch.py`, and the duplicated
CI-vs-local temp-script boilerplate across `run_das_command`/`run_python_xrootd_ping`/
`run_xrdfs_command`. `docs/condor_io_review.md`'s item 1 already covers folding `psutil` into
`setup.cfg`'s `install_requires` instead of a separate `pip install psutil` in `condor/execute.sh` —
also not re-derived here.

---

### 1. `scripts/fetch.py`: an entire `elif` branch in `main()`'s dataset-resolution loop is unreachable

**Current complexity:** Inside the per-line loop of the `--from_dataset`/`--from_workflow` dataset
resolution block, `main()` has:
```python
if len(dataset) > 1:
    ...                                    # fetch.py:1939-1975
elif len(dataset) > 1:
    ...                                    # fetch.py:1977-1999 -- dead
else:
    ...
```
The `elif` at `fetch.py:1977` tests the exact same condition (`len(dataset) > 1`) the preceding `if`
at `fetch.py:1939` already tested and consumed. Since Python evaluates `if`/`elif` in order and the
first branch's guard is identical, the `elif` can never execute — it is permanently dead, along with
its own ~23 lines of nested logic (a second, differently-coded interactive campaign-picker with a
distinct prompt string, a distinct output-writing loop, and its own `args.verbose`-gated printing).

**Simplification:** Delete the dead `elif` block (`fetch.py:1977-1999`) entirely. If the two
interactive flows actually diverge in some way that matters (the `elif` block's second sub-branch,
`if args.from_workflow is None or dataset[0].endswith("SIM")`, suggests it may have been intended to
handle a case the `if` block's own logic doesn't), that divergence needs to be folded into the live
`if` branch first — but as written today, whichever behavior the `elif` implements has never once
run.

**Risk:** Very low to delete as dead code (confirmed unreachable by the literal duplicated
condition). The only care needed: before deleting, check whether the `elif`'s distinct behavior
(direct single-campaign `input()` prompt vs. the `if` branch's multi-select prompt) was the intended
UX for some case, in which case the fix is "make the `if` branch do that," not "delete it" — worth a
quick confirmation with whoever last touched this function rather than assuming.

**Files:** `scripts/fetch.py:1939-1999`.

---

### 2. `scripts/fetch.py`: `--from_dataset` without `--from_workflow` crashes on the first dataset line

**Current complexity:** In the same loop, `dataset` is only ever assigned inside a guard on
`args.from_workflow`:
```python
for l in lines:
    l = l.replace("\n", "")
    if args.from_workflow is not None:      # fetch.py:1928
        dataset = []
        for campaign in campaign_list:
            dataset.extend(direct_das_query(l, campaign))

    if not dataset:                          # fetch.py:1933 -- NameError if from_workflow is None
```
But the outer gate that reaches this loop is `if args.from_dataset or args.from_workflow is not
None:` (`fetch.py:1905`) — an "or", not an "and". So `--from_dataset` alone (no `--from_workflow`,
which the parser allows: they're independent flags with no `mutually_exclusive_group`) reaches
`fetch.py:1933`'s `if not dataset:` with `dataset` never having been assigned, raising
`NameError: name 'dataset' is not defined` on the very first non-comment line of the input file.
Confirmed this combination is unexercised: `docs/user.md:29` and `autobtv/.gitlab-ci.yml:221` both
always pass `--from_workflow`; no documented or CI invocation uses bare `--from_dataset`.

**Simplification:** Either wire up the DAS-query loop to also run for pure `--from_dataset` (drop
the `args.from_workflow is not None` guard around the `dataset = [...]` assignment, since
`direct_das_query`'s only per-mode difference is the `campaign_list`, which `--from_dataset` mode
would need to derive from `args.DAS_campaign` directly), or make the argparse help text honest that
`--from_dataset` currently requires `--from_workflow` to do anything. As written, this is a silently
broken, effectively dead code path masquerading as a supported CLI option.

**Risk:** Low to fix once the intended behavior is confirmed — this is a "what should happen"
question, not a mechanical deletion, so needs a decision from whoever relies on (or intended)
standalone `--from_dataset` before changing it.

**Files:** `scripts/fetch.py:1905-1933`.

---

### 3. `scripts/fetch.py`: `get_xrootd_sites_map()` is redefined locally, shadowing the imported shared version, and has already drifted from it

**Current complexity:** `fetch.py:12` imports `get_xrootd_sites_map` from
`src/BTVNanoCommissioning/utils/xrootdtools.py` — the same shared helper `condor_lxplus/checkoutputs.py`
actually uses. But `fetch.py:190-242` then defines its own **module-level function of the same
name**, which silently shadows the import; every call site in `fetch.py` (there is exactly one,
`fetch.py:1386`) resolves to the local copy, making the import at line 12 dead. The two copies are
already not identical: the local copy gates its "Loading SITECONF info" print behind
`args.verbose` (`fetch.py:207-208`) where the shared version always prints
(`xrootdtools.py:24`), and uses `site["rse"] == None` (`fetch.py:226`) where the shared version uses
`site.get("rse", None) is None` (`xrootdtools.py:42`) — cosmetic today, but proof the two copies are
already diverging independently rather than being edited together.

**Simplification:** Delete the local redefinition (`fetch.py:190-242`) and rely on the imported
`utils.xrootdtools.get_xrootd_sites_map`, after confirming the small behavioral deltas above
(verbose-gating, the `rse`-null check) don't matter — or backport them to the shared version first
if they do (e.g. the verbose-gating looks like a genuine improvement worth keeping).

**Risk:** Low — one call site to verify (`fetch.py:1386`), and the shared version is already
exercised by `condor_lxplus/checkoutputs.py`. Confirm the verbose-gating behavior isn't silently
lost for `fetch.py` callers who rely on the unconditional print.

**Files:** `scripts/fetch.py:12,190-242`, `src/BTVNanoCommissioning/utils/xrootdtools.py:7-58`.

---

### 4. `samplejson` is read from `arguments.json` into `ARGS[samplejson]` but never used, in both `execute.sh` variants

**Current complexity:** `condor/execute.sh:27` and `condor_lxplus/execute.sh:27` both loop over an
argument-key list that includes `samplejson`, populating `ARGS[samplejson]` from `arguments.json`
via `jq`. Neither script references `${ARGS[samplejson]}` anywhere afterward — both build the
actual per-job dataset file list from `split_samples.json`/`split_samples_resubmit.json` keyed by
`$JOBID` instead (`condor/execute.sh:86`, `condor_lxplus/execute.sh:24`), independently of the
original `--samples`/`--json` value the user passed to the submitter.

**Simplification:** Drop `samplejson` from the `for key in ...` list in both scripts. Purely removes
a dead `jq` read; no behavior depends on it.

**Risk:** Very low — mechanical, confirmed dead by grep in both files.

**Files:** `condor/execute.sh:27`, `condor_lxplus/execute.sh:27`.

---

### 5. `condor/execute.sh`: the micromamba-bootstrap gate is always true, regardless of `--remoteRepo`

**Current complexity:**
```bash
if [[ ${ARGS[remoteRepo]} != "" ]]; then       # execute.sh:37
    echo "remoteRepo is set to ${ARGS[remoteRepo]}"
    for i in {1..10}; do
        wget --timeout=20 --tries=1 -L micro.mamba.pm/install.sh
        ...
```
`ARGS[remoteRepo]` comes from `jq -r ".remoteRepo" arguments.json`. When `--remoteRepo` was **not**
passed to the submitter, `args.remoteRepo` is Python `None`, which `json.dump` serializes as JSON
`null` — and `jq -r` on a JSON `null` prints the literal 4-character string `null`, not an empty
string (confirmed: `echo '{"a": null}' | jq -r '.a'` → `null`). So `${ARGS[remoteRepo]}` is either
the string `"null"` (unset) or the actual URL (set) — **never** an empty string — meaning
`!= ""` is true unconditionally and this `if` never actually gates anything: the micromamba
installer download always runs, and the log always prints `"remoteRepo is set to null"` even on
ordinary tarball-mode jobs where the user never passed `--remoteRepo`. This is inconsistent with how
the same script correctly handles the identical situation two other times: `voms`
(`execute.sh:94`, `!= "null"`) and `isSyst` (`execute.sh:97`, `!= "false"`) both compare against the
actual jq-rendered sentinel, not `""`.

**Simplification:** Since the micromamba bootstrap is needed unconditionally in `condor/execute.sh`
regardless of `remoteRepo` (both the tarball and git-clone code-delivery paths still need a fresh
Python env), the cleanest fix is to delete the `if`/`fi` wrapper entirely and let the block run
unconditionally, which is what actually happens today anyway — this makes the code match its real
behavior instead of implying a condition that isn't real. (Alternatively, if some other conditional
skip was originally intended here, fix the comparison to `!= "null"` to match the rest of the file
— but that would change current behavior, since the block currently always runs, so confirm intent
first.)

**Risk:** Very low if removing the always-true wrapper (behavior-preserving, since the guarded code
already runs every time in practice) — but confirm no one is relying on the misleading "remoteRepo
is set to ..." log line as a signal. Slightly higher if instead "fixing" the comparison to
`!= "null"`, since that would change behavior (skip the bootstrap when remoteRepo is unset) and
needs sign-off, not silent adoption.

**Files:** `condor/execute.sh:37-52,94,97`.

---

### 6. `scripts/fetch.py`: the "new dsname" and "existing dsname" file-path branches duplicate the same ~15-line redirector-resolution block

**Current complexity:** `fetch.py:1520-1554`, the two branches of `if dsname not in fdict: ... else:
...` each independently repeat: check `site_url_formats` for a cached redirector, build `paths` via
either the cached redirector or `_get_pfn_for_site`, validate via `validate_xrootd_path`, and only
then differ in `fdict[dsname] = validated_paths` vs. `fdict[dsname].extend(validated_paths)`. The
`else` branch's guard is even subtly different (`best_site and best_site in site_url_formats` vs.
just `best_site in site_url_formats`) though functionally equivalent, since `None in site_url_formats`
already evaluates to `False` without the extra `and best_site` check.

**Simplification:** Compute `validated_paths` once, then do
`fdict.setdefault(dsname, []).extend(validated_paths)` — collapses both branches into one, and drops
the redundant `best_site and` guard along with it.

**Risk:** Very low — pure code motion, same computed value assigned either way;
`fdict.setdefault(...).extend(...)` is equivalent to the `not in fdict` / `else` split it replaces.

**Files:** `scripts/fetch.py:1516-1554`.

---

## Not flagged as findings (checked, confirmed clean or already tracked)

- `condor/submitter.py`'s `get_condor_submitter_parser` / `condor_lxplus/submitter.py`'s
  `get_condor_submitter_parser` are two separate, independent functions (same name, different
  modules, different flag sets — e.g. `condor/`'s has `--jobqueue`/`--accountingGroup`,
  `condor_lxplus/`'s has `--noSpool`) — this is the intentional lightweight-vs-full split
  `condor_lxplus/README.md` documents, not accidental duplication; already tracked as "no urgent
  action" in `architecture_review.md` item 6 and `docs/condor_review.md`'s "Not re-litigated"
  section.
- `_prompt_yn()` in `condor/submitter.py` is a genuine, load-bearing EOF-safety wrapper (not an
  unnecessary abstraction) — `condor_lxplus/submitter.py` lacking the same wrapper is already
  tracked as `docs/condor_review.md`'s S2, not re-derived here.
- `condor/execute.sh`'s `phase()` timing-marker helper is a small, single-purpose function used
  consistently at every phase boundary — not overbuilt.
- `condor_lxplus/submitter.py`'s `--remoteRepo` being a dead/no-op argument (accepted by the parser,
  read into `ARGS[remoteRepo]` in `condor_lxplus/execute.sh:27`, never branched on) is the same
  finding as `docs/condor_review.md`'s S1 — not re-derived here.
- `pip install psutil` as a separate step in `condor/execute.sh:82`, standing in for a missing
  `setup.cfg` `install_requires` entry, is already covered (with the same fix — add it to
  `install_requires`) by `docs/condor_io_review.md`'s item 1 — not re-derived here as a simplify
  finding since it was scoped there as an I/O (redundant network round-trip) issue.

---

## Summary table

| # | Issue | Location | Risk to fix |
|---|-------|----------|--------------|
| 1 | Unreachable `elif len(dataset) > 1` branch | `scripts/fetch.py:1977-1999` | Very low (confirm no intended-but-unwired UX first) |
| 2 | `--from_dataset` alone crashes (`NameError`) | `scripts/fetch.py:1905-1933` | Low (needs a design decision, not just deletion) |
| 3 | Local `get_xrootd_sites_map()` shadows shared import, already diverging | `scripts/fetch.py:12,190-242` | Low |
| 4 | Dead `samplejson` ARGS read | `condor/execute.sh:27`, `condor_lxplus/execute.sh:27` | Very low |
| 5 | Always-true `remoteRepo` bootstrap gate (jq `null` ≠ `""`) | `condor/execute.sh:37-52` | Very low (remove wrapper) / Medium (change semantics) |
| 6 | Duplicated dsname-branch redirector logic | `scripts/fetch.py:1516-1554` | Very low |

Recommend doing #4 and #6 first (mechanical, zero-risk deletions/collapses), then #3 and #5
(low-risk but worth a one-line sanity check each: verbose-gating behavior for #3, that no log
consumer depends on the misleading `remoteRepo` message for #5), and treating #1 and #2 as needing a
quick check-in with whoever last touched `fetch.py`'s dataset-resolution loop before changing
behavior — both sit in an interactive, human-in-the-loop path that's easy to misjudge from static
reading alone.
