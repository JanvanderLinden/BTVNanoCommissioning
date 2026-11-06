# Modernization Plan — BTVNanoCommissioning

Synthesis of `docs/architecture_review.md`, `docs/performance_review.md`, `docs/io_review.md`,
`docs/simplify_review.md`, plus verification of the load-bearing claims against the current tree.
Nothing here is implemented. This is a proposed sequence and a ranking.

Scoring is 1-5 in each column. **Runtime** and **Maint** are benefits (higher = better payoff).
**Diff** (difficulty) and **Risk** are costs (higher = worse). Risk means "risk of silently
changing physics output", which is priority #1 in `CLAUDE.md`.

---

## What the verification pass changed

Two corrections to the source reviews, both of which move items in the ranking:

**1. `btagSFs()` is much hotter than the performance review estimated — and the loop bound looks
wrong.** The review assumed 8-14 systematics. The actual `UParTAK4BC` systlist
(`correction.py:2142`) has **~100 entries**, and it is the only `btagSFs` call that is live
(`correction.py:4240`; the DeepJet/DeepCSV/PNet calls are commented out). With `--isSyst all` that
is ~100 systematics × max-jet-slots `correctionlib.evaluate()` calls, per shift, per chunk — and
the central SF is re-evaluated inside every systematic iteration. `syst=False` short-circuits after
one pass (`correction.py:2655`), so this only bites systematic runs, which is exactly where the
wall-clock pain is.

Separately, the jet-slot loop bound is `for nj in range(ak.num(alljet.pt)[0])`
(`correction.py:2464`) — the jet multiplicity of the **first event in the chunk**. `SelJet` is
jagged and unpadded (`_2D_Wc_valid_sf.py:470`, `sjets = event_jet[event_level]`), so if the first
surviving event has 2 jets, jets 3+ in every other event of that chunk never receive a shape SF.
The number of jets that get an SF then depends on chunk boundaries, which makes the result
`--chunk`-dependent. **This needs to be confirmed or refuted before anything else on this list**,
because if it is real it is a live physics bug, and if it is intentional it needs a comment saying
so. It sits in the same function the vectorization work would rewrite, so the two go together.

**2. The startup double-open is already half-mitigated.** `io_review.md:9` proposes making
validation cheaper, but `--validate-files-per-sample` already exists (`runner.py:380`) and does
exactly that. The remaining work is choosing a sane default, not building new machinery. That
drops the item's difficulty from "medium" to "one line plus a discussion".

One more correction, minor: `axes/qgtag.py` is **not** a byte-identical duplicate of `axes/common.py`
(`architecture_review.md:49`). It genuinely differs in `pt` (variable geomspace-style binning vs.
`Regular(60,0,300)`), `eta` (`±5.0` vs `±2.5`), adds `pflav`, and drops `ljpt`. Merging via
`common.axes | {...}` is still right, but it is a real (if harmless) change to the axes dict, not
a no-op.

---

## Phase 0 — the prerequisite (not one of the ten)

**Build a reference-output regression check before touching the hot path.**

There is no unit test suite; CI runs `runner.py --limit 1 --executor iterative`, which proves the
code doesn't crash and nothing else. Items 1, 4, 5, 6 and 10 below all modify code that produces
physics numbers, and "avoid large rewrites / never change physics silently" cannot be enforced by
inspection alone at this scale.

Minimum viable version: a script that runs 2-3 representative workflows (one 2D, one ttbar, one
BTA) over a fixed small fileset with `--isSyst all`, and diffs the resulting `.coffea` histograms
and ROOT ntuple branches bin-by-bin against a stored baseline. A few hundred lines, run manually
before each of the risky merges. Without it, every item scored risk ≥3 below is effectively
unmergeable with confidence.

Cost: ~1-2 days. This is the single highest-leverage thing on the page, because it converts four
"too risky to attempt" items into "verifiable".

**Status: implemented.** `scripts/compare_coffea.py` diffs two `runner.py` output trees --
`.coffea` histograms bin-by-bin (matching categorical axes like `syst`/`osss` by label, not
position) and `--isArray` ROOT ntuple branches (jagged-count-aware). `scripts/phase0_regression.sh`
drives it: runs `2D_mu_Wc_sf` and `2D_emu_ttdilep_sf` locally over the small test filesets
`SFbc-2D/01_test_workflows.sh` already uses, for both MC and data, always with `--isArray`, MC with
`--isSyst all` by default (data doesn't carry MC weight systematics, so it's not useful there).
`--chunk`/`--max` bound each run to a small fixed slice so it stays runnable in minutes rather than
processing the full (much larger than expected) test files.

Usage:
```bash
scripts/phase0_regression.sh baseline   # snapshot the reference, e.g. on master before item 1
# ... make the risky change ...
scripts/phase0_regression.sh check      # rerun and diff against the snapshot
```

`phase0_baseline/`/`phase0_check/` land at the repo root and are gitignored (they match the
existing `hists_*`/`arrays_*`/`*.coffea` patterns), same as `01_test_workflows.sh`'s own output --
**the baseline is local and uncommitted, not a shared fixture.** Each `baseline` run stamps
`phase0_baseline/PHASE0_BASELINE_COMMIT` with the commit hash, branch, UTC timestamp, and whether
the tree was dirty at snapshot time, since nothing else records what code state it reflects;
`check` prints that stamp before diffing. A teammate who needs to validate their own change reruns
`baseline` themselves from their own known-good state -- there is no cross-machine baseline to
share.

Any reported diff must be reviewed and signed off by the workflow owner before merging, not
silently absorbed (this applies with extra force to item 1, where a legitimate loop-bound fix is
*expected* to produce a real, intentional difference from the stored baseline).

---

## The ranking

| # | Change | Runtime | Maint | Diff | Risk |
|---|--------|:-------:|:-----:|:----:|:----:|
| 1 | `btagSFs()`: fix jet-slot bound, flatten-and-vectorize the syst×jet loop | **5** | 3 | 3 | 3 |
| 2 | Dead code + dead parameter sweep | 1 | **4** | **1** | **1** |
| 3 | Hot-path micro-vectorizations (`fixPhiRange`, 2D WP map, `partial_weight`) | 3 | 2 | **1** | **1** |
| 4 | `weight_manager()`: skip shift-invariant weight recomputation | **5** | 2 | 5 | 4 |
| 5 | Load correction payload once per worker, not once per chunk task | 4 | 3 | 4 | 3 |
| 6 | Unify the two histogram-filling dispatch implementations | 2 | **5** | 4 | 4 |
| 7 | `ttbar_reweights`: collapse the two config channels into one | 1 | 3 | 2 | 3 |
| 8 | Stop opening every input file twice at job start | 3 | 2 | **1** | 2 |
| 9 | `JPCalibHandler`: stop reopening the calib file every shift | 2 | 2 | **1** | **1** |
| 10 | `add_fragmentation_decay_weights()`: numba the gen-particle loop | 4* | 1 | 4 | 3 |

\* Item 10's runtime score applies only when `--ttbar-reweights full` is active; it is 0 otherwise.

**Recommended execution order: 2, 3, 9 → Phase 0 → 1, 8 → 5, 4 → 6, 7, 10.**
Items 2, 3 and 9 are safe enough to do without the regression harness and buy goodwill/wall-clock
immediately. Everything from item 1 onward should wait for Phase 0.

---

## The ten, in detail

### 1. `btagSFs()` — fix the jet-slot bound, then vectorize

**What:** First, determine whether `range(ak.num(alljet.pt)[0])` (`correction.py:2464`) is a bug or
an undocumented "leading jets only" intent. Then restructure: flatten `SelJet` once with
`ak.flatten`, evaluate each systematic on the full flat jet array in a single `correctionlib`
call, `ak.unflatten` and reduce per event. Hoist the `"central"` evaluation out of the systematic
loop — it is currently recomputed ~100 times per shift with identical inputs.

**Expected runtime:** Largest single win available on systematic runs. Call count drops from
`n_syst × max_jets` to `n_syst`, i.e. roughly the average jet multiplicity (3-6x fewer calls), plus
another ~1/100th of the work saved by not re-evaluating central per systematic. On `--isSyst all`
jobs where `btagSFs` is a meaningful fraction of `weight_manager()` time, expect a clearly
measurable end-to-end improvement, not a micro-optimization.

**Maintenance:** Moderate improvement — the nested loop is ~200 lines of near-duplicated
`SFtype` branches; flattening collapses much of it.

**Difficulty:** Medium. Confirm `correctionlib==2.7.0`'s `evaluate()` accepts flat numpy arrays for
the shape SF (it should; verify rather than assume). The `ak.fill_none`/`masknone` handling has to
survive flattening — that is where the care goes.

**Risk:** Medium-high in principle, low in practice *if* Phase 0 exists — a bin-identical histogram
diff on a systematic run either passes or it doesn't. Note that if the loop-bound issue is real,
the "correct" version will legitimately differ from the baseline, and that difference has to be
reviewed and signed off by the workflow owner, not silently absorbed.

**Files:** `utils/correction.py` (`btagSFs`, ~2119-2670).

---

### 2. Dead code and dead parameter sweep

**What:** One mechanical PR bundling the confirmed-dead items from the simplification review:
- unused `campaign` parameter in `softmu_mask`/`mu_idiso`/`mu_promptmvaid`/`btag_mu_idiso`
  (`selection.py:227,239,248,261`) and its ~15 call sites
- unreachable ternary in `ele_promptmvaid` (`selection.py:220-222`) — it returns a bare `0.3`
  scalar and would raise if it ever fired
- shared `etaSC` block duplicated across three electron-ID functions (`selection.py:186-224`)
- dead `get_condor_submitter_parser` copy in `runner.py:168`
- dead `memory_usage_psutil()` / `num()` (`helpers/func.py:49,91`) — verified zero references
- `axes/qgtag.py` → `common.axes | {overrides}` (see correction above; not a pure no-op)

**Leave out** of this PR: `jmar_sf()` and the `"JMAR"` `load_SF()` branch. It is genuinely unused,
but JMAR SF support may be intentionally staged rather than abandoned — ask before deleting.

**Expected runtime:** None. Included because it is nearly free and shrinks the surface everything
else has to be reviewed against.

**Maintenance:** High for the effort. Removes the "is this parameter load-bearing?" question from
the selection layer, which is the file new workflow authors copy from most.

**Difficulty:** Trivial, entirely grep-verifiable.

**Risk:** Very low. The one thing to watch is the `qgtag` axes merge re-introducing `ljpt`; confirm
nothing keys off the axes dict's exact contents.

**Files:** `utils/selection.py`, `helpers/func.py`, `runner.py`,
`utils/histogramming/axes/qgtag.py`.

---

### 3. Hot-path micro-vectorizations

**What:** Three independent, mechanical replacements:
- `fixPhiRange` list comprehensions → `np.mod(phi + np.pi, 2*np.pi) - np.pi`. Note this is
  **7 call sites**, not 1 as the performance review states: `correction.py:912, 942, 945, 956, 959,
  969, 972`. Six of them are in the unclustered-MET variation block, which runs per shift.
- 2D WP bin lookup (`helpers/update_branch.py:416-428`) → convert `["2D"]["mapping"]` to a numpy 2D
  array once at load, replace the per-jet comprehension with `mapping[ihfvlf, ibvc]` plus a
  `np.where` for the `-1` sentinel.
- Hoist `weights.partial_weight(exclude=exclude_btv)` to a local in `ctag_Wctt_valid_sf.py` — same
  exclusion set recomputed at lines 498, 514, 574, 598.

**Expected runtime:** Moderate and cheap. The `fixPhiRange` sites are the best of the three
(O(n_events) Python × ~15 shifts × 7 sites). The others are per-chunk or small.

**Maintenance:** Slight improvement.

**Difficulty:** Low. Each is a few lines and independently checkable — the `fixPhiRange` rewrite is
exactly equivalent for all finite inputs and can be checked against the scalar version on boundary
values near ±π.

**Risk:** Low. These are the three items in the whole plan whose output can be proven identical
without a full reference run.

**Files:** `utils/correction.py`, `helpers/update_branch.py`,
`workflows/ctag_Wctt_valid_sf.py`.

---

### 4. `weight_manager()` — don't recompute shift-invariant weights on every shift

**What:** `weight_manager()` runs once per entry in the `shifts` list (~30-40 with `--isSyst all`),
recomputing pileup, PDF/scale/PS, muon ID and electron ID SFs each time. For pure jet-energy
shifts (JES sources, JER) none of those inputs change. Cache the shift-invariant components on
first evaluation and reuse; keep recomputing anything that reads shifted jet or lepton kinematics
(b-tag shape SF must stay per-shift).

**Expected runtime:** Potentially the largest structural win on the list, plausibly comparable to
item 1. But the split between shift-invariant and shift-dependent cost has never been profiled —
**profile before building this.** A `--isSyst all` run under `cProfile` or the existing phase
timers (added in commit `350f601`) will say in an hour whether this is worth two weeks.

**Maintenance:** Slightly negative, honestly — it adds a caching layer and a correctness invariant
("this weight does not depend on this shift") that is not currently expressible in the code and
will need a comment block explaining itself.

**Difficulty:** High. Requires classifying every weight component by which shift types it depends
on, and threading that through `common_shifts`/`process_shift`.

**Risk:** High. A misclassified component silently freezes a weight that should have varied — which
does not crash, does not show up in `--limit 1` CI, and produces a subtly wrong systematic band.
This is the item that most needs Phase 0, and it should additionally be validated by checking that
each systematic band's *width* is unchanged, not just the nominal.

**Files:** `utils/correction.py` (`weight_manager`, `common_shifts`), every workflow's
`process_shift`.

---

### 5. Load the correction payload once per worker

**What:** `runner.py:531-540` builds one `processor_instance` with `SF_map` fully populated
(multiple `correctionlib.CorrectionSet`s, a cloudpickled JEC factory, `lookup_tools` evaluators).
Coffea's dask/parsl executors close over it per chunk task, so a multi-MB static payload is
pickled and shipped per task rather than per worker. Replace with a lazily-initialized
module-level singleton keyed by `(year, campaign)`, or a dask worker plugin.

**Expected runtime:** No CPU change, but real wall-clock and scheduler-pressure improvement on
jobs with thousands of chunks — serialization cost stops scaling with chunk count. This matters
most for exactly the large systematic productions items 1 and 4 target.

**Maintenance:** Moderate improvement; makes the "corrections are static per campaign" assumption
explicit instead of accidental.

**Difficulty:** High-ish, because it fights coffea 0.7's processor-instance model. See the
strategic note below — coffea's own execution model is going to change under this code eventually,
so keep the implementation small and don't build an elaborate abstraction here.

**Risk:** Medium. Lazy per-worker loading must produce identical correction values; the failure
mode (worker loads a different campaign's corrections due to a key collision) is silent and severe.
Worth an explicit assertion that the cached key matches the requested one.

**Files:** `runner.py`, `utils/correction.py` (`load_SF`).

---

### 6. Unify the two histogram-filling dispatch implementations

**What:** `histogrammer.py:58-641` (`histo_writter`) and `ctag_Wctt_valid_sf.py:475-657` are two
independently-maintained ~580-line substring-dispatch chains filling the same physics content,
including two copies of the genflavor/smflav computation. The second backs 7 of 31 registered
workflows. Diff them histogram-by-histogram, then either extend `histo_writter` to cover the Wc/ttdilep
cases or confirm it already does and delete the copy.

**Expected runtime:** Low directly; unlocks performance review #7 (redundant `ak.flatten`/`ak.num`
across dispatch branches), which is worth having but is not why you do this.

**Maintenance:** The highest maintenance payoff on the list, by a distance. Two copies of the most
safety-critical logic in the framework means every histogram fix has to be applied twice, and
nothing catches it when it isn't.

**Difficulty:** High — 580 lines × 2, and the diff is the whole job.

**Risk:** High. Any behavioural difference between the paths becomes a histogram-content change
across 7 production workflows the moment they're unified. Non-negotiably requires Phase 0 plus
workflow-owner sign-off on every intentional difference found.

**Files:** `utils/histogramming/histogrammer.py`, `workflows/ctag_Wctt_valid_sf.py`.

---

### 7. `ttbar_reweights` — one config channel, not two

**What:** The value travels both as `os.environ["BTV_TTBAR_REWEIGHTS"]` (`runner.py:414`) and as a
post-construction `setattr` (`runner.py:540`), with `weight_manager` falling back from one to the
other (`correction.py:4216-4217`) and workflows reading it defensively via `getattr`. Determine
whether the env var exists because of a dask/condor serialization constraint. If not, make it a
normal `__init__` argument like `isSyst`/`isArray` and delete the env var. If it is, document the
constraint at the env-var line — right now nothing says why.

**Status: partially fixed.** The investigation found the two channels *did* disagree in practice.
14 of 18 workflows pass `ttbar_reweights` explicitly to `weight_manager()` via
`getattr(self, "ttbar_reweights", "none")`, reading the `setattr`-based channel — safe under any
executor since the value is baked into `processor_instance` before it's ever pickled. But
`dijet.py`, `photonjet.py`, `_2D_Wc_valid_sf.py` and `_2D_ttsemilep_valid_sf.py` called
`weight_manager()` without passing it at all, silently falling through to the
`os.environ["BTV_TTBAR_REWEIGHTS"]` fallback — a lookup performed on whatever process actually runs
`process_shift`. That's fine under `iterative`/`futures` (same process or a fork, which inherits
`os.environ`), but not under `dask/*` or `--executor condor_standalone`, whose workers don't inherit
a driver-side `os.environ` mutation made after the interpreter started. The user confirmed
`2D_mu_Wc_sf`/`2D_e_Wc_sf`/`2D_mu_ttsemilep_sf`/`2D_e_ttsemilep_sf` are run via dask/condor_standalone
in production, i.e. this was live, not latent: `--ttbar-reweights full` was silently becoming
`"none"` for those workflows under those executors.

Fixed (commit `c3a7cfd`) by adding the same explicit `ttbar_reweights=getattr(self,
"ttbar_reweights", "none")` call to the 4 affected workflows, matching the other 14. This closes the
divergence but is narrower than the item as originally scoped: the env var and the `setattr` channel
both still exist, and `weight_manager()`'s fallback is untouched. Deleting the env var entirely and
threading `ttbar_reweights` through every workflow's `__init__` (the original full scope) remains
undone — a lower-urgency follow-up now that the actual divergence is closed.

**Expected runtime:** None.

**Maintenance:** Good. Removes a config path that a new workflow author has no way to discover.

**Difficulty:** Low-medium; the investigation is most of the work.

**Risk:** Medium and worth taking seriously despite the small diff. If the two channels can ever
disagree across an executor boundary, the failure mode is applying the wrong ttbar reweighting
silently. Must be verified on all of iterative, dask and condor before the fallback is removed —
which is also the test that tells you whether the env var was load-bearing.

**Files:** `runner.py`, `utils/correction.py`, workflow `__init__`s.

---

### 8. Stop opening every input file twice at job start

**What:** `validate_dataset_structure()` (`runner.py:25`) opens every file with `uproot.open` and
reads keys and entry count; coffea then reopens all of them for real. As noted above,
`--validate-files-per-sample` (`runner.py:380`) already provides the escape hatch — it just
defaults to `None`, meaning "check everything, every run".

**Expected runtime:** No CPU change, but removes a full serial XRootD header round-trip sweep from
job startup. On thousand-file filesets over WAN this is minutes of wall clock per submission, paid
every run.

**Maintenance:** Slight.

**Difficulty:** Low. The cheap version is changing the default to a small number (e.g. 1-2 files
per sample) so re-runs of a known-good fileset don't re-scan everything. The thorough version —
folding the branch check into coffea's own first open — is a bigger change with a worse
effort/benefit ratio; not recommended now.

**Risk:** Low-moderate, and it is a *usability* risk rather than a physics one: a weaker default
means a corrupt file is caught later (mid-job) instead of at startup. That is a real tradeoff and
should be the user's call, not silently changed. Recommend proposing the default change explicitly
rather than deciding it in a refactor PR.

**Files:** `runner.py`.

---

### 9. `JPCalibHandler` — stop reopening the calibration file every shift

**What:** Constructed at `BTA_producer.py:766`, which is inside `process_shift` (starts at line
106), so it re-runs `uproot.open` plus the `np.cumsum`/`np.sum` template preprocessing on every
chunk × every shift. It is the only correction object in the codebase not loaded once per processor
instance. Move it to `BTA_producer.__init__`, or add a module-level cache keyed by
`(campaign, filename)` mirroring the existing `_TTBAR_REWEIGHT_CACHE` (`correction.py:38`).

**Expected runtime:** Small file, but N chunks × M shifts redundant opens for BTA workflows. Modest
absolute win, confined to one workflow family.

**Maintenance:** Small improvement — makes the loading pattern uniform across all correction
objects, which is worth something on its own.

**Difficulty:** Low. The cache-dict version is the safer of the two since it doesn't change
construction order.

**Risk:** Low. Only caveat: confirm the handler holds no per-chunk or per-shift mutable state
before caching it. `simplify_review.md:83` says its state is loaded templates and precomputed
cumulative arrays, which is cacheable — verify rather than trust.

**Files:** `utils/correction.py`, `workflows/BTA_producer.py`.

---

### 10. numba the fragmentation/decay gen-particle loop

**What:** `add_fragmentation_decay_weights()` (`correction.py:3760-3809`) is a per-event,
per-genparticle pure-Python double loop building a children map, plus a genealogy search. At
~50k events × ~200 GenPart entries this is ~10M Python-level operations per chunk. Wrap the
per-event body in `@numba.njit` rather than attempting an awkward-native rewrite — the logic is
already flat per-event lists, which numba handles well, and events are independent.

**Expected runtime:** Very large *for this function* (10-50x plausible), but it only fires under
`--ttbar-reweights full`, and it runs once per chunk rather than per shift. Ranked tenth because of
that gating, not because the win is small — if `full` is a routine production setting rather than
an occasional study, promote this to the top three.

**Maintenance:** Slightly negative — adds numba to the hot path and a compilation-time cost on first
call.

**Difficulty:** Medium-high. numba's typing of the existing list-of-lists structures is where the
time goes.

**Risk:** Medium. Genealogy logic is fiddly and a subtly wrong children map produces plausible-looking
wrong weights. Mitigated by the fact that this one is easy to test in isolation: run both
implementations on the same chunk and assert bit-identical weight arrays.

**Files:** `utils/correction.py`.

---

## Below the line (worth doing, didn't make the ten)

- **`make_tarfile()` duplicated 4x, safety guard in only one copy** (`runner.py:155`,
  `condor/submitter.py:19`, `condor_lxplus/submitter.py:9`, `scripts/maketar.py:5`). The 3 unguarded
  copies can nest a stale tarball inside the new one. Recommend applying the guard to all four
  **without** consolidating into a shared module — consolidation fights the "independently
  runnable scripts" design that `condor_lxplus/README.md` documents.
- **`fetch.py` redundant DAS queries and exhaustive XRootD variant probes** (io #3, #4). Real
  dataset-prep latency, zero physics risk, but it is a one-off cost per `fetch.py` invocation
  rather than per job.
- **`normalizeArrays.py` fully materializes post-hadd files** (io #6). Check the largest expected
  merged file against condor node memory limits first; if it fits comfortably, this is a
  non-problem.
- **`array_writer()`'s three calling conventions** (arch #3). Documentation-only fix, and the file
  has uncommitted changes in the working tree — wait until those settle.
- **`common_shifts(self, ...)` / `array_writer(processor_class, ...)` taking `self`** (arch #2) and
  **three parallel argparse definitions** (arch #6). Both are correct observations and both are
  wide, low-value diffs. Track; fix opportunistically when touching those files anyway.

---

## Strategic note: the coffea pin

Not a ranked item because it contradicts "avoid large rewrites", but it shapes several of the
decisions above.

The repo pins `coffea==0.7.31` and Python `<3.11`. That branch is end-of-life; `processor.run_uproot_job`
and the whole processor-instance execution model this codebase is built on were replaced by the
dask-awkward-based API in coffea 2023.x+ and have since been removed. Python 3.10 goes EOL in
October 2026. Nothing breaks tomorrow, but the environment is frozen and will get harder to
reproduce, not easier.

The practical implication for this plan: **item 5 (per-worker correction loading) is solving a
problem that the coffea execution model change would also solve, differently.** Keep that
implementation minimal and disposable rather than building a durable abstraction around coffea 0.7's
task model. Same caution, more weakly, applies to item 4.

Recommendation: don't start a migration now, but timebox a spike — take one simple workflow
(`example.py`), port it to the current coffea API in a branch, and find out whether this is a
two-week or a six-month job. That number is worth knowing before committing to the more expensive
items on this list. If it turns out cheap, the ordering above changes substantially.

---

## Post-plan additions — condor submission infrastructure (sequence after everything above)

`condor/`, `condor_lxplus/` and `scripts/fetch.py`'s redirector logic were out of scope for the
four source reviews this plan synthesizes (`architecture_review.md`, `performance_review.md`,
`io_review.md`, `simplify_review.md` all stop at `runner.py`/`correction.py`/workflow code). The
two items below came out of live job triage on the `2D_*_Wc_sf` condor batches
(2026-08-03), not from a code review pass, and are operational/infrastructure rather than
hot-path physics code — hence sequenced after the ranked ten and the below-the-line items, not
mixed into that ranking.

- **`condor/execute.sh`'s micromamba bootstrap can hang for hours on a worker node that can't
  reach `micro.mamba.pm`, and the job then goes on hold instead of failing cleanly.** The
  `wget -L micro.mamba.pm/install.sh` call at `condor/execute.sh:40` has no `--timeout`/`--tries`,
  so on an unreachable node wget's own internal retry/backoff runs *inside* the script's existing
  10-attempt outer loop (`:39-46`) — observed in a held job's `.err` as 20 straight
  `Connection timed out` attempts over ~45 minutes. The script's shebang also lacks `-e`, so the
  chain of resulting failures (`chmod`/`install.sh` missing, `micromamba: command not found`) is
  silently swallowed until the explicit `|| exit 1` on the later `micromamba install python=3.10`
  line fires — by which point `.success` was never written. Since `.success` is a required
  `transfer_output_files` entry, HTCondor then holds the job with an opaque "Transfer output
  files failure" reason instead of the real cause. Confirmed against the two currently-held jobs
  in `jobs_CustomSummer24_2D_e_Wc_sf_data`, which show exactly this sequence. Affects every
  workflow submitted via `condor/submitter.py`, not something specific to `Wc_sf` — that batch is
  just the largest currently running, so it's where it's most visible.
  **Fix:** add `--timeout=20 --tries=1` to the `wget` call, keeping the existing outer
  retry/sleep loop as the retry mechanism, so a bad node fails each attempt in ~20s instead of
  minutes. **Risk:** low, setup-script only, no physics impact. **Files:** `condor/execute.sh`.
  **Status: implemented** (commit `922004c`).
- **Dataset file lists route through the global XRootD redirector (`root://cms-xrd-global.cern.ch`)
  even for files that are already on CERN EOS, adding redirect/discovery overhead per file-open
  across thousands of jobs.** `scripts/fetch.py` already has the machinery for this
  (`--redirector`, `test_site_redirector()`, `access_xrootd_file()`), it just wasn't used when the
  current `2D_*_Wc_sf` metadata JSONs were generated. **Explicitly not a replace:** the ask is to
  *prefer* a CERN-local/EOS path per file when it's reachable, and fall back to the existing
  global-redirector URL otherwise — `user.md:210` already documents the global redirector being
  occasionally insufficient and recommends `--retries`/`--skipbadfiles` as the existing mitigation,
  which must keep working for files where no local path resolves. Concretely: reuse
  `fetch.py`'s existing per-file redirector probe to try a local EOS door first at metadata-generation
  time, keeping the current global URL as the fallback when the probe fails, rather than hardcoding
  a single redirector. **Risk:** low if the fallback path is preserved; must not silently drop a
  file if neither resolves (same failure mode `--skipbadfiles` already handles). **Files:**
  `scripts/fetch.py`, regenerated `metadata/CustomSummer24/*_2D_*_Wc_sf_*.json`.
  **Status: implemented** (commit `5650752`).

**Remark:** neither item above was caught by the original four-review sweep because none of them
looked at `condor/`/`condor_lxplus/` at all.

**Status: the dedicated review pass has since been done, twice.** A first pass produced
`docs/condor_review.md` (combined architecture/simplify/performance, commit `7ae66a6`, same commit
as this plan). A second, deeper pass — one agent per review type, cross-checked against the first
pass to avoid duplicate findings — produced `docs/condor_architecture_review.md`,
`docs/condor_performance_review.md`, `docs/condor_io_review.md`, `docs/condor_simplify_review.md`.
Nothing from either pass has been actioned yet beyond the two items above; documenting only for now.
Highest-priority items surfaced, not yet fixed:

- **Correctness, not just performance:** `scripts/fetch.py`'s `process_single_dataset()` (used only
  by `--executor futures`) applies a `MuonEG`-specific dataset-name correction that the
  `--executor iterative` path (calling `getFilesFromDas()` directly) never sees — the same input
  dataset can produce different `metadata/*.json` keys depending purely on which executor flag was
  used to generate them. See `docs/condor_architecture_review.md` finding 2.
- `scripts/fetch.py:12` imports `get_xrootd_sites_map` from `utils/xrootdtools.py` then shadows it
  with a local redefinition at `fetch.py:190` — two copies of the SITECONF-parsing logic (also used
  by `condor_lxplus/checkoutputs.py`) that have already drifted apart. Flagged independently by both
  the architecture and simplify passes.
- Every job rebuilds its Python env from scratch (`condor/execute.sh`: micromamba bootstrap +
  `pip install -e .` pulling all of `coffea`'s deps + a separate `psutil` install) — flagged as top
  severity by both the io and performance passes. `condor_lxplus` avoids this via a prebuilt
  EOS-mounted env instead.
- `fetch.py --from_dataset` without `--from_workflow` hits an unguarded `NameError` (currently
  unexercised — every known call site passes both flags); a `--from_dataset`-without-`--from_workflow`-mode
  is presumably intentional and dormant. `condor/execute.sh`'s micromamba-bootstrap gate is
  always-true due to a `jq`-renders-unset-as-`"null"` vs. `!= ""` mismatch, inconsistent with the
  correctly-written `!= "null"` checks used elsewhere in the same file.

Full findings, including lower-severity items and what was confirmed sound, are in the four
`docs/condor_*_review.md` files above.
