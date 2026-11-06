# Architecture Review — BTVNanoCommissioning

Based on a structural investigation of `runner.py`, `workflows/`, `utils/correction.py`, `utils/AK4_parameters.py`, `utils/histogramming/`, `utils/array_writer.py`, and the `SFbc-2D` submodule boundary.

---

### 1. Duplicated histogram-filling dispatch (`histo_writter` vs. `ctag_Wctt_valid_sf.py`)

**Problem:** `histogrammer.py:58-641` implements a shared 580-line substring-dispatch function (`histo_writter`) for filling histograms, but `ctag_Wctt_valid_sf.py:475-657` (backing 7 of 31 registered workflows) bypasses it entirely and hand-rolls an independent, near-identical dispatch loop — including its own copy of the genflavor/smflav computation that also exists in `histogrammer.py:104-129`.

**Why it matters:** Two independently-maintained histogram-filling implementations for the same physics content is a direct physics-correctness risk (priority #1 in this repo's rules) — a fix or a new systematic added to one will silently not apply to the other. It also 2x's the maintenance surface for the single most safety-critical piece of logic in the framework.

**Proposed direction:** Not a rewrite — this needs a careful, workflow-owner-reviewed unification pass: diff the two dispatch chains histogram-by-histogram to find where they actually differ (if anywhere), then either extend `histo_writter` to cover Wc/ttdilep's extra cases or confirm it already does and switch `ctag_Wctt_valid_sf.py` over to calling it.

**Risk:** Medium-high — any behavior difference between the two paths, once unified, changes histogram content for 7 production workflows. Must be validated against existing reference outputs before merging, not just CI's `--limit 1`.

**Priority:** High.

---

### 2. `array_writer()` and `common_shifts()` take the workflow's `self` as an argument

**Problem:** `common_shifts(self, events)` (`correction.py:4059`) and `array_writer(processor_class, ...)` (`array_writer.py:129`) both accept the calling processor object itself, then reach into its private attributes (`self.SF_map`, `self._year`, `self._campaign`, `processor_class.name`, `.chunksize`). Dependency direction is inverted: a "utility" module is coupled to the internal attribute names of every `NanoProcessor` subclass.

**Why it matters:** It's a consistent pattern (all callers do it the same way), so it isn't broken today, but it means `utils/` isn't independently testable or reusable outside the exact `NanoProcessor` attribute contract, and any rename of a processor attribute silently breaks correction/array-writing code with no static check.

**Proposed direction:** Longer-term, pass explicit parameters (`SF_map`, `year`, `campaign`, `name`, `chunksize`) instead of `self`. Not urgent — defer unless touching these functions for another reason, since it's cosmetic/structural rather than a live bug.

**Risk:** Low risk to fix (signature-only change), but touches every workflow file's call sites — a wide, low-value diff for this repo's stated preference to avoid large rewrites.

**Priority:** Low — track but don't do proactively.

---

### 3. `array_writer.py`'s "generic" default path is not actually generic

**Problem:** The default/no-`schema`/no-`doOnly` branch of `array_writer()` (`array_writer.py:245-273`) infers branches via `np.setdiff1d`, but its default argument lists (`kinOnly`, `kins`, `othersData`, lines 138-175) hardcode Wc/DY-specific object names (`SoftMuon`, `MuonJet`, `dilep`, `OtherJets`). Three incompatible calling conventions exist (`doOnly=`, `schema=`, default-inference), used inconsistently across workflow files, with no documented rule for which a new workflow should use.

**Why it matters:** Directly increases risk of silent I/O correctness bugs (priority #2 in this repo's rules: reduce I/O, but never silently) — a new workflow author has no clear guidance and may pick a mode that silently drops or mis-includes branches for their physics case.

**Proposed direction:** Don't refactor the function now (recent working-tree changes to this exact file suggest active work in progress). Instead, once current changes settle, document the three modes' intended use cases in a comment/docstring, and flag to workflow authors which mode is canonical for new workflows. A deeper redesign (single explicit-branch-list API replacing all three modes) is worth considering later but is a larger change than warranted right now.

**Risk:** Low to document; medium if attempting consolidation given it's mid-edit.

**Priority:** Medium (documentation-only fix); defer structural consolidation.

---

### 4. `axes/qgtag.py` fully duplicates `axes/common.py`

**Problem:** `axes/qgtag.py:4-38` redefines all 24 axis keys already in `axes/common.py:4-29` with byte-identical `hist.axis.*` constructors, rather than importing `common.axes` and overriding only its 3 actually-different keys — despite `hist_helpers.py:23-44` already supporting dict-merge composition (`output = output | available_collections[ax]`) that would make this trivial.

**Why it matters:** Pure duplication with no functional divergence found — a bug fixed or bin range changed in `common.py` won't propagate to `qgtag.py`'s copies, and there's no test to catch the drift.

**Proposed direction:** Small, mechanical, low-risk fix: replace `qgtag.py`'s 21 duplicated axis definitions with `from .common import axes as common_axes; axes = common_axes | {qgtag-specific overrides}`. This is exactly the kind of "delete duplicated code" simplification this repo's rules prefer.

**Risk:** Very low — axis definitions are declarative (bin edges/labels), easy to diff before/after to confirm zero behavior change.

**Priority:** Medium — safe win, small effort.

---

### 5. `ttbar_reweights` propagated through two parallel channels

**Problem:** `runner.py:414` sets `os.environ["BTV_TTBAR_REWEIGHTS"]`, and separately `runner.py:540` does `setattr(processor_instance, "ttbar_reweights", args.ttbar_reweights)` post-construction (not via `__init__`). `weight_manager()` in `correction.py:4216-4217` falls back to the env var only when the constructor-injected value is `None`. Workflow files then defensively read it via `getattr(self, "ttbar_reweights", "none")`.

**Why it matters:** Two channels carrying the same config value, with a fallback between them, is exactly the kind of fragile coupling that can silently produce wrong physics if the two ever disagree (e.g., env var leaks across a multiprocessing/dask worker boundary that doesn't inherit the parent's constructed attribute, or vice versa) — a serious concern for priority #1 (never change physics behaviour silently).

**Proposed direction:** Investigate why `ttbar_reweights` isn't just a normal `NanoProcessor.__init__` constructor argument like `isSyst`/`isArray`/`campaign` — if there's no reason (e.g., no dask-serialization constraint forcing the env-var path), consolidate to constructor-only and drop the env var. If dask serialization *is* the reason, that constraint should be a comment at the env-var line, since it isn't currently documented.

**Risk:** Medium — must verify across all executors (iterative/dask/condor) that a constructor-only path behaves identically before removing the env var fallback.

**Priority:** Medium-high — worth a focused, careful fix given the physics-correctness angle, but requires validation across executors first.

---

### 6. Three independently-maintained argparse definitions (`runner.py`, `suball.py`, `condor_lxplus/submitter.py`)

**Problem:** `scripts/suball.py:11` imports `runner.py`'s parser-builder functions directly, then re-serializes the merged `Namespace` back into a CLI string via a hand-maintained skip-list (`suball.py:236-259`) to invoke `runner.py` as a subprocess — a hybrid double-coupling. `condor_lxplus/submitter.py` builds an entirely separate, third `argparse.ArgumentParser` from scratch.

**Why it matters:** Any new flag added to `runner.py`'s parsers must be manually remembered to add to `suball.py`'s skip-list or it leaks through incorrectly; `submitter.py` can silently drift out of sync with `runner.py`'s actual flags with no shared source of truth. This is a real submission-pipeline fragility risk, though not a physics-correctness one.

**Proposed direction:** No urgent action — this is how CI and submission currently work correctly, so it's a maintenance-burden observation, not a bug report. If touching `runner.py`'s argument list for another reason, flag the corresponding `suball.py` skip-list update as a required companion change in that PR.

**Risk:** N/A (observation only, not proposing a change now).

**Priority:** Low — track for awareness, don't refactor proactively.

---

## Not flagged as problems (explicitly checked, found sound)

- **workflows/__init__.py plugin registry**: the `functools.partial(SharedProcessor, selectionModifier=...)` pattern covering 20 of 31 entries is a consistent, appropriate plugin design — not raised as an issue.
- **`AK4_parameters.py`**: clean, centralized, data-driven config; the one real leak (campaign-conditional branching in `utils/selection.py:30,47,92,129,171,221`) is minor and physics-selection logic legitimately needs some campaign awareness in the selection layer itself.
- **No circular imports** between `utils/` and `workflows/` — layering is directionally correct there. `helpers/update_branch.py` importing from `utils/selection.py` is a minor layering inversion, not a cycle — not worth fixing on its own.
- **SFbc-2D boundary**: mixed coupling (some live imports, some intentionally-copied-and-commented code) is a reasonable tradeoff for a submodule with its own independent tooling per your CLAUDE.md — flagged for awareness only, not as something to fix.
