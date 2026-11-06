# Awkward2 / Coffea Migration — Simplification Findings

Investigation only. No changes made. Baseline versions checked against: `coffea==0.7.31`,
`awkward 1.10.5`, `uproot 4.3.7`, `correctionlib==2.7.0`, `numpy 1.26.4`, `numba 0.66.0`,
`python<3.11`.

Scope: awkward1-specific idioms that need per-callsite changes under awkward2, and whether any of
the repo's own abstractions become unnecessary once on awkward2.

---

### `helpers/BTA_helper.py:147-156` (`cumsum`)

**Current complexity:** manual low-level layout surgery (`array.layout`,
`ak.layout.ListOffsetArray64`, `ak.layout.NumpyArray`) that exists purely because awkward1 lacked
a clean high-level cumulative-sum-per-jagged-list idiom.

**Simplification found:** since this must be rewritten for awkward2 regardless (the `ak.layout`
API doesn't exist there), it's worth checking at that time whether a high-level
`np.cumsum`-plus-`ak.unflatten` round trip (or a first-class `ak.cumsum` if available in the
target release) replaces the whole layout-object dance, rather than porting the low-level
construction 1:1 to `ak.contents.*`.

**Risk:** low to simplify, but the replacement's output must be verified bit-identical to the
current one before it's trusted — this function feeds impact-parameter output consumed downstream.

---

### `utils/histogramming/*`, `utils/plot_utils.py`

**Current complexity:** none found. Already on the modern `hist` package.

**Simplification found:** nothing — no awkward1-era workaround exists here to clean up.

**Risk:** none.

---

### `runner.py` executor dispatch (~600-1150)

**Current complexity:** roughly 500+ lines with substantial duplication — `HighThroughputExecutor(...)`
constructed with near-identical config ~7 times across different sites, and repeated
`HTCondorCluster`/`LPCCondorCluster` branches.

**Simplification found:** this duplication isn't awkward-migration debt on its own, but the modern
coffea `Runner` API takes a more declarative `executor=`/`client=` configuration, which would
collapse much of this duplication as a side effect of any executor-layer rewrite this migration
requires. Noting it here as an observed-but-untouched opportunity, not a separate cleanup to do
independently.

**Risk:** medium if pursued — avoid introducing a new configuration abstraction beyond what's
needed; prefer deleting duplicated blocks over building a new wrapper layer.

---

### `array_writer.py`

**Current complexity:** none found related to awkward1. All operations used
(`ak.num`, `ak.broadcast_arrays`, `ak.flatten`, `ak.zip`, `ak.all`, `ak.sum`) are high-level and
already portable-style calls.

**Simplification found:** nothing to simplify pre-migration.

**Risk:** none directly; general broadcasting-rule risk is noted separately in the architecture
findings.

---

### `utils/selection.py`

**Current complexity:** none found. Targeted grep for `ak.layout`, `ak.behavior`,
`register_behavior`, `ak.mixin_class`, and raw `.layout` access returned no hits in this file.

**Simplification found:** nothing — this file already appears awkward1/2-agnostic.

**Risk:** none.

---

### Repo-wide check: no partial-migration or legacy-shim leftovers

Grep across `src/`, `runner.py`, `condor*/`, `scripts/` for `JaggedArray`, `awkward0`, `uproot3`
returned zero hits. There is no dead code from a prior, abandoned migration attempt to remove as
part of this work.

---

### Net observation

The awkward1→2 *code-level* footprint is smaller than the coffea-*version* footprint. Only one
confirmed must-rewrite call site was found (`BTA_helper.py`'s `cumsum`); the rest of the codebase's
awkward usage is either already high-level/portable, or needs verification rather than rewriting.
