# Awkward2 / Coffea Migration — Refactor Feasibility Findings

Investigation only. No changes made. Baseline versions checked against: `coffea==0.7.31`,
`awkward 1.10.5`, `uproot 4.3.7`, `correctionlib==2.7.0`, `numpy 1.26.4`, `numba 0.66.0`,
`python<3.11`.

Scope: whether the existing validation harness is sufficient, blocking dependency relationships,
and effort tiers per subsystem. No sequencing or execution plan — that is out of scope for this
report.

---

### Finding: CI alone is not sufficient; a suitable harness already exists

`.github/workflows/*_workflow.yml` runs `runner.py --executor iterative --limit 1`, which only
proves the code runs without crashing — it says nothing about physics output. That gap is why
`docs/modernization_plan.md`'s "Phase 0" harness was built: `scripts/compare_coffea.py` +
`scripts/phase0_regression.sh` already bin-diff `.coffea` histograms (categorical axes matched by
label) and `--isArray` ROOT ntuple branches (jagged-count-aware) against a stored baseline.

This harness directly applies to validating an awkward2/coffea migration — it doesn't need to be
rebuilt, only pointed at old-environment vs. new-environment output for the same fixed input files.

---

### Finding: dependency chain has no hard blockers found

- `correctionlib==2.7.0` — independent of awkward version; `evaluate()` operates on flat numpy
  arrays. A bump may separately be desirable for newer POG JSON schema versions, but nothing found
  ties it to the awkward1→2 transition directly.
- `vector` (the standalone scikit-hep package) — not imported anywhere in the repo (zero grep
  hits); all vector behavior is routed through `coffea.nanoevents.methods.vector`, so this tracks
  the coffea version rather than being an independent dependency to manage.
- `numba 0.66.0` — aging. A numpy bump (implied by modern awkward2/coffea) will likely require a
  numba bump alongside it. A minor additional link in the chain, not a blocker.
- `SFbc-2D` submodule and `condor_lxplus/` scripts consume `array_writer.py`'s ROOT ntuple *output
  format*, not awkward/coffea internals directly. As long as Phase 0's ROOT-branch diff stays
  clean, these consumers are not directly coupled to the migration.

---

### Finding: effort tiers by subsystem

| Subsystem | Tier | Basis |
|---|---|---|
| `utils/histogramming/` | S | Already on modern `hist`; verification only. |
| `array_writer.py` | S | High-level ops only; no awkward1-specific code found. |
| `utils/selection.py` | S | No awkward1-specific patterns found. |
| `utils/correction.py` | M | Large file, several `coffea.*` imports need per-signature re-verification, but no raw-layout code found — a verification pass, not a rewrite. |
| `helpers/BTA_helper.py` (`cumsum`, vector behaviors) | M | One confirmed hard rewrite (`cumsum`), localized but physics-sensitive (impact-parameter calculation). |
| `runner.py` executors | L-XL | Full rewrite of the scaleout dispatch layer; parsl executors have no direct successor (a site-support question, not only code); `condor_standalone` needs porting to the modern `Runner`+`DaskExecutor` model. |
| `condor/`, `condor_lxplus/` scripts | S-M | Consume ROOT/`.coffea` outputs rather than awkward/coffea internals directly; exposure limited to whatever `runner.py` CLI-surface changes ripple out. |

---

### Finding: physics-risk flags requiring explicit validation (not inspection)

- `helpers/BTA_helper.py`'s `cumsum()` rewrite — must be checked bit-identical against the current
  implementation's output, not reviewed by eye.
- Awkward2's stricter None/option-type broadcasting rules interacting with `array_writer.py`'s
  `ak.broadcast_arrays`/`ak.all(arr_counts == ref_counts)` masking logic — subtle enough that it
  needs the Phase 0 harness's bin-level diff to confirm, not a code read.
- Vector-behavior class resolution (`with_name=` sites) — needs an explicit per-class-name check
  under the target coffea release rather than an assumption of backward compatibility.
