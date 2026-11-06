# Awkward2 / Coffea Migration — Architecture Findings

Investigation only. No changes made. Baseline versions checked against: `coffea==0.7.31`,
`awkward 1.10.5`, `uproot 4.3.7`, `correctionlib==2.7.0`, `numpy 1.26.4`, `numba 0.66.0`,
`python<3.11` (`setup.cfg`, `test_env.yml`).

This migration was previously flagged, not planned, in the "Strategic note" at the bottom of
`docs/modernization_plan.md`, which recommended a timeboxed spike before committing to it. This
document is the architecture-side findings feeding that question.

---

### Finding: `runner.py`'s executor layer is built entirely on removed coffea 0.7 API

`processor.run_uproot_job`, `processor.iterative_executor`, `processor.futures_executor`,
`processor.parsl_executor`, `processor.dask_executor`, and parsl's `HighThroughputExecutor`
(`runner.py:626-1150`, ~100 call sites) are coffea-0.7-only. Modern coffea (2023.x+) replaced this
whole model with a `coffea.processor.Runner` object plus `IterativeExecutor`/`FuturesExecutor`/
`DaskExecutor` executor classes, and dropped parsl support entirely.

**Why it matters:** this is the largest single surface in the repo coupled to the coffea version.
`--executor parsl/slurm`, `parsl/condor`, `parsl/condor/naf_lite` have no direct successor under
modern coffea — sites currently using those executors would need to move to `dask/*` equivalents,
which is a site-support question, not purely a code change.

**Risk:** high on scope, medium on physics (this is orchestration/dispatch, not event-level math).

---

### Finding: `condor_standalone` executor rests on the same removed API

Built on `dask_jobqueue`/`lpcjobqueue` `HTCondorCluster`/`LPCCondorCluster` plus
`processor.dask_executor` (`runner.py:1013-1142`). Per `docs/modernization_plan.md` item 7, the
`2D_*_Wc_sf` workflows run via this executor in production.

**Risk:** medium — porting to `Runner` + `DaskExecutor` with the same cluster classes is coffea's
own documented pattern, but it is production-load-bearing code.

---

### Finding: histogramming is already on modern `hist`, not `coffea.hist`

`utils/histogramming/*` imports `hist` (scikit-hep) everywhere. The only `coffea.hist` reference
in the repo is a comment (`utils/plot_utils.py:271`); the one live `hist.Hist(...)` construction
(`utils/plot_utils.py:881`) is already the modern package.

**Why it matters:** this is the piece every generic "coffea migration" writeup assumes is the big
lift. Here it's already done — no action needed in this area.

---

### Finding: `helpers/BTA_helper.py` `cumsum()` uses raw awkward1 layout internals

```
helpers/BTA_helper.py:147-156
    layout = array.layout
    ...
    ak.layout.ListOffsetArray64(layout.offsets, ak.layout.NumpyArray(np.cumsum(layout.content)))
```

`ak.layout` does not exist in awkward2 (restructured into `ak.contents.*` with different
constructor semantics). This function feeds `calc_ip_vector()`, which computes the impact
parameter used in BTA ntuple production — physics content consumed downstream by the `SFbc-2D`
submodule.

**Risk:** high — this is exactly the class of silent-physics-change risk the repo's CLAUDE.md
rules call out. Any replacement must be validated bit-for-bit against a reference-output baseline
before being trusted.

---

### Finding: `NanoEventsFactory`/`PFNanoAODSchema` usage needs schema-compat verification

`runner.py`, `workflows/sf_ttdilep_kin.py` import `PFNanoAODSchema` from `coffea.nanoevents`. The
class itself still exists in modern coffea, but this repo's custom-branch backfill
(`helpers/update_branch.py`'s `missing_branch`) has not been checked against the newer schema's
validation behavior.

**Risk:** medium — custom NanoAOD branches are central to this framework; a schema behavior change
here is the second most physics-sensitive spot in the codebase after selection logic.

---

### Finding: vector-behavior usage (`with_name=`) needs per-class verification

21 call sites across `helpers/BTA_helper.py`, `workflows/BTA_producer.py`, and most other
workflows use `with_name="PtEtaPhiMLorentzVector"`, `"FourVector"`, `"ThreeVector"`,
`"PtEtaPhiECandidate"`, all resolved via `coffea.nanoevents.methods.vector`
(`vector.behavior=`, e.g. `BTA_helper.py:170,179,196`, `BTA_producer.py:619,881`).

**Why it matters:** awkward2 changed behavior-class registration mechanics (dict-merge order,
global `ak.behavior` mutation rules). These class names are believed retained for backward
compatibility in modern coffea, but this is unverified against the target release.

**Risk:** medium — most likely "just works," but per the physics-preservation priority, "most
likely" isn't sufficient without an explicit check per class name used.

---

### Finding: `utils/correction.py`'s coffea imports changed less than the executor layer

`coffea.lookup_tools.{extractor, txt_converters, rochester_lookup}`, `coffea.lumi_tools.LumiMask`,
`coffea.analysis_tools.Weights`, `coffea.btag_tools.BTagScaleFactor` (`correction.py:21-24`). These
subpackages have historically been more stable across coffea releases than the processor/executor
layer, though `rochester_lookup` in particular has had signature changes historically.

**Why it matters:** this module carries almost all SF/correction logic — highest-value,
highest-risk code in the framework.

**Risk:** medium — likely a version-bump-and-verify exercise per import rather than a rewrite, but
each signature needs individual confirmation, not an assumption.

---

### Finding: `array_writer.py` uses only high-level, largely portable `ak` operations

`ak.num`, `ak.broadcast_arrays`, `ak.flatten`, `ak.zip`, `ak.all`, `ak.sum` — no raw layout access,
no `ak.behavior` manipulation. Awkward2 did tighten broadcasting/None-handling rules for
option-typed arrays relative to awkward1, which is a subtle rather than structural risk here.

**Risk:** low-medium, not structural. The uncommitted local diff to this file (adding one `osss`
schema field) is unrelated to this migration and was not touched during this investigation.

---

### Finding: no leftover partial-migration artifacts found

Targeted search for `JaggedArray`, `awkward0`, `uproot3`, `ak.layout` (outside `BTA_helper.py`),
and `ak.behavior` registration patterns turned up nothing beyond what's listed above. There is no
abandoned prior migration attempt to account for.
