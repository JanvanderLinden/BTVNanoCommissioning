# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CMS BTV POG commissioning framework: derives b-/c-tagging scale factors (SFs) and produces
validation/commissioning plots from (custom) NanoAOD. Built on `coffea` processors
(`coffea==0.7.31`, `correctionlib==2.7.0`). Each analysis region/method is a `coffea` **workflow**
(processor class) mapping NanoAOD events to histograms (`.coffea`) or flattened ROOT ntuples.
The repo is also used as the frontend for the GitLab `autobtv` automation pipeline.

## Setup

```bash
# env is normally pre-provisioned on lxplus at /eos/home-m/milee/miniforge3/envs/btv_coffea
conda/micromamba activate btv_coffea
pip install -e .            # or -e .[dev,doc] for dev/docs extras
source env_setup_complete.sh  # ensures `dasgoclient` is on $PATH
```
Standalone env can be created from `test_env.yml` via micromamba if not on lxplus.

Python `<3.11` is required (see `setup.cfg`). This repo has a git submodule, `SFbc-2D` (2D SF
derivation pipeline — see below); it has its own separate context/tooling.

## Running a workflow

```bash
python runner.py --workflow example --json metadata/test_bta_run3.json --campaign Summer23 --year 2023
```
Key `runner.py` flags (grouped by parser function in `runner.py`):
- `IO_parser`: `--workflow/--wf` (must be a key in `workflows` dict), `-o/--output`, `--outputdir`, `--json` (dataset fileset JSON)
- `config_parser`: `--year`, `--campaign` (selects correction files, see `AK4_parameters.py`), `--isSyst` (`False|all|weight_only|JEC_full|JEC_reduced|JEC_reduced_JER_split|JEC_total|JP_MC`), `--ttbar-reweights` (`none|hdamp_ml|full`), `--isArray` (write ROOT ntuples instead of/alongside histograms), `--noHist`, `--overwrite`
- `scaleout_parser`: `--executor` (`iterative|futures|parsl/*|dask/*|condor_standalone`), `-j/--workers`, `-s/--scaleout`, `--memory`, `--disk`, `--voms`, `--chunk`, `--retries`, `--fsize`, `--index` (dask/lxplus file-splitting)
- `debug_parser`: `--validate` (just check file accessibility), `--skipbadfiles`, `--only`, `--skip-structure-validation`

There is no unit test suite (no `pytest`/`test_*.py`); correctness is validated by CI
(`.github/workflows/*_workflow.yml`) actually running `runner.py` against small sample datasets
(e.g. `metadata/test_bta_run3.json`) with `--executor iterative --limit 1`. When changing
`workflows/`, `utils/correction.py`, `utils/selection.py`, or `helpers/`, check which workflow CI
files (`ttbar_workflow.yml`, `ctag_*_workflow.yml`, `QCD_workflow.yml`, `BTA_workflow.yml`,
`validation.yml`) match the touched paths, and run the equivalent `runner.py` command locally
with `--limit 1 --executor iterative` before pushing. Commit messages can include
`ci:skip array`, `ci:skip syst`, `ci:JEC_reduced`, `ci:weight_only` to adjust which CI checks run.

Linting: `flake8` (see `[flake8]` in `setup.cfg` for ignored codes) and `black` (`psf/black`,
enforced in `.github/workflows/python_linting.yml`). Run `black .` before committing.

## Architecture

### `runner.py`
CLI entry point. Builds the `fileset` from the `--json` dataset file, optionally validates files
(`validate_dataset_structure`), selects the processor from `workflows` dict, and dispatches to a
`coffea.processor` executor (iterative/futures/parsl/dask/condor). `scripts/suball.py` and
`condor/`/`condor_lxplus/` submission scripts import parser helpers directly from `runner.py`.

### `src/BTVNanoCommissioning/workflows/`
One file per analysis; each defines a `NanoProcessor(coffea.processor.ProcessorABC)` with
`__init__(year, campaign, name, isSyst, isArray, noHist, chunksize)`, `process()` (applies
JES/JEC/JER shifts via `common_shifts`, dispatches to `process_shift` per shift), and
`process_shift()` which does: histogram init (`histogrammer`) → selections (using
`utils/selection.py` helpers) → object pruning → SF/weight application (`weight_manager`) →
histogram filling (`histo_writer`) → optional ROOT ntuple output (`array_writer`).
`src/BTVNanoCommissioning/workflows/example.py` is the canonical template for new workflows.
Many workflows are shared processors parametrized via `functools.partial(..., selectionModifier=...)`
registered in `workflows/__init__.py` — that dict's keys are exactly the valid `--workflow` values.
Adding a new workflow means: (1) import + register it in `workflows/__init__.py` (put `WP` in the
name if it's a working-point SF workflow), (2) add histogram axes/collections under
`utils/histogramming/`, (3) implement selections in the processor. Full walkthrough in
`docs/developer.md`.

Workflow families: plain b-tag/c-tag validation SF (`ctag_*_valid_sf`, `ttdilep_valid_sf`,
`ttsemilep_valid_sf`), 2D pseudo-continuous-tagger SF (`_2D_*_valid_sf`, `--workflow 2D_*`),
QCD/soft-mu validation, quark/gluon tagging (`dijet.py`, `photonjet.py`), `BTA_producer*` (BTA
ntuple production for offline SF derivation), `DY_sfl.py` (negative-tag light-flavor SF).

### `src/BTVNanoCommissioning/utils/`
- `correction.py` (large): loads/applies all corrections and SFs — JES/JEC/JER shifts
  (`JME_shifts`), rochester muon corrections, MUO/EGM SFs, pileup, b-tag SFs (`btagSFs`),
  PDF/scale/PS event weights, ttbar reweighting. `weight_manager()` is the single entry point
  workflows call to get a fully-configured `coffea.analysis_tools.Weights` object.
- `selection.py`: shared object ID / kinematic selection functions (jet/muon/electron IDs,
  `HLT_helper`, `btag_wp`/`wp_dict` for working-point cuts), used across workflows.
- `AK4_parameters.py`: per-campaign/year correction file configuration (`correction_config`) —
  changing campaign/year support means editing this.
- `sample.py`: per-workflow refined MC sample lists (`predefined_sample`), used by `scripts/suball.py`/`fetch.py`.
- `array_writer.py`: flattens pruned per-event awkward arrays to per-jet ROOT ntuples (`--isArray`).
- `histogramming/`: `histogrammer.py` builds the histogram dict from named `hist_collections`
  (defined per-topic in `histogramming/histograms/*.py`) and shared `histogramming/axes/*.py`;
  `histo_writer` fills them per-systematic-shift.

### `src/BTVNanoCommissioning/helpers/`
`func.py` (misc utilities: `flatten`, `update`, `uproot_writeable`, `dump_lumi`),
`update_branch.py` (`missing_branch` — backfills branches absent in older NanoAOD),
`xsection(_13TeV).py` (cross-section tables), `BTA_helper.py` (BTA-workflow-specific tools),
`MuonScaRe.py` (muon scale/resolution corrections), `definitions.py` (histogram naming).

### `src/BTVNanoCommissioning/data/`
Correction inputs organized by POG (`BTV`, `MUO`, `EGM`, `JME`, `LUM`, `JPCalib`, `DC`,
`Prescales`, `TTBAR_REWEIGHT`), mirroring `jsonpog-integration` layout, split by campaign.

### `metadata/`
Dataset fileset JSONs (`--json` argument), organized by campaign directory, plus small
`test_*.json` files used by CI.

### `scripts/`
`fetch.py` (build dataset JSON via DAS), `suball.py` (all-in-one commissioning/data-MC-check
driver, reuses `runner.py` parsers), post-processing (`make_template.py` coffea→ROOT,
`dohadd.py`, `dump_prescale.py`, `dump_processed.py`, `missingFiles.py`) and plotting
(`plotdataMC.py`, `comparison.py`, `validation_plot.py`, `2Dhistograms.py`, `correlation_plots.py`).

### `condor/` and `condor_lxplus/`
Standalone condor submission (not the CI-only executors in `runner.py --executor condor_standalone`).
`condor_lxplus/` is the lighter-weight variant relying on eos/afs mounts on condor nodes (good for
up to a few hundred files; see `condor_lxplus/README.md` for the submit → checkoutputs → haddoutputs
→ normalizeArrays → haddFinal pipeline).

### `SFbc-2D/` (git submodule)
Separate 2D (pseudo-continuous tagger) SF derivation pipeline downstream of this framework's
`--isArray` ROOT outputs — has its own scripts (`01_..sh`–`10_..sh`, `A_..sh`–`XX_..sh`) and is
largely independent tooling; treat as a distinct project when working inside it.

## Conventions
- Match `black` formatting; don't hand-format around it.
- `--campaign`/`--year` select correction files — most bugs in SF/correction application trace to
  `AK4_parameters.py` config for that campaign, not the correction code itself.
- Workflow names are the contract with CI and `condor_lxplus`/`condor` submission scripts — don't
  rename an existing entry in `workflows/__init__.py` without checking `.github/workflows/*.yml`
  and `condor*/` usages.


# Priorities:

1. Preserve physics results
2. Reduce I/O
3. Reduce runtime
4. Reduce memory usage
5. Simplify architecture
6. Reduce maintenance burden

# Rules:

- Never change physics behaviour silently.
- Never remove systematic variations.
- Never change event selection logic without explicit validation.
- Prefer deleting code over adding abstractions.
- Avoid large rewrites.
- Make small measurable improvements.
- Ignore formatting-only improvements.
- Ignore naming-only improvements.

# When suggesting optimizations:

Always provide:

- expected benefit
- implementation complexity
- risk
- affected files