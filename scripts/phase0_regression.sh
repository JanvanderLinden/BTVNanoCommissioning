#!/bin/bash
# Phase 0 regression check (docs/modernization_plan.md).
#
# Runs 2D_mu_Wc_sf and 2D_emu_ttdilep_sf locally (iterative executor) over the
# small test filesets already used by SFbc-2D/01_test_workflows.sh, plus
# BTA_addPFMuons/BTA_addAllTracks over metadata/test_bta_run3.json, then diffs
# the resulting .coffea histograms and ROOT ntuples bin-by-bin
# (scripts/compare_coffea.py) against a stored baseline. Run this before/after
# any change to a hot path that touches physics output (weight_manager,
# btagSFs, histo_writter, BTA_helper's cumsum()/calc_ip_vector(), ...).
#
# Usage:
#   scripts/phase0_regression.sh baseline   # (re)create the reference snapshot
#   scripts/phase0_regression.sh check      # run again and diff against it
#
# Both MC and data are run for both workflows, and every run writes both
# histograms and --isArray ROOT ntuples, so compare_coffea.py always diffs
# both output kinds -- this is not optional.
#
# Env vars (all optional):
#   ISSYST      isSyst mode for the MC samples (default: all -- this is the
#               expensive setting that actually exercises the btagSFs/
#               weight_manager hot path; drop to JEC_reduced_JER_split for a
#               faster iteration loop at the cost of not covering that path)
#   CHUNK       chunk size (default: 10000)
#   MAXCHUNKS   max chunks per file -- bounds local runtime regardless of how
#               large the backing NanoAOD file is (default: 2)

set -e

if [ "$#" -ne 1 ] || { [ "$1" != "baseline" ] && [ "$1" != "check" ]; }; then
  echo "usage: $0 baseline|check"
  echo "  env vars: ISSYST (default: all), CHUNK (default: 10000),"
  echo "            MAXCHUNKS (default: 2)"
  exit 1
fi
mode="$1"

ISSYST="${ISSYST:-all}"
CHUNK="${CHUNK:-10000}"
MAXCHUNKS="${MAXCHUNKS:-2}"

cd "$(dirname "$0")/.."

campaign="Summer24"
year="2024"
outdir="phase0_${mode}"
rm -rf "${outdir}"

run() {
  local wf="$1" json="$2" tag="$3" issyst="$4"
  python runner.py \
    --workflow "${wf}" --year "${year}" --campaign "${campaign}" \
    --executor iterative --chunk "${CHUNK}" --max "${MAXCHUNKS}" \
    --isSyst "${issyst}" --ttbar-reweights full --overwrite --isArray \
    --json "SFbc-2D/metadata/${campaign}/${json}.json" \
    -o "hists_${wf}_${tag}.coffea" --outputdir "${outdir}"
}

# BTA_producer.py: sole consumer of BTA_helper.py's cumsum()/calc_ip_vector(),
# the most physics-sensitive function in the awkward2 migration
# (docs/awkward2_migration_plan.md, correction #5) and previously uncovered by
# this harness. Its normal I/O path xrdcp's output to a shared CERN EOS area
# and deletes the local copy (BTA_producer.py:1150-1180) -- BTA_LOCAL_OUTPUT=
# (a directory) keeps the ntuple there instead of shipping/deleting it, and
# skips the remote "already produced" check that would otherwise
# short-circuit the run. Note: BTA_producer writes dataset/*.root relative to
# BTA_LOCAL_OUTPUT rather than under runner.py's --outputdir, unlike every
# other workflow -- do not `cd` into a subdir to work around this, since
# workflows/__init__.py imports sf_ttsemilep_tnp.py, which opens
# helpers/sf_ttsemilep_likelihoods_pas.root via a path relative to the
# process's cwd at import time and breaks if cwd isn't the repo root.
# Mirrors BTA_workflow.yml's dataset/campaign/year exactly. addPFMuons covers
# cumsum() (2 call sites) and calc_ip_vector()'s addPFMuons-only call sites;
# addAllTracks covers cumsum()'s third call site (JP calibration Track
# collection).
run_bta() {
  local wf="$1"
  local bta_dir="${outdir}/${wf}"
  mkdir -p "${bta_dir}"
  BTA_LOCAL_OUTPUT="${bta_dir}" python runner.py \
    --workflow "${wf}" --year 2023 --campaign Summer23 \
    --executor iterative --chunk "${CHUNK}" --max "${MAXCHUNKS}" \
    --overwrite \
    --json "metadata/test_bta_run3.json" \
    -o "hists_${wf}.coffea" --outputdir "${bta_dir}"
}

# MC: exercise the full systematic hot path (items 1 and 4 in modernization_plan.md).
run 2D_mu_Wc_sf       test_mc_WJets mc "${ISSYST}"
run 2D_emu_ttdilep_sf test_mc_TT2L  mc "${ISSYST}"
# Data: no MC weight systematics, but exercises the lumiMask/real-data path.
run 2D_mu_Wc_sf       test_data_mu  data False
run 2D_emu_ttdilep_sf test_data_emu data False
run_bta BTA_addPFMuons
run_bta BTA_addAllTracks

if [ "${mode}" == "baseline" ]; then
  # phase0_baseline/ is gitignored (matches hists_*/arrays_*/*.coffea), so this
  # stamp is the only record of which code state it reflects.
  {
    echo "commit: $(git rev-parse HEAD)"
    echo "branch: $(git rev-parse --abbrev-ref HEAD)"
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if git diff --quiet && git diff --cached --quiet; then
      echo "dirty: no"
    else
      echo "dirty: yes (uncommitted changes were present at snapshot time)"
    fi
  } > "${outdir}/PHASE0_BASELINE_COMMIT"
  pip freeze > "${outdir}/PHASE0_BASELINE_PIP_FREEZE"
fi

if [ "${mode}" == "check" ]; then
  echo
  if [ -f phase0_baseline/PHASE0_BASELINE_COMMIT ]; then
    echo "=== Baseline snapshot info ==="
    cat phase0_baseline/PHASE0_BASELINE_COMMIT
    echo
  fi
  echo "=== Diffing phase0_check against phase0_baseline ==="
  python scripts/compare_coffea.py phase0_baseline phase0_check
fi
