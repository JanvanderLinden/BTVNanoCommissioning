#!/bin/bash -xe
# Condor job of submit_merge.py: mergeoutputs.py for one output directory, with the conda
# environment from the sandbox tarball (as execute.sh), nothing read from AFS.
#   merge_job.sh <outputDir> <name> <only: hists|arrays|all> <jobs> <hadd threads> <hist workers>

OUTPUTDIR=$1
NAME=$2
ONLY=$3
JOBS=$4
HADD_THREADS=$5
HIST_WORKERS=$6

WORKDIR=$(pwd)
export HOME=$WORKDIR
export XDG_CACHE_HOME=$WORKDIR/.cache
mkdir -p $XDG_CACHE_HOME $WORKDIR/tmp
# nothing large is left in the scratch directory (condor would transfer it back)
trap 'rm -rf $WORKDIR/env $WORKDIR/tmp $WORKDIR/.cache' EXIT

setup_failed() { echo "ERROR: merge job setup failed: $1" >&2; exit 99; }
ENV_TAR=$(ls $WORKDIR/btv_env_*.tar.gz 2>/dev/null | head -1)
[ -n "$ENV_TAR" ] || setup_failed "no environment tarball"
mkdir -p $WORKDIR/env
tar -xzf $ENV_TAR -C $WORKDIR/env || setup_failed "unpacking $(basename $ENV_TAR)"
rm -f $ENV_TAR
set +x
source $WORKDIR/env/bin/activate || setup_failed "activating the environment"
conda-unpack || setup_failed "conda-unpack"
set -x
for sp in $WORKDIR/env/lib/python3.*/site-packages; do
    [ -d "$sp" ] || continue
    rm -f $sp/__editable__*BTVNanoCommissioning* $sp/__editable__*btvnanocommissioning* \
          $sp/BTVNanoCommissioning.egg-link
done
export PYTHONDONTWRITEBYTECODE=1
export X509_USER_PROXY=${X509_USER_PROXY:-$WORKDIR/proxy}
voms-proxy-info || true
python -c "import coffea, uproot, fsspec, lz4, cloudpickle" || setup_failed "python modules"
command -v eos >/dev/null || setup_failed "no eos client on this node"
command -v xrdcp >/dev/null || setup_failed "no xrdcp on this node"

OPTS=(--name "$NAME" -j "$JOBS" --hadd-threads "$HADD_THREADS" --hist-workers "$HIST_WORKERS"
      --tmpdir "$WORKDIR/tmp")
[ "$ONLY" != "all" ] && OPTS+=(--only "$ONLY")
python -u $WORKDIR/mergeoutputs.py "$OUTPUTDIR" "${OPTS[@]}"
