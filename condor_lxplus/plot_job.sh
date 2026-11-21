#!/bin/bash -xe
# Condor job of submit_plots.py: scripts/plotdataMC.py for one histogram, with the conda
# environment and the code from the sandbox tarballs (as execute.sh), nothing read from AFS.
#   plot_job.sh <variable> <phase> <file with the input dir URLs> <output dir URL>
# Input: <input dir URL>/<variable>.coffea of every input directory that has it.
# Further plotdataMC.py options: plot_opts.txt (one argument per line).

VAR=$1
PHASE=$2
INDIRS=$3
OUTURL=$4

WORKDIR=$(pwd)
export HOME=$WORKDIR
export XDG_CACHE_HOME=$WORKDIR/.cache
export MPLCONFIGDIR=$WORKDIR/.cache/matplotlib
export NUMBA_CACHE_DIR=$WORKDIR/.cache/numba
mkdir -p $XDG_CACHE_HOME $MPLCONFIGDIR $NUMBA_CACHE_DIR
# nothing large is left in the scratch directory (condor would transfer it back)
trap 'rm -rf $WORKDIR/env $WORKDIR/code $WORKDIR/.cache' EXIT

setup_failed() { echo "ERROR: plot job setup failed: $1" >&2; exit 99; }
ENV_TAR=$(ls $WORKDIR/btv_env_*.tar.gz 2>/dev/null | head -1)
CODE_TAR=$(ls $WORKDIR/btv_code_*.tar.gz 2>/dev/null | head -1)
[ -n "$ENV_TAR" ] && [ -n "$CODE_TAR" ] || setup_failed "no sandbox tarballs"
mkdir -p $WORKDIR/env $WORKDIR/code
tar -xzf $ENV_TAR -C $WORKDIR/env || setup_failed "unpacking $(basename $ENV_TAR)"
tar -xzf $CODE_TAR -C $WORKDIR/code || setup_failed "unpacking $(basename $CODE_TAR)"
rm -f $ENV_TAR $CODE_TAR
set +x
source $WORKDIR/env/bin/activate || setup_failed "activating the environment"
conda-unpack || setup_failed "conda-unpack"
set -x
for sp in $WORKDIR/env/lib/python3.*/site-packages; do
    [ -d "$sp" ] || continue
    rm -f $sp/__editable__*BTVNanoCommissioning* $sp/__editable__*btvnanocommissioning* \
          $sp/BTVNanoCommissioning.egg-link
    if [ -f $sp/easy-install.pth ]; then sed -i '/BTVNanoCommissioning/d' $sp/easy-install.pth; fi
done
export PYTHONPATH=$WORKDIR/code/src
export PYTHONDONTWRITEBYTECODE=1
export X509_USER_PROXY=${X509_USER_PROXY:-$WORKDIR/proxy}

# input: the file of this histogram in each input directory (split_coffea.py: one directory;
# mergeoutputs.py --split-hists: one per sample list)
mapfile -t URLS < $WORKDIR/$INDIRS
INPUTS=()
for i in "${!URLS[@]}"; do
    u=${URLS[$i]}
    [ -n "$u" ] || continue
    host=${u#root://}; host=${host%%/*}; path=/${u#root://*/}
    xrdfs root://$host stat "$path/$VAR.coffea" >/dev/null 2>&1 || continue
    xrdcp -f "$u/$VAR.coffea" $WORKDIR/in_${i}_$VAR.coffea || setup_failed "xrdcp $u/$VAR.coffea"
    INPUTS+=($WORKDIR/in_${i}_$VAR.coffea)
done
[ "${#INPUTS[@]}" -gt 0 ] || setup_failed "no input file $VAR.coffea"
INLIST=$(IFS=,; echo "${INPUTS[*]}")
mapfile -t OPTS < $WORKDIR/plot_opts.txt

# the framework opens some files relative to the repository directory
cd $WORKDIR/code
mkdir -p $WORKDIR/plots
python $WORKDIR/plotdataMC.py -p "$PHASE" -i "$INLIST" -v "$VAR" \
    -o $WORKDIR/plots "${OPTS[@]}"
rm -f "${INPUTS[@]}"

# output: the plots of this histogram
n=0
for f in $WORKDIR/plots/*; do
    [ -f "$f" ] || continue
    ok=0
    for attempt in 1 2 3; do
        xrdcp -f "$f" "$OUTURL/$(basename $f)" && { ok=1; break; }
        sleep 10
    done
    [ $ok -eq 1 ] || { echo "ERROR: xrdcp $f failed" >&2; exit 1; }
    n=$((n + 1))
done
rm -rf $WORKDIR/plots
echo "$n files copied to $OUTURL"
