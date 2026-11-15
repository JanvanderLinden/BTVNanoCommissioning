#!/bin/bash -xe

JOBID=$1
COMMDIR=$2
OUTPUTDIR=$3
ENVPATH=$4
NTHREADS=$5

WORKDIR=`pwd`
# everything the job writes (caches, config) stays in its scratch directory
export HOME=$WORKDIR
export XDG_CACHE_HOME=$WORKDIR/.cache
export MPLCONFIGDIR=$WORKDIR/.cache/matplotlib
export NUMBA_CACHE_DIR=$WORKDIR/.cache/numba
mkdir -p $XDG_CACHE_HOME $MPLCONFIGDIR $NUMBA_CACHE_DIR

ENV_TAR=$(ls $WORKDIR/btv_env_*.tar.gz 2>/dev/null | head -1)
CODE_TAR=$(ls $WORKDIR/btv_code_*.tar.gz 2>/dev/null | head -1)
if [ -n "$ENV_TAR" ] && [ -n "$CODE_TAR" ]; then
    # Sandbox (submitter.py default): conda environment and code from the tarballs that
    # condor transferred from EOS, unpacked here -- nothing is read from AFS
    echo "Unpacking $(basename $ENV_TAR) and $(basename $CODE_TAR)"
    # a failure here fails every job in the same way: exit code 99 puts the job on hold
    # (on_exit_hold in submit.jdl) instead of requeueing it
    setup_failed() { echo "ERROR: sandbox setup failed: $1" >&2; exit 99; }
    mkdir -p $WORKDIR/env $WORKDIR/code
    tar -xzf $ENV_TAR -C $WORKDIR/env || setup_failed "unpacking $(basename $ENV_TAR)"
    rm -f $ENV_TAR
    tar -xzf $CODE_TAR -C $WORKDIR/code || setup_failed "unpacking $(basename $CODE_TAR)"
    rm -f $CODE_TAR
    set +x
    source $WORKDIR/env/bin/activate || setup_failed "activating the environment"
    # conda-unpack rewrites the packing prefix in the files of the environment, including
    # those of the editable install: run it before those are removed
    conda-unpack || setup_failed "conda-unpack"
    set -x
    # the editable install of BTVNanoCommissioning points to the submitting checkout: drop
    # it, the code is taken from code/src
    for sp in $WORKDIR/env/lib/python3.*/site-packages; do
        [ -d "$sp" ] || continue
        rm -f $sp/__editable__*BTVNanoCommissioning* $sp/__editable__*btvnanocommissioning* \
              $sp/BTVNanoCommissioning.egg-link
        if [ -f $sp/easy-install.pth ]; then sed -i '/BTVNanoCommissioning/d' $sp/easy-install.pth; fi
    done
    export PYTHONPATH=$WORKDIR/code/src
    export PYTHONDONTWRITEBYTECODE=1
    cd $WORKDIR/code
else
    # --noSandbox: checkout and environment on the shared filesystem (AFS); only for a few
    # jobs
    echo "WARNING: no sandbox tarballs, running from $COMMDIR with $ENVPATH"
    cd $COMMDIR
    export PATH="$ENVPATH:$PATH"
fi
env

# proxy shipped with the job (x509userproxy in submit.jdl; condor sets X509_USER_PROXY)
export X509_USER_PROXY=${X509_USER_PROXY:-$COMMDIR/proxy}
voms-proxy-info
python -c "import BTVNanoCommissioning, coffea; print('BTVNanoCommissioning from', BTVNanoCommissioning.__file__, '/ coffea', coffea.__version__)" \
    || { echo "ERROR: BTVNanoCommissioning/coffea not importable" >&2; exit 99; }

# Build the sample json given the job id
python -c "import json, os; flname = 'split_samples.json' if os.path.isfile(f'$WORKDIR/split_samples.json') else 'split_samples_resubmit.json';  json.dump(json.load(open(f'$WORKDIR/{flname}'))['$JOBID'], open('$WORKDIR/sample.json', 'w'), indent=4)"

declare -A ARGS
for key in workflow output samplejson year campaign isSyst isArray array_systs ttbar_reweights vjets_stitch nlo_reweight noHist overwrite voms chunk skipbadfiles outputDir remoteRepo selectionModifier; do
    ARGS[$key]=$(jq -r ".$key" $WORKDIR/arguments.json)
done

# Unparse arguments and send to runner.py
OPTS="--wf ${ARGS[workflow]} --year ${ARGS[year]} --campaign ${ARGS[campaign]} --chunk ${ARGS[chunk]} "
if [ "${ARGS[voms]}" != "null" ]; then
    OPTS="$OPTS --voms ${ARGS[voms]}"
fi
if [ "${ARGS[selectionModifier]}" != "null" ]; then
    OPTS="$OPTS --selectionModifier ${ARGS[selectionModifier]}"
fi
if [ "${ARGS[isSyst]}" != "false" ]; then
    OPTS="$OPTS --isSyst ${ARGS[isSyst]}"
fi
if [ "${ARGS[array_systs]}" != "none" ]; then
    OPTS="$OPTS --array-systs ${ARGS[array_systs]}"
fi
if [ "${ARGS[nlo_reweight]}" == true ]; then
    OPTS="$OPTS --nlo-reweight"
fi
if [ "${ARGS[vjets_stitch]}" != "null" ]; then
    OPTS="$OPTS --vjets-stitch ${ARGS[vjets_stitch]}"
fi
if [ "${ARGS[ttbar_reweights]}" != "none" ]; then
    OPTS="$OPTS --ttbar-reweights ${ARGS[ttbar_reweights]}"
fi

for key in  isArray noHist overwrite skipbadfiles; do
    if [ "${ARGS[$key]}" == true ]; then
        OPTS="$OPTS --$key"
    fi
done
OPTS="$OPTS --output ${ARGS[output]//.coffea/_$JOBID.coffea}"  # add a suffix to output file name
OPTS="$OPTS --json $WORKDIR/sample.json"  # use the sample json for this JOBID
OPTS="$OPTS --worker $NTHREADS"  # use number of worker = 1

if [ $NTHREADS -gt 1 ]; then
    OPTS="$OPTS --executor futures"
else
    OPTS="$OPTS --executor iterative"
fi

# outputs are written to the job's local directory and copied to OUTPUTDIR at the end
LOCALOUT=$WORKDIR/output
mkdir -p $LOCALOUT
OPTS="$OPTS --overwrite --outputdir $LOCALOUT"

# Launch
echo "Now launching: python runner.py $OPTS"
python runner.py $OPTS

# Stage-out: one copy per output file to OUTPUTDIR, xrdcp (3 attempts) for /eos paths and
# root:// URLs, cp otherwise
xrd_url() {
    local p="$1"
    case "$p" in
        root://*) echo "$p"; return ;;
        /eos/home-*) p="/eos/user/${p:10:1}/${p:12}" ;;
    esac
    case "$p" in
        /eos/user/*) echo "root://eosuser.cern.ch/$p" ;;
        /eos/cms/*) echo "root://eoscms.cern.ch/$p" ;;
        /eos/project/*) echo "root://eosproject.cern.ch/$p" ;;
    esac
}
DESTURL=$(xrd_url "$OUTPUTDIR")
NFILES=0
cd $LOCALOUT
while IFS= read -r -d '' f; do
    rel=${f#./}
    if [ -n "$DESTURL" ]; then
        ok=0
        for i in 1 2 3; do
            if xrdcp -f -p -N "$f" "$DESTURL/$rel"; then ok=1; break; fi
            sleep $((30 * i))
        done
        if [ $ok -ne 1 ]; then
            echo "ERROR: stage-out of $rel to $DESTURL failed" >&2
            exit 1
        fi
    else
        mkdir -p "$OUTPUTDIR/$(dirname "$rel")"
        cp -f "$f" "$OUTPUTDIR/$rel"
    fi
    NFILES=$((NFILES + 1))
done < <(find . -type f -print0)
echo "Staged out $NFILES file(s) to ${DESTURL:-$OUTPUTDIR}"

exit 0
