#!/bin/bash

JOBID=$1
NCPU=$2

# Phase timing markers -- grep 'PHASE' out of the condor .out to attribute
# wall time to setup / extract / processing / stage-out.
PHASE_T0=$(date +%s)
PHASE_TPREV=$PHASE_T0
phase() {
    local now=$(date +%s)
    echo "PHASE $1 elapsed=$((now - PHASE_TPREV))s total=$((now - PHASE_T0))s"
    PHASE_TPREV=$now
}

export HOME=`pwd`
if [ -d /afs/cern.ch/user/${USER:0:1}/$USER ]; then
  export HOME=/afs/cern.ch/user/${USER:0:1}/$USER  # crucial on lxplus condor but cannot set on cmsconnect
fi
env


WORKDIR=`pwd`

# Get arguments
declare -A ARGS
for key in workflow output samplejson year campaign isSyst ttbar_reweights vjets_stitch nlo_reweight local_copy isArray array_systs tagger_compare noHist overwrite voms chunk skipbadfiles outputDir remoteRepo; do
    echo $(jq -r ".$key" $WORKDIR/arguments.json)
    ARGS[$key]=$(jq -r ".$key" $WORKDIR/arguments.json)
done

# Set up mamba environment
## Interactive bash script with fallback pointing to $HOME, hence setting $PWD of worker node as $HOME
export HOME=`pwd`

echo "Setting up mamba environment"
if [[ ${ARGS[remoteRepo]} != "" ]]; then
    echo "remoteRepo is set to ${ARGS[remoteRepo]}"
    for i in {1..10}; do
        wget --timeout=20 --tries=1 -L micro.mamba.pm/install.sh
        if [ $? -eq 0 ]; then
            break
        fi
	echo "Failed attempt #"$i" to download mamba installer. Will retry."
        sleep 30
    done
    chmod +x install.sh
    ## FIXME parsing arguments does not work. will use defaults in install.sh instead, see https://github.com/mamba-org/micromamba-releases/blob/main/install.sh 
    ## Tried solutions listed in https://stackoverflow.com/questions/14392525/passing-arguments-to-an-interactive-program-non-interactively
    ./install.sh <<< $'bin\nY\nY\nmicromamba\n' 
    source .bashrc
fi

export PATH=$WORKDIR:$PATH

if [ ! -d /afs/cern.ch/user/${USER:0:1}/$USER ]; then
    ## install necessary packages if on cmsconnect
    micromamba install -c conda-forge jq --yes || { echo "ERROR: micromamba install of jq failed."; exit 1; }
fi

# Create base env with python=3.10 and setuptools<=70.1.1
micromamba activate
micromamba install python=3.10 -c conda-forge xrootd --yes || { echo "ERROR: micromamba install of python/xrootd failed."; exit 1; }
micromamba activate base
micromamba install setuptools=70.1.1 || { echo "ERROR: micromamba install of setuptools failed."; exit 1; }
phase mamba

# Install BTVNanoCommissioning
mkdir BTVNanoCommissioning
cd BTVNanoCommissioning
if [ ! -f $WORKDIR/BTVNanoCommissioning.tar.gz ]; then
    ## clone the BTVNanoCommissioning repo only, no submodule
    git clone ${ARGS[remoteRepo]} . || { echo "ERROR: git clone of ${ARGS[remoteRepo]} failed."; exit 1; }
else
    tar xaf $WORKDIR/BTVNanoCommissioning.tar.gz || { echo "ERROR: extracting BTVNanoCommissioning.tar.gz failed."; exit 1; }
fi
phase extract

pip install -e . || { echo "ERROR: pip install -e . failed."; exit 1; }

## other dependencies
pip install psutil || { echo "ERROR: pip install psutil failed."; exit 1; }
phase pip

SPLIT=$WORKDIR/split_samples.json
[ -f $SPLIT ] || SPLIT=$WORKDIR/split_samples_resubmit.json

# run_one <workflow> <year> <campaign> <outputDir> <sample json>: runner.py + stage-out
run_one() {
local WF=$1 YEAR=$2 CAMPAIGN=$3 OUTDIR=$4
local DONE=$OUTDIR/hists_$JOBID/hists_$JOBID.coffea
if [[ $OUTDIR == root://* ]]; then
    host=${OUTDIR#root://}; xrdfs root://${host%%/*} stat /${host#*/}/hists_$JOBID/hists_$JOBID.coffea >/dev/null 2>&1 && DONE_OK=1 || DONE_OK=0
else
    [ -f $DONE ] && DONE_OK=1 || DONE_OK=0
fi
if [ $DONE_OK == 1 ] && [ "${MULTI:-0}" == 1 ]; then
    echo "Output of $WF $CAMPAIGN exists (earlier attempt) -- skipping"
    return 0
fi
cp $5 $WORKDIR/BTVNanoCommissioning/sample.json

# Unparse arguments and send to runner.py
OPTS="--wf $WF --year $YEAR --campaign $CAMPAIGN --chunk ${ARGS[chunk]}"
if [ "${ARGS[voms]}" != "null" ]; then
    OPTS="$OPTS --voms ${ARGS[voms]}"
fi
if [ "${ARGS[isSyst]}" != "false" ]; then
    OPTS="$OPTS --isSyst ${ARGS[isSyst]}"
fi
if [ "${ARGS[ttbar_reweights]}" != "none" ]; then
    OPTS="$OPTS --ttbar-reweights ${ARGS[ttbar_reweights]}"
fi
if [ "${ARGS[array_systs]}" != "none" ] && [ "${ARGS[array_systs]}" != "null" ]; then
    OPTS="$OPTS --array-systs ${ARGS[array_systs]}"
fi
if [ "${ARGS[local_copy]}" == true ]; then
    OPTS="$OPTS --local-copy"
fi
if [ "${ARGS[nlo_reweight]}" == true ]; then
    OPTS="$OPTS --nlo-reweight"
fi
if [ "${ARGS[vjets_stitch]}" != "null" ]; then
    OPTS="$OPTS --vjets-stitch ${ARGS[vjets_stitch]}"
fi
if [ "${ARGS[tagger_compare]}" == true ]; then
    OPTS="$OPTS --tagger-compare"
fi
for key in  isArray noHist overwrite skipbadfiles; do
    if [ "${ARGS[$key]}" == true ]; then
        OPTS="$OPTS --$key"
    fi
done
OPTS="$OPTS --output ${ARGS[output]//.coffea/_$JOBID.coffea}"  # add a suffix to output file name
OPTS="$OPTS --json sample.json"  # use the sample json for this JOBID

# Check the number of CPUs requested and set the worker accordingly.
# If nCPU > 1, use futures executor with nCPU workers. If nCPU = 1, use iterative executor with 1 worker.
if [ $NCPU -gt 1 ]; then
    OPTS="$OPTS --worker $NCPU"  # use number of worker = nCPU
    OPTS="$OPTS --executor futures"
else
    OPTS="$OPTS --worker 1"  # use number of worker = 1
    OPTS="$OPTS --executor iterative"
fi

# Launch
echo "Now launching: python runner.py $OPTS"
python runner.py $OPTS
RUNNER_RC=$?
phase "runner $WF $CAMPAIGN"
if [ $RUNNER_RC -ne 0 ]; then
    echo "ERROR: runner.py failed -- not transferring output or marking success."
    return 1
fi

# Transfer output: arrays first, the .coffea last (it marks the run as done)
if [[ $OUTDIR == root://* ]]; then

    if [[ "$OPTS" == *"isArray"* ]]; then
	xrdcp --silent -p -f -r arrays_* $OUTDIR/
	if [ $? -ne 0 ]; then
	    echo "ERROR: xrdcp of arrays_* to $OUTDIR failed."
	    return 1
	fi
    fi
    xrdcp --silent -p -f -r hists_* $OUTDIR/
    if [ $? -ne 0 ]; then
        echo "ERROR: xrdcp of hists_* to $OUTDIR failed."
        return 1
    fi
else
    mkdir -p $OUTDIR
    if [[ "$OPTS" == *"isArray"* ]]; then
	cp -p -f -r arrays_* $OUTDIR/
	if [ $? -ne 0 ]; then
	    echo "ERROR: cp of arrays_* to $OUTDIR failed."
	    return 1
	fi
    fi
    cp -p -f -r hists_* $OUTDIR/
    if [ $? -ne 0 ]; then
        echo "ERROR: cp of hists_* to $OUTDIR failed."
        return 1
    fi
fi
phase stageout

rm -rf hists_* arrays_*
}

if [ "$(jq -r '.configs' $WORKDIR/arguments.json)" == "null" ]; then
    python -c "import json; json.dump(json.load(open('$SPLIT'))['$JOBID'], open('$WORKDIR/sample.json', 'w'), indent=4)"
    run_one ${ARGS[workflow]} ${ARGS[year]} ${ARGS[campaign]} ${ARGS[outputDir]} $WORKDIR/sample.json || exit 1
else
    export BTV_LOCALCOPY_KEEP=1  # the configs of a job share the local input copies
    MULTI=1  # a failed run does not stop the others; a retry skips the runs with output
    FAILED=""
    # job = {dataset: {file: [config, ...]}} -> sample_<config>.json, prints the configs
    CFGS=$(python -c "
import json
job = json.load(open('$SPLIT'))['$JOBID']
cfgs = sorted({c for d in job.values() for cs in d.values() for c in cs}, key=int)
for c in cfgs:
    s = {ds: [f for f, cs in d.items() if c in cs] for ds, d in job.items()}
    json.dump({ds: fl for ds, fl in s.items() if fl}, open(f'$WORKDIR/sample_{c}.json', 'w'))
print(*cfgs)") || { echo "ERROR: cannot read job $JOBID from $SPLIT"; exit 1; }
    for i in $CFGS; do
        C=".configs[$i]"
        run_one $(jq -r "$C.workflow, $C.year, $C.campaign, $C.outputDir" $WORKDIR/arguments.json) $WORKDIR/sample_$i.json || { FAILED="$FAILED $i"; rm -rf hists_* arrays_*; }
    done
    rm -rf localcopy
    if [ -n "$FAILED" ]; then
        echo "ERROR: configs$FAILED failed -- exit 1 so that condor retries them"
        exit 1
    fi
fi

### one can also consider origanizing the root files in the subdirectory structure ###
# for filename in `\ls *.root`; do
#     SAMPLENAME=$(echo "$filename" | sed -E 's/(.*)_f[0-9-]+_[0-9]+\.root/\1/')
#     # SAMPLENAME=$(echo "$filename" | sed -E 's/(.*)_[0-9a-z]{9}-[0-9a-z]{4}-.*\.root/\1/')
#     xrdcp --silent -p -f $filename ${ARGS[outputDir]}/$SAMPLENAME/
# done

touch $WORKDIR/.success
