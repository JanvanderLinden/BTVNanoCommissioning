#!/bin/bash -xe

JOBID=$1
COMMDIR=$2
OUTPUTDIR=$3
ENVPATH=$4
NTHREADS=$5

export HOME=`pwd`
if [ -d /afs/cern.ch/user/${USER:0:1}/$USER ]; then
  export HOME=/afs/cern.ch/user/${USER:0:1}/$USER  # crucial on lxplus condor but cannot set on cmsconnect
fi
env

WORKDIR=`pwd`

cd $COMMDIR
# proxy shipped with the job (x509userproxy in submit.jdl; condor sets X509_USER_PROXY)
export X509_USER_PROXY=${X509_USER_PROXY:-$COMMDIR/proxy}
voms-proxy-info

export PATH="$ENVPATH:$PATH" 

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
