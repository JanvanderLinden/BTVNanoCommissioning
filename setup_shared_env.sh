#!/bin/bash
# Activates commfw2 -- the single env that can run both this repo (BTVNanoCommissioning,
# coffea 2026.7.0/awkward2) and rabbit (tensorflow-based fitting), now that commfw2 has
# been bumped to numpy==2.2.6 to match rabbit_env and had rabbit's runtime deps
# (tensorflow, tensorflow-probability, wums, h5py) installed into it.
#
# For BTVNanoCommissioning: just activating commfw2 is enough (this repo is pip-installed
# editable into commfw2 already).
# For rabbit: mirrors rabbit/setup.sh (RABBIT_BASE, PYTHONPATH, PATH, XLA_FLAGS), pointed
# at commfw2 instead of rabbit_env.
#
# Source it, do not execute it:  source setup_shared_env.sh
#
# Overridable, all with working defaults:
#   CONDA_BASE        conda install root; otherwise taken from CONDA_EXE, then `conda`
#                     on PATH, then the usual locations
#   COMMFW_ENV        conda env name                       (default: commfw2)
#   ISARRAY           write per-jet ROOT arrays            (default: 1)
#   ARRAY_SYSTS       systematic content of those arrays   (default: weights)
#   ACCOUNTING_GROUP  condor accounting group              (default: group_u_CMST3.all)

# Everything is derived from where this script lives, so the repo can sit anywhere
# and belong to anyone without editing paths in here.
REPO_ROOT=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi

initConda

# commfw2 is a real conda env (unlike rabbit_env, which is a plain venv with its
# own bin/activate) -- needs the conda hook, not `source .../bin/activate`.
_conda_base="${CONDA_BASE:-}"
if [ -z "${_conda_base}" ] && [ -n "${CONDA_EXE:-}" ]; then
    _conda_base=$("${CONDA_EXE}" info --base 2>/dev/null)
fi
if [ -z "${_conda_base}" ] && command -v conda >/dev/null 2>&1; then
    _conda_base=$(conda info --base 2>/dev/null)
fi
if [ -z "${_conda_base}" ]; then
    for _cand in "/eos/user/${USER:0:1}/${USER}/MyConda" \
                 "/eos/home-${USER:0:1}/${USER}/MyConda" \
                 "${HOME}/MyConda" "${HOME}/miniforge3" "${HOME}/miniconda3"; do
        if [ -x "${_cand}/condabin/conda" ]; then
            _conda_base="${_cand}"
            break
        fi
    done
fi
if [ -z "${_conda_base}" ] || [ ! -x "${_conda_base}/condabin/conda" ]; then
    echo "ERROR: no conda installation found. Set CONDA_BASE to its root." >&2
    return 1 2>/dev/null || exit 1
fi
unset _cand

eval "$("${_conda_base}/condabin/conda" shell.bash hook)"
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    conda deactivate
fi

export RABBIT_BASE="${REPO_ROOT}/rabbit"
export PATH="$PATH:${RABBIT_BASE}/bin"

# Enable XLA's multi-threaded Eigen path on CPU. ~1.3x speedup on dense
# large-model HVP/loss+grad on many-core systems, no downside on smaller
# problems. Append to any existing XLA_FLAGS so user-set flags survive.
if [[ ":${XLA_FLAGS:-}:" != *":--xla_cpu_multi_thread_eigen=true:"* ]]; then
    export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_cpu_multi_thread_eigen=true"
fi

conda activate "${COMMFW_ENV:-commfw2}"

# One assignment covering rabbit and this repo. Previously RABBIT_BASE was put on
# PYTHONPATH and then overwritten further down, which left rabbit unimportable.
export PYTHONPATH="${RABBIT_BASE}:${REPO_ROOT}:${REPO_ROOT}/SFbc-2D/scripts/utilities${PYTHONPATH:+:${PYTHONPATH}}"

export ISARRAY="${ISARRAY:-1}"
export ARRAY_SYSTS="${ARRAY_SYSTS:-weights}"
export ACCOUNTING_GROUP="${ACCOUNTING_GROUP:-group_u_CMST3.all}"
# export X509_USER_PROXY="${X509_USER_PROXY:-$HOME/x509up_u$(id -u)}"
export X509_USER_PROXY=$HOME/x509up_u`id -u`
export GFAL_PYTHONBIN="${GFAL_PYTHONBIN:-/usr/bin/python3}"

# commfw2's ca-policy-lcg pins X509_CERT_DIR to its own bundled CRLs, which go stale
# (they expire ~weekly) and break GSI auth with "CRL has expired". The CVMFS CA dir is
# refreshed automatically -- prefer it when present, keep the conda default otherwise.
if [ -d /cvmfs/grid.cern.ch/etc/grid-security/certificates ]; then
    export X509_CERT_DIR=/cvmfs/grid.cern.ch/etc/grid-security/certificates
fi

alias python=python3

echo "Activated ${CONDA_DEFAULT_ENV} (numpy $(python -c 'import numpy; print(numpy.__version__)') / coffea $(python -c 'import coffea; print(coffea.__version__)'))"
echo "  REPO_ROOT=${REPO_ROOT}  RABBIT_BASE=${RABBIT_BASE}"
echo "  ISARRAY=${ISARRAY}  ARRAY_SYSTS=${ARRAY_SYSTS}  ACCOUNTING_GROUP=${ACCOUNTING_GROUP}"
echo "  X509_USER_PROXY=${X509_USER_PROXY}  GFAL_PYTHONBIN=${GFAL_PYTHONBIN}"
echo "  X509_CERT_DIR=${X509_CERT_DIR}"
