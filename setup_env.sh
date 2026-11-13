#!/bin/bash
# Activates the BTVNanoCommissioning conda environment created from test_env.yml
# (coffea 2026.7.0 / awkward 2) and sets up the grid, CMS and condor environment for lxplus.
#
# Source it, do not execute it:  source setup_env.sh
#
# Overridable, all with working defaults:
#   CONDA_BASE                    conda install root; otherwise taken from CONDA_EXE, then
#                                 `conda` on PATH, then the usual locations
#   BTV_ENV                       conda environment name or prefix   (default: btv_coffea)
#   X509_USER_PROXY               proxy file, readable by condor jobs (default: $HOME/x509up_u<uid>)
#   BTV_CONDOR_JOBDIR             parent of the condor job directories jobs_<jobName>
#                                 (default: /afs/cern.ch/work/<u>/<user>/btv_jobs if AFS work exists)
#   BTV_CONDOR_ACCOUNTING_GROUP   condor accounting group             (default: group_u_CMST3.all)
#
# One-time setup of the environment (README, "Standalone conda environment"): on lxplus remove
# the conda voms package (`conda remove --force voms`), whose voms-proxy-init hangs on an
# outdated VOMS server list; this script falls back to the system voms tools if it is present.

# Everything is derived from where this script lives, so the repo can sit anywhere.
REPO_ROOT=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi

# personal conda init function, if defined in the login scripts
if declare -F initConda >/dev/null; then
    initConda
fi

_conda_base="${CONDA_BASE:-}"
if [ -z "${_conda_base}" ] && [ -n "${CONDA_EXE:-}" ]; then
    _conda_base=$("${CONDA_EXE}" info --base 2>/dev/null)
fi
if [ -z "${_conda_base}" ] && command -v conda >/dev/null 2>&1; then
    _conda_base=$(conda info --base 2>/dev/null)
fi
if [ -z "${_conda_base}" ]; then
    for _cand in "/afs/cern.ch/work/${USER:0:1}/${USER}/miniforge3" \
                 "/afs/cern.ch/work/${USER:0:1}/${USER}/private/miniforge3" \
                 "${HOME}/miniforge3" "${HOME}/miniconda3"; do
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
if ! conda activate "${BTV_ENV:-btv_coffea}"; then
    echo "ERROR: could not activate '${BTV_ENV:-btv_coffea}'. Set BTV_ENV to the environment name or prefix." >&2
    unset _conda_base
    return 1 2>/dev/null || exit 1
fi
unset _conda_base

# Grid certificates: ca-policy-lcg (test_env.yml) points X509_CERT_DIR to its own bundled
# CRLs, which expire after about a week ("CRL has expired"). The CVMFS directory is refreshed
# automatically; use it when present.
if [ -d /cvmfs/grid.cern.ch/etc/grid-security/certificates ]; then
    export X509_CERT_DIR=/cvmfs/grid.cern.ch/etc/grid-security/certificates
fi
# The conda voms package's voms-proxy-init uses an outdated VOMS server list and hangs;
# use the system tools while it is still installed.
if [ -x "${CONDA_PREFIX}/bin/voms-proxy-init" ] && [ -x /usr/bin/voms-proxy-init ]; then
    alias voms-proxy-init=/usr/bin/voms-proxy-init
    alias voms-proxy-info=/usr/bin/voms-proxy-info
    echo "NOTE: conda voms package found in ${CONDA_PREFIX}; using /usr/bin/voms-proxy-*" \
         "(remove it with: conda remove --force voms)"
fi
# proxy on AFS home, where the condor submission can pick it up
export X509_USER_PROXY="${X509_USER_PROXY:-$HOME/x509up_u$(id -u)}"

# CMS tools (dasgoclient) and gfal
export PATH="/cvmfs/cms.cern.ch/common:${PATH}"
export GFAL_PYTHONBIN="${GFAL_PYTHONBIN:-/usr/bin/python3}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# condor_lxplus/submitter.py defaults
if [ -z "${BTV_CONDOR_JOBDIR:-}" ] && [ -d "/afs/cern.ch/work/${USER:0:1}/${USER}" ]; then
    export BTV_CONDOR_JOBDIR="/afs/cern.ch/work/${USER:0:1}/${USER}/btv_jobs"
fi
export BTV_CONDOR_ACCOUNTING_GROUP="${BTV_CONDOR_ACCOUNTING_GROUP:-group_u_CMST3.all}"

alias python=python3

# checks: editable install of this clone, code and environment not on EOS
_pkg=$(python -c "import importlib.util as u; s = u.find_spec('BTVNanoCommissioning'); print(s.origin if s else '')" 2>/dev/null)
if [ -z "${_pkg}" ]; then
    echo "WARNING: BTVNanoCommissioning is not installed in this environment: pip install -e ${REPO_ROOT}"
elif [[ "${_pkg}" != "${REPO_ROOT}/src/"* ]]; then
    echo "WARNING: BTVNanoCommissioning is imported from ${_pkg}, not from this clone: pip install -e ${REPO_ROOT}"
fi
unset _pkg
for _p in "${REPO_ROOT}" "${CONDA_PREFIX}"; do
    if [[ "${_p}" == /eos/* ]]; then
        echo "WARNING: ${_p} is on EOS; keep code and environments on AFS (/afs/cern.ch/work)"
    fi
done
unset _p

echo "Activated ${CONDA_DEFAULT_ENV} (numpy $(python -c 'import numpy; print(numpy.__version__)') / coffea $(python -c 'import coffea; print(coffea.__version__)'))"
echo "  REPO_ROOT=${REPO_ROOT}"
echo "  X509_USER_PROXY=${X509_USER_PROXY}  X509_CERT_DIR=${X509_CERT_DIR}"
echo "  BTV_CONDOR_JOBDIR=${BTV_CONDOR_JOBDIR:-<current directory>}  BTV_CONDOR_ACCOUNTING_GROUP=${BTV_CONDOR_ACCOUNTING_GROUP}"
