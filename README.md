
# BTVNanoCommissioning
[![Linting](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/python_linting.yml/badge.svg)](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/python_linting.yml)
[![btag ttbar](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/ttbar_workflow.yml/badge.svg)](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/ttbar_workflow.yml)
[![ctag ttbar](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/ctag_ttbar_workflow.yml/badge.svg)](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/ctag_ttbar_workflow.yml)
[![ctag DY+jets Workflow](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/ctag_DY_workflow.yml/badge.svg)](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/ctag_DY_workflow.yml)
[![ctag W+c Workflow](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/ctag_Wc_workflow.yml/badge.svg)](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/ctag_Wc_workflow.yml)
[![BTA Workflow](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/BTA_workflow.yml/badge.svg)](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/BTA_workflow.yml)
[![QCD Workflow](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/QCD_workflow.yml/badge.svg)](https://github.com/cms-btv-pog/BTVNanoCommissioning/actions/workflows/QCD_workflow.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Repository for Commissioning studies in the BTV POG based on (custom) nanoAOD samples


This framework is based on [coffea](https://coffeateam.github.io/coffea/) and using [btvnano](https://btv-wiki.docs.cern.ch/SoftwareAlgorithms/PFNano/) as input. The framework is also used as frontend for the btv automation task [autobtv](https://gitlab.cern.ch/cms-analysis/btv/software-and-algorithms/autobtv)

This framework is based on [coffea processor](https://coffeateam.github.io/coffea/concepts.html#coffea-processor). Each workflow can be a separate **processor** file in the `workflows`, creating the mapping from `PFNano` to the histograms as `coffea` file or creating `.root` files by saving awkward arrays. Workflow processors can be passed to the `runner.py` script along with the fileset these should run over. Multiple executors can be chosen
(`iterative` - one by one, `futures` - multiprocessing). Scale out to clusters depend on facilities. Obtain the histograms as plot(`.pdf`) or save to template `.root` file with dedicated scripts

The minimum requirement commands are shown in follow, specified the selections, datataset, campaign and year
```
python runner.py --workflow example --json metadata/test_bta_run3.json --campaign Summer23 --year 2023
```
- Detailed documentation [here](https://btvnanocommissioning.readthedocs.io/en/latest/)
- To running the commissioning task or producing the template: go to [Preparation for commissioning/SFs tasks](https://btvnanocommissioning.readthedocs.io/en/latest/user.html)
- To develop new workflow, the instruction can be found in [Instruction for developers](https://btvnanocommissioning.readthedocs.io/en/latest/user.html)
- Current working in progress [issues](https://gitlab.cern.ch/cms-btv-coordination/tasks/-/issues/?label_name%5B%5D=Software%3A%3A%20BTVnano%20%26CommFW)



## Setup 

You can install your [standalone conda envrionment](#standalone-conda-environment) via `yaml` or on the lxplus you can directly jump to [setup](#setup-the-framework)
### Standalone conda environment
> [!Caution]
> suggested to install under `bash` environment


Install Miniforge (provides `conda` and `mamba`; for a standalone `micromamba` see [here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)).
The environment needs a few GB: on lxplus choose an install path with enough space (`-p <path>`).
```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
# Run and follow instructions on screen
bash Miniforge3-$(uname)-$(uname -m).sh

# in every new shell (or once: <install path>/bin/conda init bash)
source <install path>/etc/profile.d/conda.sh
```
NOTE: always make sure that conda, python, and pip point to your local installation (`which conda` etc.).


Create the environment from `test_env.yml` (coffea is installed with it) and activate it
```
mamba env create -f test_env.yml   # environment btv_coffea: python 3.10, coffea 2026.7.0, awkward 2
conda activate btv_coffea
```
On lxplus, put the Miniforge installation and the environment on AFS work
(`/afs/cern.ch/work/<u>/<user>`, `mamba env create -f test_env.yml -p <path>/envs/btv_coffea`),
not on EOS: tens of thousands of small files make imports slow and unreliable there. Keep the
conda package caches out of the AFS home (10 GB quota), e.g. `conda config --add pkgs_dirs
/afs/cern.ch/work/<u>/<user>/conda_pkgs` and `export PIP_CACHE_DIR=/afs/cern.ch/work/<u>/<user>/pip_cache`.

One-time fixes for the grid proxy on lxplus:
```bash
conda activate btv_coffea
# the conda voms package uses an outdated VOMS server list (voms-proxy-init hangs):
# remove only this package, without re-solving the environment
conda remove --force voms
hash -r; which voms-proxy-init        # -> /usr/bin/voms-proxy-init

# ca-policy-lcg sets X509_CERT_DIR to bundled CRLs that expire after about a week
# ("CRL has expired"): point it to the CVMFS copy, after the package's own activation script
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/zz_x509_cert_dir.sh <<'EOF'
if [ -d /cvmfs/grid.cern.ch/etc/grid-security/certificates ]; then
    export X509_CERT_DIR=/cvmfs/grid.cern.ch/etc/grid-security/certificates
fi
EOF

# keep later installs into this environment on conda-forge only
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
conda deactivate && conda activate btv_coffea
```
Use `conda remove --force voms`: a plain `conda remove voms` re-solves the whole environment
and can replace the conda-forge packages with builds from other channels. `setup_env.sh`
(below) applies both fixes for its shell as well.

### Setup the framework

```bash
# activate enviroment once you have coffea framework 
conda activate btv_coffea

# only first time, including submodules
git clone --recurse-submodules git@github.com:cms-btv-pog/BTVNanoCommissioning.git 
# (existing clone: git submodule update --init <submodule>, e.g. sfs-3d)
# Once the environment is set up, compile the python package:
pip install -e .
pip install -e .[dev,docs] # for developer
```

In every new shell on lxplus:
```bash
source setup_env.sh                   # BTV_ENV=<name or prefix> if the environment is not btv_coffea
voms-proxy-init --rfc --voms cms -valid 192:00
```
`setup_env.sh` activates the environment and sets
- `X509_CERT_DIR` to the CVMFS certificates, and the system `voms-proxy-*` if the conda `voms`
  package is still installed;
- `X509_USER_PROXY=$HOME/x509up_u<uid>` (unless set);
- `/cvmfs/cms.cern.ch/common` on the `PATH` (`dasgoclient`) and `GFAL_PYTHONBIN`;
- the defaults of `condor_lxplus/submitter.py`: `BTV_CONDOR_JOBDIR`
  (`/afs/cern.ch/work/<u>/<user>/btv_jobs`) and `BTV_CONDOR_ACCOUNTING_GROUP`
  (`group_u_CMST3.all`).

It warns if `BTVNanoCommissioning` is not installed from this clone (e.g. after moving it:
rerun `pip install -e .`) or if the clone or the environment is on EOS. `CONDA_BASE` sets the
conda installation if it is not found automatically.

The central environment `/eos/home-m/milee/miniforge3/envs/btv_coffea` contains coffea 0.7
and does not run the current version (coffea 2026 / awkward 2); create your own from
`test_env.yml`.

You can still install additional packages itself by `pip install $PACKAGE`

Make sure that 'dasgoclient' command is found by ensuring it is in your `$PATH` variable
(`setup_env.sh` does this; without it, run `source env_setup_complete.sh` after activating the
environment).

