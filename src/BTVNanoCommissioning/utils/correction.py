import importlib.resources
import contextlib
import copy
import gzip
import json
import os
import re
import warnings
from collections.abc import MutableMapping
from types import SimpleNamespace

import numpy as np
import awkward as ak
import uproot
from BTVNanoCommissioning.helpers import correctionlib_compat
import numba

try:
    import onnxruntime as ort
except Exception:
    ort = None

from coffea.lookup_tools import extractor, txt_converters, rochester_lookup
from coffea.lumi_tools import LumiMask
from coffea.analysis_tools import Weights, WeightStatistics

from BTVNanoCommissioning.helpers.MuonScaRe import (
    pt_resol,
    pt_scale,
    pt_resol_var,
    pt_scale_var,
)
from BTVNanoCommissioning.helpers.func import campaign_map
from BTVNanoCommissioning.utils.AK4_parameters import correction_config as config
from BTVNanoCommissioning.utils.selection import jet_id

_TTBAR_REWEIGHT_CACHE = {}
_JPCALIB_CACHE = {}


def _cvmfs_dir(campaign, pog):
    """Get CVMFS subdirectory for a campaign and POG, with optional per-POG overrides.

    If the campaign config contains a 'cvmfs_override' dict with a key matching *pog*,
    return that value directly. Otherwise fall back to campaign_map()[campaign].
    This allows campaigns like Winter25 to route MUO/EGM to a different CVMFS era
    than the default one derived from the campaign name.
    """
    overrides = config.get(campaign, {}).get("default", {}).get("cvmfs_override", {})
    if pog in overrides:
        return overrides[pog]
    return campaign_map()[campaign]


def _load_SF_impl(year, campaign, selMod="default", syst=False):
    """
    Load scale factors (SF) for a given year and campaign.

    This function reads scale factors from the specified campaign configuration and returns them in a suitable format.
    It handles different types of scale factors, such as pileup weights, and checks for the existence of files in
    the CMS analysis corrections metadata directory or custom files.

    Example:
    ```python
    ## Initialization, add EGM map from correctionlib
    correction_map["EGM"] = correctionlib_compat.from_file(
        f"/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/{campaign_map()[campaign]}/latest/{_ele_file}.json.gz"
    )
    ## Initialization, add EGM map from custom file by extractor
    ext = extractor()
    ext.add_weight_sets(["eleID EGamma2D {filename}.root"])
    ext.finalize()
    correction_map["EGM"] = ext.make_evaluator()
    ```

    Parameters:
    year (str): The year for which to load the scale factors.
    campaign (str): The name of the campaign for which to load the scale factors.
    selMod (str, optional): The unique selection modifier of the workflow for which to load the scale
        factors. Default is "default".
    syst (bool, optional): A flag to indicate whether to load systematic variations. Default is False.

    Returns:
    dict: A dictionary containing the scale factors, where keys are the relevant identifiers and values are the scale factors.

    Raises:
    FileNotFoundError: If the specified file does not exist.
    ValueError: If the file content is not in the expected format.
    KeyError: If the specified campaign or year is not found in the configuration.
    """
    # read the configuration file to get the correct SFs
    correct_map = {"campaign": campaign}

    conf = copy.copy(config[campaign]["default"])
    if selMod != "default" and selMod in config[campaign].keys():
        for key in config[campaign][selMod].keys():
            conf[key] = copy.copy(config[campaign][selMod][key])

    for SF in conf.keys():
        if SF == "DC":
            continue

        ## pileup weight
        if SF == "LUM":
            ## Check whether files in jsonpog-integration exist
            _lum_cvmfs = _cvmfs_dir(campaign, "LUM")
            if os.path.exists(
                f"/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/{_lum_cvmfs}/latest/"
            ):
                _lum_dir = (
                    f"/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/{_lum_cvmfs}/latest"
                )
                if conf.get("LUM") and os.path.exists(f"{_lum_dir}/{conf['LUM']}"):
                    # the file named in the campaign config (e.g. 2024 C-I only)
                    correct_map["LUM"] = correctionlib_compat.from_file(
                        f"{_lum_dir}/{conf['LUM']}"
                    )
                else:
                    try:
                        correct_map["LUM"] = correctionlib_compat.from_file(
                            f"/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/{_lum_cvmfs}/latest/puWeights.json.gz"
                        )
                    except FileNotFoundError:
                        try:
                            correct_map["LUM"] = correctionlib_compat.from_file(
                                f"/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/{_lum_cvmfs}/latest/puWeights_BCDEFGHI.json.gz"
                            )
                        except FileNotFoundError:
                            correct_map["LUM"] = correctionlib_compat.from_file(
                                f"/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/{_lum_cvmfs}/latest/puWeights_2025pp_Golden_Summer24_25ns_69200ub.json.gz"
                            )
            ## Otherwise custom files
            else:
                _pu_path = f"BTVNanoCommissioning.data.LUM.{campaign}"
                with importlib.resources.path(_pu_path, conf["LUM"]) as filename:
                    if str(filename).endswith(".json.gz"):
                        correct_map["LUM"] = correctionlib_compat.from_file(
                            str(filename)
                        )
                    elif str(filename).endswith(".histo.root"):
                        ext = extractor()
                        ext.add_weight_sets([f"* * {filename}"])
                        ext.finalize()
                        correct_map["LUM"] = ext.make_evaluator()

        ## btag weight
        elif SF == "BTV":
            if "btag" in conf["BTV"].keys() and conf["BTV"]["btag"].endswith(
                ".json.gz"
            ):
                filename = conf["BTV"]["btag"]
                correct_map["btag"] = correctionlib_compat.from_file(
                    f"{os.getcwd()}/src/BTVNanoCommissioning/data/BTV/{campaign}/{filename}"
                )
            if "ctag" in conf["BTV"].keys() and conf["BTV"]["ctag"].endswith(
                ".json.gz"
            ):
                filename = conf["BTV"]["ctag"]
                correct_map["ctag"] = correctionlib_compat.from_file(
                    f"{os.getcwd()}/src/BTVNanoCommissioning/data/BTV/{campaign}/{filename}"
                )
            _btv_cvmfs = _cvmfs_dir(campaign, "BTV")
            if os.path.exists(
                f"/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/{_btv_cvmfs}/latest/"
            ):
                if "btag" not in correct_map.keys():
                    correct_map["btag"] = correctionlib_compat.from_file(
                        f"/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/{_btv_cvmfs}/latest/btagging.json.gz"
                    )
                if "ctag" not in correct_map.keys():
                    correct_map["ctag"] = correctionlib_compat.from_file(
                        f"/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/{_btv_cvmfs}/latest/ctagging.json.gz"
                    )
            else:
                correct_map["btag"] = {}
                correct_map["ctag"] = {}
                correct_map["BTV_cfg"] = conf["BTV"]
                _btag_path = f"BTVNanoCommissioning.data.BTV.{campaign}"
                for tagger in conf["BTV"]:
                    with importlib.resources.path(
                        _btag_path, conf["BTV"][tagger]
                    ) as filename:
                        if "B" in tagger:
                            correct_map["btag"] = correctionlib_compat.from_file(
                                filename
                            )
                        else:
                            correct_map["ctag"] = correctionlib_compat.from_file(
                                filename
                            )

        ## lepton SFs
        elif SF == "MUO":
            correct_map["MUO_cfg"] = {
                mu: f
                for mu, f in conf["MUO"].items()
                if "mu" in mu and "_json" not in mu
            }
            _muo_cvmfs = _cvmfs_dir(campaign, "MUO")
            _mu_path = f"/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/{_muo_cvmfs}/latest/muon_Z.json.gz"
            if os.path.exists(_mu_path):
                correct_map["MUO"] = correctionlib_compat.from_file(_mu_path)
            ## json
            if any(np.char.find(np.array(list(conf["MUO"].keys())), "mu_json") != -1):
                correct_map["MUO"] = correctionlib_compat.from_file(
                    f"src/BTVNanoCommissioning/data/MUO/{_muo_cvmfs}/latest/{conf['MUO']['mu_json']}"
                )

            ## check if any custom corrections needed
            # FIXME: (some low pT muons not supported in CMS analysis corrections at the moment)
            if (
                "histo.json" in "\t".join(list(conf["MUO"].values()))
                or "histo.txt" in "\t".join(list(conf["MUO"].values()))
                or "histo.root" in "\t".join(list(conf["MUO"].values()))
            ):
                _mu_path = f"BTVNanoCommissioning.data.MUO.{campaign}"
                ext = extractor()
                with contextlib.ExitStack() as stack:
                    inputs, real_paths = (
                        [
                            k
                            for k in correct_map["MUO_cfg"].keys()
                            if "histo.json" in correct_map["MUO_cfg"][k]
                            or "histo.txt" in correct_map["MUO_cfg"][k]
                            or "histo.root" in correct_map["MUO_cfg"][k]
                        ],
                        [
                            stack.enter_context(importlib.resources.path(_mu_path, f))
                            for f in correct_map["MUO_cfg"].values()
                            if ".json" in f or ".txt" in f or ".root" in f
                        ],
                    )

                    inputs = [
                        i.split(" ")[0] + " *" if "_low" in i else i for i in inputs
                    ]

                    ext.add_weight_sets(
                        [
                            f"{paths} {file}"
                            for paths, file in zip(inputs, real_paths)
                            if "histo.json" in str(file)
                            or "histo.txt" in str(file)
                            or "histo.root" in str(file)
                        ]
                    )
                    if syst:
                        ext.add_weight_sets(
                            paths.split(" ")[0]
                            + "_error "
                            + paths.split(" ")[1]
                            + "_error "
                            + file
                            for paths, file in zip(inputs, real_paths)
                            if ".root" in str(file)
                        )
                ext.finalize()
                correct_map["MUO_custom"] = ext.make_evaluator()
        elif SF == "EGM":
            correct_map["EGM_cfg"] = {
                e: f for e, f in conf["EGM"].items() if "ele" in e and "_json" not in e
            }
            _egm_cvmfs = _cvmfs_dir(campaign, "EGM")
            for _ele_file, _ele_map in {
                "electron": "EGM",
                "electronHlt": "EGM_HLT",
            }.items():
                _ele_path = f"/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/{_egm_cvmfs}/latest/{_ele_file}.json.gz"
                if not os.path.exists(_ele_path):
                    _ele_path = f"src/BTVNanoCommissioning/data/EGM/{_egm_cvmfs}/latest/{_ele_file}.json.gz"
                if os.path.exists(_ele_path):
                    correct_map[_ele_map] = correctionlib_compat.from_file(
                        _ele_path
                    )
            ## json
            if any(np.char.find(np.array(list(conf["EGM"].keys())), "ele_json") != -1):
                correct_map["EGM"] = correctionlib_compat.from_file(
                    f"src/BTVNanoCommissioning/data/EGM/{_egm_cvmfs}/latest/{conf['EGM']['ele_json']}"
                )

            ## check if any custom corrections needed
            # FIXME: (some low pT muons not supported in CMS analysis corrections at the moment)
            if (
                "histo.json" in "\t".join(list(conf["EGM"].values()))
                or "histo.txt" in "\t".join(list(conf["EGM"].values()))
                or "histo.root" in "\t".join(list(conf["EGM"].values()))
            ):
                _ele_path = f"BTVNanoCommissioning.data.EGM.{campaign}"
                ext = extractor()
                with contextlib.ExitStack() as stack:
                    inputs, real_paths = (
                        [
                            k
                            for k in correct_map["EGM_cfg"].keys()
                            if "histo.json" in correct_map["EGM_cfg"][k]
                            or "histo.txt" in correct_map["EGM_cfg"][k]
                            or "histo.root" in correct_map["EGM_cfg"][k]
                        ],
                        [
                            stack.enter_context(importlib.resources.path(_ele_path, f))
                            for f in correct_map["EGM_cfg"].values()
                            if "histo.json" in f or ".txt" in f or ".root" in f
                        ],
                    )
                    ext.add_weight_sets(
                        [
                            f"{paths} {file}"
                            for paths, file in zip(inputs, real_paths)
                            if "histo.json" in str(file)
                            or "histo.txt" in str(file)
                            or "histo.root" in str(file)
                        ]
                    )
                    if syst:
                        ext.add_weight_sets(
                            paths.split(" ")[0]
                            + "_error "
                            + paths.split(" ")[1]
                            + "_error "
                            + file
                            for paths, file in zip(inputs, real_paths)
                            if ".root" in str(file)
                        )
                ext.finalize()
                correct_map["EGM_custom"] = ext.make_evaluator()

        ## lepton scale & smearing
        elif SF == "muonSS":
            _muss_cvmfs = _cvmfs_dir(campaign, "muonSS")
            _mu_path = f"/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/{_muss_cvmfs}/latest/muon_scalesmearing.json.gz"
            if not os.path.exists(_mu_path):
                _mu_path = f"src/BTVNanoCommissioning/data/MUO/{_muss_cvmfs}/latest/muon_scalesmearing.json.gz"
            if os.path.exists(_mu_path):
                correct_map["muonSS"] = correctionlib_compat.from_file(_mu_path)
        elif SF == "electronSS":
            _eless_cvmfs = _cvmfs_dir(campaign, "electronSS")
            _ele_path = f"/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/{_eless_cvmfs}/latest/electronSS_EtDependent.json.gz"
            if not os.path.exists(_ele_path):
                _ele_path = f"src/BTVNanoCommissioning/data/EGM/{_eless_cvmfs}/latest/electronSS_EtDependent.json.gz"
            if os.path.exists(_ele_path):
                correct_map["electronSS"] = correctionlib_compat.from_file(
                    _ele_path
                )
            correct_map["electronSS_cfg"] = conf["electronSS"]

        ## Rochester muon momentum correction (Run 2)
        elif SF == "roccor":
            if "2016postVFP-UL" == campaign:
                filename = "RoccoR2016bUL.txt"
            elif "2016preVFP-UL" in campaign:
                filename = "RoccoR2016aUL.txt"
            elif "2017-UL" in campaign:
                filename = "RoccoR2017UL.txt"
            if "2018-UL" in campaign:
                filename = "RoccoR2018UL.txt"

            full_path = "src/BTVNanoCommissioning/data/MUO/roccor/" + filename
            rochester_data = txt_converters.convert_rochester_file(
                full_path, loaduncs=True
            )
            correct_map["roccor"] = rochester_lookup.rochester_lookup(rochester_data)

        ## JME corrections
        elif SF == "JME":
            # test hook: BTV_JEC_ERA="<cvmfs JME era>:<JEC version>:<JER version>" loads another
            # era's JEC/JER for MC and all data eras (unset = nominal config)
            if os.environ.get("BTV_JEC_ERA"):
                _era, _jec, _jer = os.environ["BTV_JEC_ERA"].split(":")
                conf = {
                    **conf,
                    "JME_path": f"/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/{_era}/latest/jet_jerc.json.gz",
                    "JME": {
                        k: f"{_jec} {_jer}" if k == "MC" else _jec for k in conf["JME"]
                    },
                }
                warnings.warn(f"BTV_JEC_ERA test hook: JME {conf['JME']}")
            if "JME_path" in conf and os.path.exists(conf["JME_path"]):
                # Custom JME path (e.g. preliminary Puppi JEC for Run 2 NanoAODv15)
                correct_map["JME"] = correctionlib_compat.from_file(
                    conf["JME_path"]
                )
                correct_map["JME_cfg"] = conf["JME"]
                # FIXME need to store the JSON path in case we want to parse the run binning edges for ad-hoc JEC run clamp fix (see _get_jec_run_edges)
                correct_map["JME_json_path"] = conf["JME_path"]
                for dataset in correct_map["JME_cfg"].keys():
                    if (
                        np.all(
                            np.char.find(
                                np.array(list(correct_map["JME"].keys())),
                                correct_map["JME_cfg"][dataset],
                            )
                        )
                        == -1
                    ):
                        raise (
                            f"{dataset} has no JEC map : {correct_map['JME_cfg'][dataset]} available"
                        )
            elif os.path.exists(
                f"/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/{_cvmfs_dir(campaign, 'JME')}/latest/jet_jerc.json.gz"
            ):
                _jme_path = f"/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/{_cvmfs_dir(campaign, 'JME')}/latest/jet_jerc.json.gz"
                # If MC corrections live in a different CVMFS era (e.g.
                # Summer24 MC truth for Winter25 data), merge at JSON
                # level so we get a single native CorrectionSet.
                _jme_mc_era = _cvmfs_dir(campaign, "JME_MC")
                _jme_data_era = _cvmfs_dir(campaign, "JME")
                _mc_path = f"/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/{_jme_mc_era}/latest/jet_jerc.json.gz"
                if _jme_mc_era != _jme_data_era and os.path.exists(_mc_path):
                    with gzip.open(_jme_path, "rt") as f:
                        _data_json = json.load(f)
                    with gzip.open(_mc_path, "rt") as f:
                        _mc_json = json.load(f)
                    existing = {c["name"] for c in _data_json["corrections"]}
                    _data_json["corrections"] += [
                        c for c in _mc_json["corrections"] if c["name"] not in existing
                    ]
                    existing_comp = {
                        c["name"] for c in _data_json.get("compound_corrections", [])
                    }
                    _data_json.setdefault("compound_corrections", []).extend(
                        c
                        for c in _mc_json.get("compound_corrections", [])
                        if c["name"] not in existing_comp
                    )
                    correct_map["JME"] = correctionlib_compat.from_string(
                        json.dumps(_data_json)
                    )
                else:
                    correct_map["JME"] = correctionlib_compat.from_file(
                        _jme_path
                    )
                correct_map["JME_cfg"] = conf["JME"]
                # FIXME need to store the JSON path in case we want to parse the run binning edges for ad-hoc JEC run clamp fix (see _get_jec_run_edges)
                correct_map["JME_json_path"] = _jme_path
                for dataset in correct_map["JME_cfg"].keys():
                    if (
                        np.all(
                            np.char.find(
                                np.array(list(correct_map["JME"].keys())),
                                correct_map["JME_cfg"][dataset],
                            )
                        )
                        == -1
                    ):
                        raise (
                            f"{dataset} has no JEC map : {correct_map['JME_cfg'][dataset]} available"
                        )
        elif SF == "JMAR":
            if os.path.exists(
                f"/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/{_cvmfs_dir(campaign, 'JMAR')}/latest/jmar.json.gz"
            ):
                correct_map["JMAR_cfg"] = {j: f for j, f in conf["JMAR"].items()}
                correct_map["JMAR"] = correctionlib_compat.from_file(
                    f"/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/{_cvmfs_dir(campaign, 'JMAR')}/latest/jmar.json.gz"
                )
        elif SF == "jetveto":
            correct_map["jetveto_cfg"] = {j: f for j, f in conf["jetveto"].items()}

            isRootFile = False
            for val in correct_map["jetveto_cfg"].values():
                if ".root" in val:
                    isRootFile = True

            if not isRootFile and os.path.exists(
                f"/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/{_cvmfs_dir(campaign, 'jetveto')}/latest/jetvetomaps.json.gz"
            ):
                correct_map["jetveto"] = correctionlib_compat.from_file(
                    f"/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/{_cvmfs_dir(campaign, 'jetveto')}/latest/jetvetomaps.json.gz"
                )
            else:
                ext = extractor()
                with contextlib.ExitStack() as stack:
                    ext.add_weight_sets(
                        [
                            f"{run} {stack.enter_context(importlib.resources.path(f'BTVNanoCommissioning.data.JME.{campaign}', file))}"
                            for run, file in conf["jetveto"].items()
                        ]
                    )
                    ext.finalize()
                    correct_map["jetveto"] = ext.make_evaluator()

    return correct_map


_SF_MAP_CACHE = {}


class _LazySFMap(MutableMapping):
    """Picklable placeholder for load_SF()'s output.

    Pickles down to just (year, campaign, selMod, syst); the actual CorrectionSets/evaluators are
    built (or reused from _SF_MAP_CACHE) lazily on first real access in whatever process unpickles
    it. This keeps coffea's per-chunk-task processor_instance pickling cheap under dask/parsl
    executors instead of shipping a multi-MB payload with every task.
    """

    def __init__(self, year, campaign, selMod="default", syst=False):
        self._key = (year, campaign, selMod, syst)

    def _resolve(self):
        if self._key not in _SF_MAP_CACHE:
            _SF_MAP_CACHE[self._key] = _load_SF_impl(*self._key)
        resolved = _SF_MAP_CACHE[self._key]
        assert (
            resolved.get("campaign") == self._key[1]
        ), f"SF_map cache key/content mismatch: expected campaign {self._key[1]!r}, got {resolved.get('campaign')!r}"
        return resolved

    def __getitem__(self, key):
        return self._resolve()[key]

    def __setitem__(self, key, value):
        self._resolve()[key] = value

    def __delitem__(self, key):
        del self._resolve()[key]

    def __iter__(self):
        return iter(self._resolve())

    def __len__(self):
        return len(self._resolve())

    def __repr__(self):
        return repr(self._resolve())

    def __reduce__(self):
        return (_LazySFMap, self._key)


def load_SF(year, campaign, selMod="default", syst=False):
    return _LazySFMap(year, campaign, selMod, syst)


def load_lumi(campaign):
    """
    Load luminosity mask for a given campaign.

    This function reads the luminosity mask file for the specified campaign and returns a `LumiMask` object.

    Parameters:
    campaign (str): The name of the campaign for which to load the luminosity mask.

    Returns:
    LumiMask: An object representing the luminosity mask for the specified campaign.

    Raises:
    KeyError: If the specified campaign is not found in the configuration.
    FileNotFoundError: If the luminosity mask file does not exist.
    """

    _lumi_path = f"/cvmfs/cms-griddata.cern.ch/cat/metadata/DC/Collisions{campaign[-2:]}/latest/{config[campaign]['default']['DC']}"
    if os.path.exists(_lumi_path):
        return LumiMask(_lumi_path)
    else:
        _lumi_path = "BTVNanoCommissioning.data.DC"
        with importlib.resources.path(
            _lumi_path, config[campaign]["default"]["DC"]
        ) as filename:
            return LumiMask(filename)


# FIXME: JEC run-number ad-hoc boundary fix
# See: https://cms-talk.web.cern.ch/t/bug-in-2025-jerc-json-file/47675
# In March 2026, there is a bug in the JEC corrections, as
# correctionlib bins are half-open [low, high), so the last run in each
# era is excluded from the L2L3Residual correction.  This helper fixes
# the valid run range so we can clamp before evaluate().

_jec_run_edge_cache = {}


def _get_jec_run_edges(correct_map, jecname, jet_algo="AK4PFPuppi"):
    """Return (lo, hi) run clamp range from L2L3Residual bin edges.

    Parses the JME JSON to find the run binning edges directly.
    correctionlib uses half-open [lo, hi) bins, so the last valid
    integer run is edges[-1] - 1.
    Returns None if no run-binned L2L3Residual exists.
    The result is cached per jecname.
    """
    cache_key = (jecname, jet_algo)
    if cache_key in _jec_run_edge_cache:
        return _jec_run_edge_cache[cache_key]

    jme_json_path = correct_map.get("JME_json_path")
    if jme_json_path is None:
        _jec_run_edge_cache[cache_key] = None
        return None

    l2l3_name = f"{jecname}_L2L3Residual_{jet_algo}"
    result = None
    with gzip.open(jme_json_path, "rt") as f:
        for corr in json.load(f).get("corrections", []):
            if corr["name"] != l2l3_name:
                continue
            inputs = corr.get("inputs", [])
            data = corr.get("data", {})
            if (
                inputs
                and inputs[0].get("name") == "run"
                and data.get("nodetype") == "binning"
                and data.get("flow") == "error"
            ):
                edges = data["edges"]
                result = (edges[0], edges[-1] - 1)
            break

    _jec_run_edge_cache[cache_key] = result
    return result


## JEC
# FIXME: would be nicer if we can move to correctionlib in the future together with factory and workable


def add_jec_variables(jets, event_rho):
    jets["pt_raw"] = (1 - jets.rawFactor) * jets.pt
    jets["mass_raw"] = (1 - jets.rawFactor) * jets.mass
    if hasattr(jets, "genJetIdx"):
        jets["pt_gen"] = ak.values_astype(
            ak.fill_none(jets.matched_gen.pt, 0), np.float32
        )
    else:
        jets["pt_gen"] = ak.zeros_like(jets.pt)
    jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]
    return jets


## Jet Veto
def jetveto(jets, correct_map):
    """
    Apply a veto to jets based on predefined transverse momentum (pt) and pseudorapidity (eta) thresholds.

    This function filters out jets that do not meet the predefined pt and eta criteria. It also utilizes a correction map
    to apply additional corrections or selections to the jets.

    Parameters:
    jets (iterable): A collection of jet objects or dictionaries containing jet properties.
    correct_map (dict): A dictionary containing correction factors or additional selection criteria for the jets.

    Returns:
    jets: A collection of jets that pass the predefined pt and eta criteria and any additional criteria from the correction map.

    Raises:
    TypeError: If the jets parameter is not an iterable.
    KeyError: If the jet objects do not contain the required 'pt' or 'eta' properties.
    """

    if "correctionlib" in str(
        type(correct_map["jetveto"][list(correct_map["jetveto"].keys())[0]])
    ):
        j, nj = ak.flatten(jets), ak.num(jets)
        return ak.unflatten(
            correct_map["jetveto"][list(correct_map["jetveto"].keys())[0]].evaluate(
                correct_map["jetveto_cfg"][list(correct_map["jetveto"].keys())[0]],
                np.clip(j.eta, -5.191, 5.191),
                np.clip(j.phi, -3.141592653589793, 3.141592653589793),
            ),
            nj,
        )
    else:
        return ak.where(
            correct_map["jetveto"][list(correct_map["jetveto"].keys())[0]](
                jets.phi, jets.eta
            )
            > 0,
            ak.broadcast_arrays(jets.eta, 100),
            ak.zeros_like(jets.eta),
        )


# from https://gitlab.cern.ch/cms-analysis-corrections/JME/examples/-/blob/latest/jercExample.py
def get_corr_inputs(input_dict, corr_obj, jersyst="nom"):
    """
    Helper function for getting values of input variables
    given a dictionary and a correction object.
    """
    input_values = []
    for inputs in corr_obj.inputs:
        if "systematic" in inputs.name:
            input_values.append(jersyst)
        else:
            input_values.append(
                np.array(
                    input_dict[
                        inputs.name.replace("Jet", "")
                        .replace("Pt", "pt")
                        .replace("Phi", "phi")
                        .replace("Eta", "eta")
                        .replace("Mass", "mass")
                        .replace("Rho", "evt_rho")
                        .replace("A", "area")
                    ]
                )
            )
    return input_values


_jersmear_path = (
    "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/JER-Smearing/latest/jer_smear.json.gz"
)
if os.path.exists(_jersmear_path):
    cset_jersmear = correctionlib_compat.from_file(_jersmear_path)
else:
    warnings.warn(
        "JER smearing JSON not found in CVMFS. JER smearing corrections will be disabled.",
        RuntimeWarning,
        stacklevel=2,
    )
    cset_jersmear = {"JERSmear": None}
sf_jersmear = cset_jersmear["JERSmear"]


# JEC/JES sources for the full set according to
# https://cms-jerc.web.cern.ch/Recommendations/#jet-energy-scale_1
def get_JES_keys(year: int | str, campaign: str = "") -> dict[str, set]:
    if campaign == "2016preVFP-UL":
        year = "2016APV"
    return {
        "full": {
            "AbsoluteMPFBias",
            "AbsoluteScale",
            "AbsoluteStat",
            "FlavorQCD",
            "Fragmentation",
            "PileUpDataMC",
            "PileUpPtBB",
            "PileUpPtEC1",
            "PileUpPtEC2",
            "PileUpPtHF",
            "PileUpPtRef",
            "RelativeFSR",
            "RelativeJEREC1",
            "RelativeJEREC2",
            "RelativeJERHF",
            "RelativePtBB",
            "RelativePtEC1",
            "RelativePtEC2",
            "RelativePtHF",
            "RelativeBal",
            "RelativeSample",
            "RelativeStatEC",
            "RelativeStatFSR",
            "RelativeStatHF",
            "SinglePionECAL",
            "SinglePionHCAL",
            "TimePtEta",
        },
        "reduced": {
            f"Regrouped_Absolute_{year}",
            "Regrouped_Absolute",
            f"Regrouped_BBEC1_{year}",
            "Regrouped_BBEC1",
            f"Regrouped_EC2_{year}",
            "Regrouped_EC2",
            "Regrouped_FlavorQCD",
            f"Regrouped_HF_{year}",
            "Regrouped_HF",
            "Regrouped_RelativeBal",
            f"Regrouped_RelativeSample_{year}",
        },
        "total": {"Total"},
    }


# JER systematic eta bins according to
# https://cms-jerc.web.cern.ch/Recommendations/#jet-energy-resolution_1
def get_JER_bins():
    return {
        "split": {
            "eta0to1p93": [0.0, 1.93],
            "eta1p93to2p5": [1.93, 2.5],
        },
        "total": {
            "Total": [0.0, 2.5],
        },
    }


def get_JER(
    correct_map,
    jername,
    j,
    jersyst,
    jet_algo="AK4PFPuppi",
):
    """
    Retrieve JER for each jet pt shift.

    Parameters:
    correct_map (dict): A dictionary containing correction factors and settings for JER.
    jername (str): name of the JER
    j (jets): jet collection
    """

    # New JERC format (2026): ScaleFactor and SFUncertainty are separate keys,
    # both taking (JetEta, JetPt) with no systematic string argument.
    # Variations are computed as SF * (1 ± SFUncertainty).
    JERSF_corr = correct_map["JME"][f"{jername}_ScaleFactor_{jet_algo}"]
    sf_nom = JERSF_corr.evaluate(*get_corr_inputs(j, JERSF_corr))

    JERSF_unc_corr = correct_map["JME"][f"{jername}_SFUncertainty_{jet_algo}"]
    sf_unc = JERSF_unc_corr.evaluate(*get_corr_inputs(j, JERSF_unc_corr))

    if jersyst == "up":
        j["JERSF"] = sf_nom * (1 + sf_unc)
    elif jersyst == "down":
        j["JERSF"] = sf_nom * (1 - sf_unc)
    else:
        j["JERSF"] = sf_nom

    JERptres = correct_map["JME"][f"{jername}_PtResolution_{jet_algo}"]
    JERptres_input = get_corr_inputs(j, JERptres)
    j["JER"] = JERptres.evaluate(*JERptres_input)

    if sf_jersmear is None:
        warnings.warn(
            "JER smearing coefficients unavailable. "
            "Proceeding without applying JER smearing.",
            RuntimeWarning,
            stacklevel=2,
        )
        return ak.ones_like(j.pt)
    else:
        JERsmear_input = get_corr_inputs(j, sf_jersmear)
        return sf_jersmear.evaluate(*JERsmear_input)


def get_MET_corr_keys():
    return [
        "pt_raw",
        "eta",
        "phi",
        "area",
        "EmEF",
        "muonSubtrFactor",
        "muonSubtrDeltaPhi",
        "muonSubtrDeltaEta",
        "Genpt",
        "evt_rho",
        "EventID",
        "run",
    ]


def _infer_jec_jet_algo(correct_map, jecname):
    """Infer jet algorithm suffix (e.g. AK4PFPuppi, AK4PFchs) from JEC compound keys."""
    preferred_algos = ["AK4PFPuppi", "AK4PFchs"]
    compound_map = correct_map["JME"].compound

    for algo in preferred_algos:
        key = f"{jecname}_L1L2L3Res_{algo}"
        if key in compound_map.keys():
            return algo

    available = [
        k for k in compound_map.keys() if k.startswith(f"{jecname}_L1L2L3Res_")
    ]
    if len(available) > 0:
        return available[0].replace(f"{jecname}_L1L2L3Res_", "")

    raise KeyError(
        f"No L1L2L3Res JEC key found for '{jecname}'. Available matches: {available}"
    )


def _resolve_jecname(correct_map, jecname):
    """Resolve configured JEC era name to an available L1L2L3Res prefix.

    Some configs use data-era strings containing a run token (e.g. ``..._RunCD_V3_DATA``)
    while the JSON can store the corresponding key without that token (e.g. ``..._V3_DATA``).
    """
    compound_map = correct_map["JME"].compound
    if any(k.startswith(f"{jecname}_L1L2L3Res_") for k in compound_map.keys()):
        return jecname

    candidates = []
    if "_Run" in jecname:
        candidates.append(re.sub(r"_Run[^_]+", "", jecname))

    for cand in candidates:
        if any(k.startswith(f"{cand}_L1L2L3Res_") for k in compound_map.keys()):
            warnings.warn(
                f"Configured JEC name '{jecname}' not found, using '{cand}' instead.",
                RuntimeWarning,
                stacklevel=2,
            )
            return cand

    return jecname


def fixPhiRange(phi):
    return np.mod(phi + np.pi, 2 * np.pi) - np.pi


def calc_T1_MET(
    met_raw,
    met_nano,
    shifted_jets,
    campaign,
    systematic=False,
):
    """
    Type-1 MET correction following the recipe from:
      https://cms-jerc.web.cern.ch/Type1MET/
    Reference talk (JME General, 2 Feb 2026, Fikri):
      https://indico.cern.ch/event/1644923/#5-met-type-1-correction-comput

    Expects the following pre-computed fields on shifted_jets
    (set by JME_shifts before calling this function):
      - pt_noMuL1:        muon-subtracted raw pT × L1FastJet
      - pt_noMu_fullcorr: muon-subtracted raw pT × L1L2L3Res (× JER/JES-syst)
    CMS Talk discussion:
      https://cms-talk.web.cern.ch/t/unclustered-met-uncertainty-2022-2023-nanoaodv12/141752/5
    """

    MET_x_baseline = met_raw.pt * np.cos(met_raw.phi)
    MET_y_baseline = met_raw.pt * np.sin(met_raw.phi)

    if campaign in [
        "Summer24",
        "CustomSummer24",
        "CustomSummer24UParTPaper",
        "CustomSummer25",
        "Winter25",
        "Prompt25",
    ]:  # NanoAODv15
        jet_mask = (
            (shifted_jets["pt_noMu_fullcorr"] > 15.0)
            & (abs(shifted_jets["eta"]) < 5.2)
            & (shifted_jets["EmEF"] < 0.9)
        )
    else:  # NanoAODv12
        jet_mask = (shifted_jets["pt_noMu_fullcorr"] > 15.0) & (
            abs(shifted_jets["eta"]) < 5.2
        )

    delta_pt = (
        shifted_jets["pt_noMu_fullcorr"][jet_mask] - shifted_jets["pt_noMuL1"][jet_mask]
    )
    try:
        # NanoAODv15: use muon-subtracted phi (phi + muonSubtrDeltaPhi)
        jet_phi = shifted_jets["phi_noMuRaw"][jet_mask]
    except Exception:
        # Fallback for older NanoAOD versions without muonSubtrDeltaPhi
        jet_phi = shifted_jets["phi"][jet_mask]

    MET_x_shifts = delta_pt * np.cos(jet_phi)
    MET_y_shifts = delta_pt * np.sin(jet_phi)
    MET_x = MET_x_baseline - ak.sum(MET_x_shifts, axis=-1)
    MET_y = MET_y_baseline - ak.sum(MET_y_shifts, axis=-1)
    pt_miss_final = np.sqrt(MET_x * MET_x + MET_y * MET_y)
    phi_miss_final = np.arctan2(MET_y, MET_x)
    phi_miss_final = fixPhiRange(phi_miss_final)

    if systematic:
        if hasattr(met_nano, "ptUnclusteredUp"):
            # NanoAODv9 and earlier: extract the unclustered delta from NanoAOD
            # by comparing the shifted MET to the nominal MET in x/y components,
            # then apply that same delta to our recomputed MET.
            met_nano_x = met_nano.pt * np.cos(met_nano.phi)
            met_nano_y = met_nano.pt * np.sin(met_nano.phi)
            dx = (
                met_nano.ptUnclusteredUp * np.cos(met_nano.phiUnclusteredUp)
                - met_nano_x
            )
            dy = (
                met_nano.ptUnclusteredUp * np.sin(met_nano.phiUnclusteredUp)
                - met_nano_y
            )
            pt_unclustered_up = np.sqrt((MET_x + dx) ** 2 + (MET_y + dy) ** 2)
            phi_unclustered_up = np.arctan2(MET_y + dy, MET_x + dx)
            dx_dn = (
                met_nano.ptUnclusteredDown * np.cos(met_nano.phiUnclusteredDown)
                - met_nano_x
            )
            dy_dn = (
                met_nano.ptUnclusteredDown * np.sin(met_nano.phiUnclusteredDown)
                - met_nano_y
            )
            pt_unclustered_down = np.sqrt((MET_x + dx_dn) ** 2 + (MET_y + dy_dn) ** 2)
            phi_unclustered_down = np.arctan2(MET_y + dy_dn, MET_x + dx_dn)
            phi_unclustered_up = fixPhiRange(phi_unclustered_up)
            phi_unclustered_down = fixPhiRange(phi_unclustered_down)
        elif hasattr(met_nano, "MetUnclustEnUpDeltaX"):
            # NanoAODv12+: directional x/y shifts for unclustered energy
            dx = met_nano.MetUnclustEnUpDeltaX
            dy = met_nano.MetUnclustEnUpDeltaY
            pt_unclustered_up = np.sqrt((MET_x + dx) ** 2 + (MET_y + dy) ** 2)
            pt_unclustered_down = np.sqrt((MET_x - dx) ** 2 + (MET_y - dy) ** 2)
            phi_unclustered_up = np.arctan2(MET_y + dy, MET_x + dx)
            phi_unclustered_down = np.arctan2(MET_y - dy, MET_x - dx)
            phi_unclustered_up = fixPhiRange(phi_unclustered_up)
            phi_unclustered_down = fixPhiRange(phi_unclustered_down)
        else:
            # Fallback: isotropic approximation from sumPtUnclustered
            delta = np.sqrt(met_nano.sumPtUnclustered)
            pt_unclustered_up = np.sqrt((MET_x + delta) ** 2 + (MET_y + delta) ** 2)
            pt_unclustered_down = np.sqrt((MET_x - delta) ** 2 + (MET_y - delta) ** 2)
            phi_unclustered_up = np.arctan2(MET_y + delta, MET_x + delta)
            phi_unclustered_down = np.arctan2(MET_y - delta, MET_x - delta)
            phi_unclustered_up = fixPhiRange(phi_unclustered_up)
            phi_unclustered_down = fixPhiRange(phi_unclustered_down)
        return (
            pt_miss_final,
            phi_miss_final,
            pt_unclustered_up,
            pt_unclustered_down,
            phi_unclustered_up,
            phi_unclustered_down,
        )
    else:
        return pt_miss_final, phi_miss_final


## JERC
def JME_shifts(
    shifts,
    correct_map,
    events,
    year,
    campaign,
    isRealData,
    systematic=False,
):
    """
    Apply Jet Energy Corrections (JEC) and Jet Energy Resolutions (JER) shifts to events.

    This function applies JEC and JER shifts to the jets in the events based on the provided correction map and campaign.
    It handles both real data and simulated data, and can optionally apply systematic variations and exclude jet vetoes.

    Parameters:
    shifts (list): A list of shift types to apply (e.g., 'up', 'down').
    correct_map (dict): A dictionary containing correction factors and settings for JEC and JER.
    events (awkward.Array): An array of events containing jet information.
    year (str): The year for which to apply the corrections.
    campaign (str): The name of the campaign for which to apply the corrections.
    isRealData (bool): A flag indicating whether the data is real or simulated.
    systematic (bool, optional): A flag to indicate whether to apply systematic variations. Default is False.

    Returns:
    awkward.Array: The events array with applied JEC and JER shifts.

    Raises:
    KeyError: If required keys are missing in the correct_map.
    ValueError: If the campaign is not recognized or supported.
    """
    dataset = events.metadata["dataset"]

    # Year-dependent JES uncertainty names (e.g. Regrouped_Absolute_YYYY)
    # must use the MC JEC campaign year, not the data year.  When MC JECs are
    # borrowed from a different era (e.g. Summer24 JECs for Winter25 data),
    # the config can set "JES_MC_year" to override.
    jes_year = config.get(campaign, {}).get("default", {}).get("JES_MC_year", year)

    if not isRealData and systematic != False:
        jerc_id_arr = systematic.split("_")
        jes_sources = get_JES_keys(jes_year, campaign)
        if len(jerc_id_arr) >= 2 and jerc_id_arr[0] == "JEC":
            jes_sources_id = jerc_id_arr[1]
        else:
            jes_sources_id = "reduced"  # Default case
        jer_split = get_JER_bins()
        if len(jerc_id_arr) == 4 and jerc_id_arr[2] == "JER":
            jer_split_id = jerc_id_arr[3]
        else:
            jer_split_id = "total"  # Default case
        if systematic == "all":
            jes_sources_id = "full"
            jer_split_id = "split"

    jecname = ""
    if "JME" in correct_map.keys():
        ## correctionlib
        if "JME_cfg" in correct_map.keys():
            if isRealData:
                jecname = [
                    v
                    for k, v in correct_map["JME_cfg"].items()
                    if k in events.metadata["dataset"]
                ]
                if len(jecname) > 1:
                    raise ValueError("Multiple uncertainties match to this era")
                elif len(jecname) == 0:
                    raise ValueError(
                        "Available JEC variations in this era are not compatible with this file. Did you choose the correct dataset-era combination?"
                    )
                else:
                    jecname = jecname[0] + "_DATA"
            else:
                jecname = correct_map["JME_cfg"]["MC"].split(" ")[0] + "_MC"
                jername = correct_map["JME_cfg"]["MC"].split(" ")[1] + "_MC"

            jecname = _resolve_jecname(correct_map, jecname)
            jet_algo = _infer_jec_jet_algo(correct_map, jecname)

            ## store the original jet info
            nocorrjet = events.Jet
            nocorrjet["pt_nano"] = ak.values_astype(events.Jet["pt"], np.float32)
            nocorrjet["mass_nano"] = ak.values_astype(events.Jet["mass"], np.float32)
            nocorrjet["pt_raw"] = (1 - events.Jet["rawFactor"]) * events.Jet["pt"]
            nocorrjet["mass_raw"] = (1 - events.Jet["rawFactor"]) * events.Jet["mass"]
            nocorrjet["EmEF"] = events.Jet["chEmEF"] + events.Jet["neEmEF"]
            if "muonSubtrDeltaPhi" in events.Jet.fields:
                nocorrjet["muonSubtrDeltaPhi"] = events.Jet["muonSubtrDeltaPhi"]
            else:
                nocorrjet["muonSubtrDeltaPhi"] = ak.zeros_like(nocorrjet.pt)
            if "muonSubtrDeltaEta" in events.Jet.fields:
                nocorrjet["muonSubtrDeltaEta"] = events.Jet["muonSubtrDeltaEta"]
            else:
                nocorrjet["muonSubtrDeltaEta"] = ak.zeros_like(nocorrjet.pt)
            if not isRealData:
                invalid_genjetidx = (events.Jet.genJetIdx == -1) | (
                    events.Jet.genJetIdx >= ak.num(events.GenJet)
                )
                genjetidx = ak.where(invalid_genjetidx, 0, events.Jet.genJetIdx)
                # pad GenJet so index 0 is always structurally valid, even for
                # events with zero GenJets -- ak.where evaluates both branches
                # eagerly, so events.GenJet[genjetidx] would otherwise raise
                # on those events despite the result being masked away below
                # (same gotcha documented in helpers/func.py's SV_link_ragged)
                padded_genjet_pt = ak.pad_none(events.GenJet.pt, 1, axis=1)
                nocorrjet["Genpt"] = ak.where(
                    invalid_genjetidx,
                    -1,
                    ak.fill_none(padded_genjet_pt[genjetidx], -1),
                )
            nocorrjet["evt_rho"] = ak.broadcast_arrays(
                events.fixedGridRhoFastjetAll, nocorrjet.pt
            )[0]
            nocorrjet["EventID"] = ak.broadcast_arrays(events.event, nocorrjet.pt)[0]
            nocorrjet["run"] = ak.broadcast_arrays(events.run, nocorrjet.pt)[0]
            jets = copy.copy(nocorrjet)
            jets["pt"] = nocorrjet["pt_raw"]
            jets["mass"] = nocorrjet["mass_raw"]

            ## build jet collection for type-1 MET correction
            nocorrt1metjet = copy.copy(events.CorrT1METJet)
            nocorrt1metjet["pt_raw"] = events.CorrT1METJet["rawPt"]
            if not isRealData:
                nocorrt1metjet["Genpt"] = ak.broadcast_arrays(
                    -1, events.CorrT1METJet.rawPt
                )[0]
            nocorrt1metjet["evt_rho"] = ak.broadcast_arrays(
                events.fixedGridRhoFastjetAll, events.CorrT1METJet.rawPt
            )[0]
            nocorrt1metjet["EventID"] = ak.broadcast_arrays(
                events.event, events.CorrT1METJet.rawPt
            )[0]
            nocorrt1metjet["run"] = ak.broadcast_arrays(
                events.run, events.CorrT1METJet.rawPt
            )[0]
            if "muonSubtrDeltaPhi" not in nocorrt1metjet.fields:
                nocorrt1metjet["muonSubtrDeltaPhi"] = ak.zeros_like(
                    events.CorrT1METJet.rawPt
                )
            if "muonSubtrDeltaEta" not in nocorrt1metjet.fields:
                nocorrt1metjet["muonSubtrDeltaEta"] = ak.zeros_like(
                    events.CorrT1METJet.rawPt
                )
            # adding dummy value of 0 will still pass the EmEF < 0.9 cut
            if "EmEF" not in nocorrt1metjet.fields:
                nocorrt1metjet["EmEF"] = ak.zeros_like(events.CorrT1METJet.rawPt)

            keys_keep = get_MET_corr_keys()
            t1jets_1 = nocorrjet[[key for key in nocorrjet.fields if key in keys_keep]]
            t1jets_2 = nocorrt1metjet[
                [key for key in nocorrt1metjet.fields if key in keys_keep]
            ]
            t1jets = ak.concatenate([t1jets_1, t1jets_2], axis=1)
            t1jets["pt"] = t1jets["pt_raw"]

            ## flatten jets
            j, nj = ak.flatten(jets), ak.num(jets)
            t1j, nt1j = ak.flatten(t1jets), ak.num(t1jets)

            ## Pre-compute muon-subtracted Type-1 MET quantities on flat t1 jets.
            ## These are constant: L1 and L1L2L3Res evaluated at pt_noMuRaw.
            t1j["pt_noMuRaw"] = t1j["pt_raw"] * (1.0 - t1j["muonSubtrFactor"])
            t1j["phi_noMuRaw"] = t1j["phi"] + t1j["muonSubtrDeltaPhi"]
            t1j["eta_noMuRaw"] = t1j["eta"] + t1j["muonSubtrDeltaEta"]

            # Type-1 MET recipe depends on jet algorithm:
            # - AK4PFPuppi (PUPPI MET): L2L3-only, no L1 subtraction
            # - AK4PFchs  (PF MET): L1L2L3 - L1
            # See: https://cms-jerc.web.cern.ch/Type1MET/
            if jet_algo == "AK4PFPuppi":
                t1j["pt_noMuL1"] = t1j["pt_noMuRaw"]
            else:
                l1_key = f"{jecname}_L1FastJet_{jet_algo}"
                if l1_key in correct_map["JME"].keys():
                    L1corr = correct_map["JME"][l1_key]
                    t1j["pt_noMuL1"] = t1j["pt_noMuRaw"] * L1corr.evaluate(
                        np.array(t1j["area"]),
                        np.array(t1j["eta_noMuRaw"]),
                        np.array(t1j["pt_noMuRaw"]),
                        np.array(t1j["evt_rho"]),
                    )
                else:
                    warnings.warn(
                        f"Missing {l1_key}; falling back to no L1 subtraction for Type-1 MET.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    t1j["pt_noMuL1"] = t1j["pt_noMuRaw"]

            ## store the original met info (nocorrmet), raw met, nanoaod met
            ## Use MET object matching the JEC jet algorithm.
            if (
                jet_algo == "AK4PFPuppi"
                and hasattr(events, "PuppiMET")
                and hasattr(events, "RawPuppiMET")
            ):
                nocorrmet = events.PuppiMET
                met_raw = events.RawPuppiMET
                met_nano = events.PuppiMET
            else:
                nocorrmet = events.MET
                met_raw = events.RawMET
                met_nano = events.MET
            met = copy.copy(nocorrmet)

            ## JES/JEC
            JECcorr = correct_map["JME"].compound[f"{jecname}_L1L2L3Res_{jet_algo}"]
            JEC_input = get_corr_inputs(j, JECcorr)
            JEC_input_t1 = get_corr_inputs(t1j, JECcorr)

            # JES/JEC recipe for 2024 (slide 26)
            # https://indico.cern.ch/event/1602054/timetable/?view=standard_inline_minutes#12-jet-performance-in-run-3
            # test hook: BTV_JEC_CLIP30=1/0 forces the pT >= 30 GeV evaluation on/off (unset = 2024 only)
            _clip = os.environ.get("BTV_JEC_CLIP30")
            if (year == "2024") if _clip is None else (_clip == "1"):
                tmp_pt = copy.copy(JEC_input[2])
                tmp_pt = np.clip(tmp_pt, 30, None)
                JEC_input[2] = tmp_pt
                tmp_pt_t1 = JEC_input_t1[2]
                tmp_pt_t1 = np.clip(tmp_pt_t1, 30, None)
                JEC_input_t1[2] = tmp_pt_t1

            # FIXME Ad-hoc fix: clamp run number for data L2L3Residual boundary bug.
            # correctionlib uses half-open [low, high) bins, so the last run in
            # each era is excluded.  Clamp to (first_edge, last_edge - 1) so
            # that boundary runs still get a valid correction.
            # See: https://cms-talk.web.cern.ch/t/bug-in-2025-jerc-json-file/47675
            if isRealData:
                run_idx = next(
                    (i for i, inp in enumerate(JECcorr.inputs) if inp.name == "run"),
                    None,
                )
                if run_idx is not None:
                    _run_edges = _get_jec_run_edges(correct_map, jecname, jet_algo)
                    if _run_edges is not None:
                        lo, hi = _run_edges
                        JEC_input[run_idx] = np.clip(JEC_input[run_idx], lo, hi)
                        JEC_input_t1[run_idx] = np.clip(JEC_input_t1[run_idx], lo, hi)

            JECflatCorrFactor = JECcorr.evaluate(*JEC_input)
            j["pt_JECnom"] = j["pt"] * JECflatCorrFactor
            j["mass_JECnom"] = j["mass"] * JECflatCorrFactor

            JECflatCorrFactor_t1 = JECcorr.evaluate(*JEC_input_t1)
            t1j["pt_JECnom"] = t1j["pt"] * JECflatCorrFactor_t1

            ## Compound L1L2L3Res at muon-subtracted raw pT (for Type-1 MET)
            JEC_input_t1_nomu = get_corr_inputs(t1j, JECcorr)
            # Override pT and eta with muon-subtracted values
            pt_idx_t1 = next(
                i
                for i, inp in enumerate(JECcorr.inputs)
                if inp.name.replace("Jet", "").replace("Pt", "pt") == "pt"
            )
            JEC_input_t1_nomu[pt_idx_t1] = np.array(t1j["pt_noMuRaw"])
            eta_idx_t1 = next(
                i
                for i, inp in enumerate(JECcorr.inputs)
                if inp.name.replace("Jet", "").replace("Eta", "eta") == "eta"
            )
            JEC_input_t1_nomu[eta_idx_t1] = np.array(t1j["eta_noMuRaw"])
            if year == "2024":
                JEC_input_t1_nomu[pt_idx_t1] = np.clip(
                    JEC_input_t1_nomu[pt_idx_t1], 30, None
                )
            if isRealData:
                run_idx = next(
                    (i for i, inp in enumerate(JECcorr.inputs) if inp.name == "run"),
                    None,
                )
                if run_idx is not None:
                    _run_edges = _get_jec_run_edges(correct_map, jecname, jet_algo)
                    if _run_edges is not None:
                        lo, hi = _run_edges
                        JEC_input_t1_nomu[run_idx] = np.clip(
                            JEC_input_t1_nomu[run_idx], lo, hi
                        )
            t1j["pt_noMu_JEC"] = t1j["pt_noMuRaw"] * JECcorr.evaluate(
                *JEC_input_t1_nomu
            )

            if isRealData:
                ## Only JEC/JES applied to data, no JER
                jets["pt"] = ak.unflatten(j["pt_JECnom"], nj)
                jets["mass"] = ak.unflatten(j["mass_JECnom"], nj)
                t1jets["pt"] = ak.unflatten(t1j["pt_JECnom"], nt1j)
                t1jets["pt_noMuL1"] = ak.unflatten(t1j["pt_noMuL1"], nt1j)
                t1jets["phi_noMuRaw"] = ak.unflatten(t1j["phi_noMuRaw"], nt1j)
                t1jets["pt_noMu_fullcorr"] = ak.unflatten(t1j["pt_noMu_JEC"], nt1j)
                met_pt, met_phi = calc_T1_MET(met_raw, met_nano, t1jets, campaign)
                met["pt"] = met_pt
                met["phi"] = met_phi
            else:
                ## apply nominal JER to nominal JES/JEC
                j["pt"] = j["pt_JECnom"]
                j["mass"] = j["mass_JECnom"]
                jer_smear_nom = get_JER(correct_map, jername, j, "nom", jet_algo)
                j["pt_JECnom_JERnom"] = j["pt_JECnom"] * jer_smear_nom
                j["mass_JECnom_JERnom"] = j["mass_JECnom"] * jer_smear_nom
                jets["pt"] = ak.unflatten(j["pt_JECnom_JERnom"], nj)
                jets["mass"] = ak.unflatten(j["mass_JECnom_JERnom"], nj)

                t1j["pt"] = t1j["pt_JECnom"]
                jer_smear_nom_t1 = get_JER(correct_map, jername, t1j, "nom", jet_algo)
                t1j["pt_JECnom_JERnom"] = t1j["pt_JECnom"] * jer_smear_nom_t1
                t1jets["pt"] = ak.unflatten(t1j["pt_JECnom_JERnom"], nt1j)
                t1jets["pt_noMuL1"] = ak.unflatten(t1j["pt_noMuL1"], nt1j)
                t1jets["phi_noMuRaw"] = ak.unflatten(t1j["phi_noMuRaw"], nt1j)
                t1jets["pt_noMu_fullcorr"] = ak.unflatten(
                    t1j["pt_noMu_JEC"] * jer_smear_nom_t1, nt1j
                )
                (
                    met_pt,
                    met_phi,
                    pt_unclustered_up,
                    pt_unclustered_down,
                    phi_unclustered_up,
                    phi_unclustered_down,
                ) = calc_T1_MET(
                    met_raw,
                    met_nano,
                    t1jets,
                    campaign,
                    True,
                )
                met["pt"] = met_pt
                met["phi"] = met_phi

                ## JES/JEC & JER systematics
                if systematic != False:
                    unc_jets, unc_met = {}, {}
                    for var in ["up", "down"]:
                        fac = 1.0 if var == "up" else -1.0

                        ## apply up/down JER to nominal JES/JEC
                        j["pt"] = j["pt_JECnom"]
                        j["mass"] = j["mass_JECnom"]
                        jer_smear_var = get_JER(correct_map, jername, j, var, jet_algo)
                        j[f"pt_JECnom_JER{var}"] = ak.values_astype(
                            j["pt_JECnom"] * jer_smear_var,
                            np.float32,
                        )
                        j[f"mass_JECnom_JER{var}"] = ak.values_astype(
                            j["mass_JECnom"] * jer_smear_var,
                            np.float32,
                        )
                        t1j["pt"] = t1j["pt_JECnom"]
                        jer_smear_var_t1 = get_JER(
                            correct_map, jername, t1j, var, jet_algo
                        )
                        t1j[f"pt_JECnom_JER{var}"] = ak.values_astype(
                            t1j["pt_JECnom"] * jer_smear_var_t1,
                            np.float32,
                        )
                        t1j["pt"] = t1j[f"pt_JECnom_JER{var}"]
                        _t1jets_var = ak.unflatten(t1j, nt1j)
                        _t1jets_var["pt_noMuL1"] = ak.unflatten(t1j["pt_noMuL1"], nt1j)
                        _t1jets_var["phi_noMuRaw"] = ak.unflatten(
                            t1j["phi_noMuRaw"], nt1j
                        )
                        _t1jets_var["pt_noMu_fullcorr"] = ak.unflatten(
                            t1j["pt_noMu_JEC"] * jer_smear_var_t1, nt1j
                        )

                        ## loop over JER uncertainties
                        if jer_split_id in jer_split.keys():
                            jer_bins = jer_split[jer_split_id]
                            for jer_bin_name in jer_bins.keys():
                                jer_bin_low = jer_bins[jer_bin_name][0]
                                jer_bin_high = jer_bins[jer_bin_name][1]

                                jer_shifted_pt = np.where(
                                    (abs(j["eta"]) >= jer_bin_low)
                                    & (abs(j["eta"]) < jer_bin_high),
                                    j[f"pt_JECnom_JER{var}"],
                                    j["pt_JECnom_JERnom"],
                                )
                                jer_shifted_mass = np.where(
                                    (abs(j["eta"]) >= jer_bin_low)
                                    & (abs(j["eta"]) < jer_bin_high),
                                    j[f"mass_JECnom_JER{var}"],
                                    j["mass_JECnom_JERnom"],
                                )
                                _t1jets_var["pt"] = np.where(
                                    (abs(_t1jets_var["eta"]) >= jer_bin_low)
                                    & (abs(_t1jets_var["eta"]) < jer_bin_high),
                                    _t1jets_var[f"pt_JECnom_JER{var}"],
                                    _t1jets_var["pt_JECnom_JERnom"],
                                )
                                _t1jets_var["pt"] = np.where(
                                    (abs(_t1jets_var["eta"]) >= jer_bin_low)
                                    & (abs(_t1jets_var["eta"]) < jer_bin_high),
                                    _t1jets_var["pt_noMu_fullcorr"],
                                    t1jets["pt_noMu_fullcorr"],
                                )
                                (
                                    met_pt_JECnom_JERvar,
                                    met_phi_JECnom_JERvar,
                                ) = calc_T1_MET(
                                    met_raw,
                                    met_nano,
                                    _t1jets_var,
                                    campaign,
                                )

                                unc_jets[f"JER{jer_bin_name}{var}"] = copy.copy(
                                    nocorrjet
                                )
                                unc_jets[f"JER{jer_bin_name}{var}"]["pt"] = (
                                    ak.values_astype(
                                        ak.unflatten(jer_shifted_pt, nj),
                                        np.float32,
                                    )
                                )
                                unc_jets[f"JER{jer_bin_name}{var}"]["mass"] = (
                                    ak.values_astype(
                                        ak.unflatten(jer_shifted_mass, nj),
                                        np.float32,
                                    )
                                )

                                unc_met[f"JER{jer_bin_name}{var}"] = copy.copy(
                                    nocorrmet
                                )
                                unc_met[f"JER{jer_bin_name}{var}"]["pt"] = (
                                    ak.values_astype(
                                        met_pt_JECnom_JERvar,
                                        np.float32,
                                    )
                                )
                                unc_met[f"JER{jer_bin_name}{var}"]["phi"] = (
                                    ak.values_astype(
                                        met_phi_JECnom_JERvar,
                                        np.float32,
                                    )
                                )

                        ## unclustered MET uncertainty
                        if var == "up":
                            shifted_met_pt = pt_unclustered_up
                            shifted_met_phi = phi_unclustered_up
                        else:
                            shifted_met_pt = pt_unclustered_down
                            shifted_met_phi = phi_unclustered_down

                        unc_met[f"MET_UnclusteredEnergy{var}"] = copy.copy(nocorrmet)
                        unc_met[f"MET_UnclusteredEnergy{var}"]["pt"] = ak.values_astype(
                            shifted_met_pt,
                            np.float32,
                        )
                        unc_met[f"MET_UnclusteredEnergy{var}"]["phi"] = (
                            ak.values_astype(
                                shifted_met_phi,
                                np.float32,
                            )
                        )

                        ## loop over JES/JEC uncertainties
                        if jes_sources_id in jes_sources.keys():
                            for jes_syst in jes_sources[jes_sources_id]:
                                jesuncmap = correct_map["JME"][
                                    f"{jecname}_{jes_syst}_{jet_algo}"
                                ]
                                jesunc = jesuncmap.evaluate(j.eta, j.pt_JECnom)
                                jesunc_t1 = jesuncmap.evaluate(t1j.eta, t1j.pt_JECnom)

                                ## apply shift up/down to JES/JEC
                                j[f"pt_JEC{jes_syst}{var}"] = ak.values_astype(
                                    j["pt_JECnom"] * (1 + fac * jesunc),
                                    np.float32,
                                )
                                j[f"mass_JEC{jes_syst}{var}"] = ak.values_astype(
                                    j["mass_JECnom"] * (1 + fac * jesunc),
                                    np.float32,
                                )
                                t1j[f"pt_JEC{jes_syst}{var}"] = ak.values_astype(
                                    t1j["pt_JECnom"] * (1 + fac * jesunc_t1),
                                    np.float32,
                                )

                                ## apply nominal JER to up/down JES/JEC. Reuse
                                ## the SAME jer_smear_nom/jer_smear_nom_t1
                                ## computed once above (not a fresh
                                ## get_JER(...,"nom",...) call here): for a
                                ## GenJet-unmatched jet the smearing formula's
                                ## stochastic draw is NOT reproducible across
                                ## repeated evaluate() calls with identical
                                ## (pt, eta) inputs -- it depends on call
                                ## order/position, not just the declared
                                ## inputs. Recomputing "nom" JER fresh inside
                                ## this loop (the previous behaviour) drew a
                                ## DIFFERENT random smear than the true
                                ## baseline for every unmatched jet, so every
                                ## JES-source template carried a spurious
                                ## random offset relative to nominal --
                                ## exactly reproducing bit-for-bit for
                                ## "up" and "down" of the same source (same
                                ## call position each time) while differing
                                ## from nominal (a different call position).
                                ## "nom" JER is physically independent of
                                ## which JES source is being varied, so the
                                ## single cached value is also the correct
                                ## one to use, not just the reproducible one.
                                j[f"pt_JEC{jes_syst}{var}_JERnom"] = ak.values_astype(
                                    j[f"pt_JEC{jes_syst}{var}"] * jer_smear_nom,
                                    np.float32,
                                )
                                j[f"mass_JEC{jes_syst}{var}_JERnom"] = ak.values_astype(
                                    j[f"mass_JEC{jes_syst}{var}"] * jer_smear_nom,
                                    np.float32,
                                )
                                t1j["pt"] = t1j[f"pt_JEC{jes_syst}{var}"]
                                t1j[f"pt_JEC{jes_syst}{var}_JERnom"] = ak.values_astype(
                                    t1j[f"pt_JEC{jes_syst}{var}"] * jer_smear_nom_t1,
                                    np.float32,
                                )
                                _t1jets_jesvar = ak.unflatten(t1j, nt1j)
                                _t1jets_jesvar["pt_noMuL1"] = ak.unflatten(
                                    t1j["pt_noMuL1"], nt1j
                                )
                                _t1jets_jesvar["phi_noMuRaw"] = ak.unflatten(
                                    t1j["phi_noMuRaw"], nt1j
                                )
                                # JES syst factor = (1 + fac*jesunc), applied to
                                # both the nominal JEC and the muon-subtracted JEC
                                _t1jets_jesvar["pt_noMu_fullcorr"] = ak.unflatten(
                                    t1j["pt_noMu_JEC"]
                                    * (1 + fac * jesunc_t1)
                                    * jer_smear_nom_t1,
                                    nt1j,
                                )
                                (
                                    met_pt_JECvar_JERnom,
                                    met_phi_JECvar_JERnom,
                                ) = calc_T1_MET(
                                    met_raw,
                                    met_nano,
                                    _t1jets_jesvar,
                                    campaign,
                                )

                                ## JES/JEC uncertainty
                                unc_jets[f"JES_{jes_syst}{var}"] = copy.copy(nocorrjet)
                                unc_jets[f"JES_{jes_syst}{var}"]["pt"] = (
                                    ak.values_astype(
                                        ak.unflatten(
                                            j[f"pt_JEC{jes_syst}{var}_JERnom"], nj
                                        ),
                                        np.float32,
                                    )
                                )
                                unc_jets[f"JES_{jes_syst}{var}"]["mass"] = (
                                    ak.values_astype(
                                        ak.unflatten(
                                            j[f"mass_JEC{jes_syst}{var}_JERnom"], nj
                                        ),
                                        np.float32,
                                    )
                                )

                                unc_met[f"JES_{jes_syst}{var}"] = copy.copy(nocorrmet)
                                unc_met[f"JES_{jes_syst}{var}"]["pt"] = (
                                    ak.values_astype(
                                        met_pt_JECvar_JERnom,
                                        np.float32,
                                    )
                                )
                                unc_met[f"JES_{jes_syst}{var}"]["phi"] = (
                                    ak.values_astype(
                                        met_phi_JECvar_JERnom,
                                        np.float32,
                                    )
                                )

                    if jer_split_id in jer_split.keys():
                        for jer_bin_name in jer_split[jer_split_id]:
                            jets[f"JER_{jer_bin_name}"] = ak.zip(
                                {
                                    "up": unc_jets[f"JER{jer_bin_name}up"],
                                    "down": unc_jets[f"JER{jer_bin_name}down"],
                                }
                            )
                            met[f"JER_{jer_bin_name}"] = ak.zip(
                                {
                                    "up": unc_met[f"JER{jer_bin_name}up"],
                                    "down": unc_met[f"JER{jer_bin_name}down"],
                                }
                            )
                    met["MET_UnclusteredEnergy"] = ak.zip(
                        {
                            "up": unc_met["MET_UnclusteredEnergyup"],
                            "down": unc_met["MET_UnclusteredEnergydown"],
                        }
                    )
                    if jes_sources_id in jes_sources.keys():
                        for jes_syst in jes_sources[jes_sources_id]:
                            jets[f"JES_{jes_syst}"] = ak.zip(
                                {
                                    "up": unc_jets[f"JES_{jes_syst}up"],
                                    "down": unc_jets[f"JES_{jes_syst}down"],
                                }
                            )
                            met[f"JES_{jes_syst}"] = ak.zip(
                                {
                                    "up": unc_met[f"JES_{jes_syst}up"],
                                    "down": unc_met[f"JES_{jes_syst}down"],
                                }
                            )

        else:
            if isRealData:
                if "2016preVFP-UL" == campaign:
                    if "2016B" in dataset or "2016C" in dataset or "2016D" in dataset:
                        jecname = "BCD"
                    elif "2016E" in dataset or "2016F" in dataset:
                        jecname = "EF"
                elif "2016postVFP-UL" == campaign:
                    jecname = "FGH"
                elif campaign == "Summer23":
                    if "v4" in dataset:
                        jecname = "Cv4"
                    else:
                        jecname = "Cv123"
                elif re.search(r"[Rr]un20\d{2}([A-Z])", dataset):
                    jecname = re.search(r"[Rr]un20\d{2}([A-Z])", dataset).group(1)
                else:
                    print("No valid jec name")
                    raise NameError
                jecname = "data" + jecname
            else:
                jecname = "MC"

            jets = correct_map["JME"]["jet_factory"][jecname].build(
                add_jec_variables(events.Jet, events.fixedGridRhoFastjetAll),
                lazy_cache=events.caches[0],
            )
            met = correct_map["JME"]["met_factory"].build(events.PuppiMET, jets, {})

        # Sort the jets by pt
        new_jet_idx = ak.argsort(jets.pt, ascending=False)
        jets = jets[new_jet_idx]

        # systematics
        if not isRealData and systematic != False:
            if "MET_UnclusteredEnergy" in met.fields:
                shifts += [
                    (
                        {
                            "Jet": jets,
                            "MET": met.MET_UnclusteredEnergy.up,
                        },
                        "UESUp",
                    ),
                    (
                        {
                            "Jet": jets,
                            "MET": met.MET_UnclusteredEnergy.down,
                        },
                        "UESDown",
                    ),
                ]
            if jer_split_id in jer_split.keys():
                for jer_bin_name in jer_split[jer_split_id]:
                    if f"JER_{jer_bin_name}" in jets.fields:
                        shifts += [
                            (
                                {
                                    "Jet": jets[f"JER_{jer_bin_name}"]["up"],
                                    "MET": met[f"JER_{jer_bin_name}"]["up"],
                                },
                                f"JER{jer_bin_name}Up",
                            ),
                            (
                                {
                                    "Jet": jets[f"JER_{jer_bin_name}"]["down"],
                                    "MET": met[f"JER_{jer_bin_name}"]["down"],
                                },
                                f"JER{jer_bin_name}Down",
                            ),
                        ]
            if jes_sources_id in jes_sources.keys():
                for jes_syst in jes_sources[jes_sources_id]:
                    if f"JES_{jes_syst}" in jets.fields:
                        shifts += [
                            (
                                {
                                    "Jet": jets[f"JES_{jes_syst}"]["up"],
                                    "MET": met[f"JES_{jes_syst}"]["up"],
                                },
                                f"JES{jes_syst}Up",
                            ),
                            (
                                {
                                    "Jet": jets[f"JES_{jes_syst}"]["down"],
                                    "MET": met[f"JES_{jes_syst}"]["down"],
                                },
                                f"JES{jes_syst}Down",
                            ),
                        ]

    else:
        met = events.PuppiMET
        jets = events.Jet

    shifts.insert(0, ({"Jet": jets, "MET": met}, None))
    return shifts


## Muon Rochester correction (Run 2)
def Roccor_shifts(shifts, correct_map, events, isRealData, systematic=False):
    """
    Apply Rochester corrections (Roccor) shifts to muons in events.

    This function applies Rochester corrections to the muons in the events based on the provided correction map and campaign.
    It handles both real data and simulated data, and can optionally apply systematic variations.

    Parameters:
    shifts (list): A list of shift types to apply (e.g., 'up', 'down').
    correct_map (dict): A dictionary containing correction factors and settings for Rochester corrections.
    events (awkward.Array): An array of events containing muon information.
    campaign (str): The name of the campaign for which to apply the corrections.
    isRealData (bool): A flag indicating whether the data is real or simulated.
    systematic (bool, optional): A flag to indicate whether to apply systematic variations. Default is False.

    Returns:
    awkward.Array: The events array with applied Rochester corrections.

    Raises:
    KeyError: If required keys are missing in the correct_map.
    ValueError: If the campaign is not recognized or supported.
    """

    mu = events.Muon
    if isRealData:
        SF = correct_map["roccor"].kScaleDT(
            events.Muon.charge, events.Muon.pt, events.Muon.eta, events.Muon.phi
        )

    else:
        hasgen = ~np.isnan(ak.fill_none(events.Muon.matched_gen.pt, np.nan))
        mc_kspread = correct_map["roccor"].kSpreadMC(
            events.Muon.charge[hasgen],
            events.Muon.pt[hasgen],
            events.Muon.eta[hasgen],
            events.Muon.phi[hasgen],
            events.Muon.matched_gen.pt[hasgen],
        )
        mc_rand = np.random.rand(len(ak.flatten(events.Muon.pt, axis=1)))
        mc_rand = ak.unflatten(mc_rand, ak.num(events.Muon.pt))
        mc_ksmear = correct_map["roccor"].kSmearMC(
            events.Muon.charge[~hasgen],
            events.Muon.pt[~hasgen],
            events.Muon.eta[~hasgen],
            events.Muon.phi[~hasgen],
            events.Muon.nTrackerLayers[~hasgen],
            mc_rand[~hasgen],
        )
        SF = np.array(ak.flatten(ak.ones_like(events.Muon.pt)))
        hasgen_flat = np.array(ak.flatten(hasgen))
        SF[hasgen_flat] = np.array(ak.flatten(mc_kspread))
        SF[~hasgen_flat] = np.array(ak.flatten(mc_ksmear))
        SF = ak.unflatten(SF, ak.num(events.Muon.pt))

    mu["pt"] = SF * events.Muon.pt
    # add rochester correction to shift
    for i in range(len(shifts)):
        shifts[i][0]["Muon"] = mu

    if systematic:
        if isRealData:
            err = correct_map["roccor"].kScaleDTerror(
                events.Muon.charge, events.Muon.pt, events.Muon.eta, events.Muon.phi
            )
        else:
            mc_errspread = correct_map["roccor"].kSpreadMCerror(
                events.Muon.charge[hasgen],
                events.Muon.pt[hasgen],
                events.Muon.eta[hasgen],
                events.Muon.phi[hasgen],
                events.Muon.matched_gen.pt[hasgen],
            )
            mc_errsmear = correct_map["roccor"].kSmearMCerror(
                events.Muon.charge[~hasgen],
                events.Muon.pt[~hasgen],
                events.Muon.eta[~hasgen],
                events.Muon.phi[~hasgen],
                events.Muon.nTrackerLayers[~hasgen],
                mc_rand[~hasgen],
            )
            err = np.array(ak.flatten(ak.ones_like(events.Muon.pt)))
            err[hasgen_flat] = np.array(ak.flatten(mc_errspread))
            err[~hasgen_flat] = np.array(ak.flatten(mc_errsmear))
            err = ak.unflatten(err, ak.num(events.Muon.pt))
        muup, mudown = events.Muon, events.Muon
        muup["pt"] = (SF + err) * events.Muon.pt
        mudown["pt"] = (SF - err) * events.Muon.pt
        shifts += [
            (
                {"Jet": shifts[0][0]["Jet"], "MET": shifts[0][0]["MET"], "Muon": muup},
                "RoccorUp",
            )
        ]
        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": shifts[0][0]["MET"],
                    "Muon": mudown,
                },
                "RoccorDown",
            )
        ]
    return shifts


def MUO_shifts(shifts, correct_map, events, isRealData, systematic=False):
    """
    Applies the Run 3 recommended muon scale and smearing corrections.
    Returns the corrected muon objects, including systematics.
    Adapted from this example of muon SS correction usage:
    https://gitlab.cern.ch/cms-analysis-corrections/MUO/examples/-/blob/latest/muoScaleAndSmearingCoffeaExample.py
    """

    mu = events.Muon

    if isRealData:
        mu_pt_corr = pt_scale(
            1,  # 1 for data, 0 for mc
            mu.pt,
            mu.eta,
            mu.phi,
            mu.charge,
            correct_map["muonSS"],
            nested=True,  # for awkward arrays, set False for 1d arrays
        )

    else:
        mu_pt_scalecorr = pt_scale(
            0, mu.pt, mu.eta, mu.phi, mu.charge, correct_map["muonSS"], nested=True
        )
        mu_pt_corr = pt_resol(
            mu_pt_scalecorr,
            mu.eta,
            mu.phi,
            mu.nTrackerLayers,
            events.event,
            events.luminosityBlock,
            correct_map["muonSS"],
            nested=True,
        )

    # scale and smearing uncertainties should be evaluated and applied on MC only
    if systematic and not isRealData:
        mu_pt_corr_scaleup = pt_scale_var(
            mu_pt_corr,
            mu.eta,
            mu.phi,
            mu.charge,
            "up",
            correct_map["muonSS"],
            nested=True,
        )
        mu_pt_corr_scaledown = pt_scale_var(
            mu_pt_corr,
            mu.eta,
            mu.phi,
            mu.charge,
            "dn",
            correct_map["muonSS"],
            nested=True,
        )
        mu_pt_corr_resolup = pt_resol_var(
            mu_pt_scalecorr,
            mu_pt_corr,
            mu.eta,
            "up",
            correct_map["muonSS"],
            nested=True,
        )
        mu_pt_corr_resoldown = pt_resol_var(
            mu_pt_scalecorr,
            mu_pt_corr,
            mu.eta,
            "dn",
            correct_map["muonSS"],
            nested=True,
        )

    mu["pt"] = mu_pt_corr
    # add nominal scale & smearing correction to shifts
    for i in range(len(shifts)):
        shifts[i][0]["Muon"] = mu

    if systematic:
        mu_up, mu_down = events.Muon, events.Muon
        mu_resol_up, mu_resol_down = events.Muon, events.Muon

        if not isRealData:
            mu_up["pt"] = mu_pt_corr_scaleup
            mu_down["pt"] = mu_pt_corr_scaledown
            mu_resol_up["pt"] = mu_pt_corr_resolup
            mu_resol_down["pt"] = mu_pt_corr_resoldown

        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": shifts[0][0]["MET"],
                    "Muon": mu_up,
                },
                "MuonScaleUp",
            )
        ]
        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": shifts[0][0]["MET"],
                    "Muon": mu_down,
                },
                "MuonScaleDown",
            )
        ]
        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": shifts[0][0]["MET"],
                    "Muon": mu_resol_up,
                },
                "MuonResolUp",
            )
        ]
        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": shifts[0][0]["MET"],
                    "Muon": mu_resol_down,
                },
                "MuonResolDown",
            )
        ]

    return shifts


def EGM_shifts(shifts, correct_map, events, isRealData, systematic=False):
    """
    Applies the Run 3 recommended electron scale and smearing corrections.
    Returns the corrected electron objects, including systematics.
    Adapted from this example of electron SS correction usage:
    https://gitlab.cern.ch/cms-analysis-corrections/EGM/examples/-/blob/latest/egmScaleAndSmearingExample.py
    """

    ele = events.Electron
    n_ele = ak.num(ele)
    events_run = ak.flatten(ak.broadcast_arrays(events.run, ele.eta)[0])
    ele_etaSC = (
        ak.flatten(ele.superclusterEta)
        if correct_map["campaign"]
        in [
            "Summer24",
            "CustomSummer24",
            "CustomSummer24UParTPaper",
            "CustomSummer25",
            "Winter25",
            "Prompt25",
        ]
        else ak.flatten(ele.eta + ele.deltaEtaSC)
    )
    ele_r9 = ak.flatten(ele.r9)
    ele_pt = ak.flatten(ele.pt)
    ele_seedGain = ak.flatten(ele.seedGain)

    if isRealData:  # scale correction is only applied to data
        scale_evaluator = correct_map["electronSS"].compound[
            correct_map["electronSS_cfg"][0]
        ]
        scale = scale_evaluator.evaluate(
            "scale",
            events_run,
            ele_etaSC,
            ele_r9,
            ele_pt,
            ele_seedGain,
        )
        scale = ak.unflatten(scale, n_ele)
        ele_pt_corr = scale * ele.pt
    else:  # smear correction is only applied to MC
        smear_and_syst_evaluator = correct_map["electronSS"][
            correct_map["electronSS_cfg"][1]
        ]
        smear = smear_and_syst_evaluator.evaluate(
            "smear", ele_pt, ele_r9, np.abs(ele_etaSC)
        )
        smear = ak.unflatten(smear, n_ele)
        # one independent random number per electron (per event would correlate the legs)
        rng = np.random.default_rng(seed=125)
        random_numbers = ak.unflatten(
            rng.normal(loc=0.0, scale=1.0, size=len(ele_pt)), n_ele
        )
        ele_pt_corr = ele.pt * (1 + smear * random_numbers)

    # scale and smearing uncertainties should be evaluated on the original MC only
    if systematic and not isRealData:
        unc_scale = smear_and_syst_evaluator.evaluate(
            "escale", ele_pt, ele_r9, np.abs(ele_etaSC)
        )
        unc_scale = ak.unflatten(unc_scale, n_ele)
        unc_smear = smear_and_syst_evaluator.evaluate(
            "esmear", ele_pt, ele_r9, np.abs(ele_etaSC)
        )
        unc_smear = ak.unflatten(unc_smear, n_ele)

    ele["pt"] = ele_pt_corr
    # add nominal scale & smearing correction to shifts
    for i in range(len(shifts)):
        shifts[i][0]["Electron"] = ele

    if systematic:
        ele_scale_up, ele_scale_down = events.Electron, events.Electron
        ele_smear_up, ele_smear_down = events.Electron, events.Electron

        if not isRealData:
            ele_scale_up["pt"] = (1 + unc_scale) * ele_pt_corr
            ele_scale_down["pt"] = (1 - unc_scale) * ele_pt_corr
            ele_smear_up["pt"] = events.Electron.pt * (
                1 + (smear + unc_smear) * random_numbers
            )
            ele_smear_down["pt"] = events.Electron.pt * (
                1 + np.maximum(0.0, (smear - unc_smear)) * random_numbers
            )

        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": shifts[0][0]["MET"],
                    "Muon": shifts[0][0]["Muon"],
                    "Electron": ele_scale_up,
                },
                "ElectronScaleUp",
            )
        ]
        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": shifts[0][0]["MET"],
                    "Muon": shifts[0][0]["Muon"],
                    "Electron": ele_scale_down,
                },
                "ElectronScaleDown",
            )
        ]
        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": shifts[0][0]["MET"],
                    "Muon": shifts[0][0]["Muon"],
                    "Electron": ele_smear_up,
                },
                "ElectronSmearUp",
            )
        ]
        shifts += [
            (
                {
                    "Jet": shifts[0][0]["Jet"],
                    "MET": shifts[0][0]["MET"],
                    "Muon": shifts[0][0]["Muon"],
                    "Electron": ele_smear_down,
                },
                "ElectronSmearDown",
            )
        ]

    return shifts


# wrapped up common shifts


def puwei(nPU, correct_map, weights, syst=False):
    """
    Return pileup weight
    Parameters
    ----------
    nPU: ak.Array
    correct_map : dict
    weights : coffea.analysis_tool.weights
    syst: "split", "weight_only"
    Apply pileup weights to events based on the number of primary vertices (nPU).

    This function applies pileup weights to the events using the provided correction map and weights.
    It can optionally apply systematic variations.

    Parameters:
    nPU (awkward.Array(int)): The number of primary vertices in the event.
    correct_map (dict): A dictionary containing correction factors and settings for pileup weights.
    weights (): A dictionary to store the calculated weights.
    syst (bool, optional): A flag to indicate whether to apply systematic variations. Default is False.

    Returns:
    None: The function modifies the weights dictionary in place.

    Raises:
    KeyError: If required keys are missing in the correct_map.
    ValueError: If the nPU value is not recognized or supported.
    """
    if "correctionlib" in str(type(correct_map["LUM"])):
        if syst:
            return weights.add(
                "puweight",
                correct_map["LUM"][list(correct_map["LUM"].keys())[0]].evaluate(
                    nPU, "nominal"
                ),
                correct_map["LUM"][list(correct_map["LUM"].keys())[0]].evaluate(
                    nPU, "up"
                ),
                correct_map["LUM"][list(correct_map["LUM"].keys())[0]].evaluate(
                    nPU, "down"
                ),
            )
        else:
            return weights.add(
                "puweight",
                correct_map["LUM"][list(correct_map["LUM"].keys())[0]].evaluate(
                    nPU, "nominal"
                ),
            )
    else:
        # Legacy ROOT histos may be keyed as PU/PUup/PUdown instead of LUM/PUup/PUdown
        if "LUM" in correct_map["LUM"]:
            central_key = "LUM"
        elif "PU" in correct_map["LUM"]:
            central_key = "PU"
        else:
            raise KeyError("Pileup central value not found in correct_map['LUM']")
        if "PUup" in correct_map["LUM"]:
            up_key = "PUup"
        elif "LUMup" in correct_map["LUM"]:
            up_key = "LUMup"
        else:
            raise KeyError("Pileup up-variation not found in correct_map['LUM']")

        if "PUdown" in correct_map["LUM"]:
            down_key = "PUdown"
        elif "LUMdown" in correct_map["LUM"]:
            down_key = "LUMdown"
        else:
            raise KeyError("Pileup down-variation not found in correct_map['LUM']")
        if syst:
            weights.add(
                "puweight",
                correct_map["LUM"][central_key](nPU),
                correct_map["LUM"][up_key](nPU),
                correct_map["LUM"][down_key](nPU),
            )
        else:
            weights.add("puweight", correct_map["LUM"][central_key](nPU))


def btagSFs(event, correct_map, weights, SFtype, syst=False):
    """
    Apply b-tagging scale factors (SFs) to a single jet.

    This function applies b-tagging scale factors to the given jet based on the provided correction map, weights, and scale factor type.
    It can optionally apply systematic variations.

    Parameters:
    event (dict): A dictionary containing the properties of the event.
    correct_map (dict): A dictionary containing correction factors and settings for b-tagging scale factors.x
    weights (coffea.weight.Weight): An instance of coffea's Weight class to store the calculated weights.
    SFtype (str): The type of scale factor to apply. Only shape-based C, B and 2D pseudocontinuous BC are supported.
    syst (bool, optional): A flag to indicate whether to apply systematic variations. Default is False.

    Returns:
    None: The function modifies the weights instance in place.

    Raises:
    KeyError: If required keys are missing in the correct_map.
    ValueError: If the SFtype is not recognized or supported.
    """
    jet = event.SelJet
    if SFtype.endswith("BC"):
        systlist = [
            "PUWeight",
            "JESRegrouped_Absolute",
            "JESRegrouped_Absolute_2024",
            "JESRegrouped_BBEC1",
            "JESRegrouped_BBEC1_2024",
            "JESRegrouped_EC2",
            "JESRegrouped_EC2_2024",
            "JESRegrouped_FlavorQCD",
            "JESRegrouped_HF",
            "JESRegrouped_HF_2024",
            "JESRegrouped_RelativeBal",
            "JESRegrouped_RelativeSample_2024",
            # "JESTotal",
            "JEReta0to1p93",
            "JEReta1p93to2p5",
            # "JERTotal",
            "MET_UnclEnergy",
            "Ele_Reco",
            "Ele_ID",
            "Ele_Trigger",
            "Ele_Scale",
            "Ele_Smear",
            "Mu_ID",
            "Mu_Iso",
            "Mu_Trigger",
            "Mu_Scale",
            "Mu_Resol",
            "TTWeight_ttbar",
            "LHEScaleWeight_PDF_ttbar",
            "LHEScaleWeight_PDF_singlet",
            "LHEScaleWeight_PDF_wjets",
            "LHEScaleWeight_PDF_zjets",
            "LHEScaleWeight_PDF_diboson",
            "LHEScaleWeight_aS_ttbar",
            "LHEScaleWeight_aS_singlet",
            "LHEScaleWeight_aS_wjets",
            "LHEScaleWeight_aS_zjets",
            "LHEScaleWeight_aS_diboson",
            "LHEScaleWeight_muF_ttbar",
            "LHEScaleWeight_muF_singlet",
            "LHEScaleWeight_muF_wjets",
            "LHEScaleWeight_muF_zjets",
            "LHEScaleWeight_muF_diboson",
            "LHEScaleWeight_muR_ttbar",
            "LHEScaleWeight_muR_singlet",
            "LHEScaleWeight_muR_wjets",
            "LHEScaleWeight_muR_zjets",
            "LHEScaleWeight_muR_diboson",
            "PSWeightFSR_ttbar",
            "PSWeightFSR_singlet",
            "PSWeightFSR_wjets",
            "PSWeightFSR_zjets",
            "PSWeightFSR_diboson",
            "PSWeightISR_ttbar",
            "PSWeightISR_singlet",
            "PSWeightISR_wjets",
            "PSWeightISR_zjets",
            "PSWeightISR_diboson",
            "XSec_ZJets_c",
            "XSec_ZJets_b",
            "XSec_WJets_c",
            "XSec_WJets_b",
            "XSec_ttbar",
            "XSec_singlet_tW",
            "XSec_singlet_tCh",
            "XSec_VV",
            "Lumi_13p6TeV_2024",
            # "Stat_flavB_B0_eta_0p00toinf_pt_25to35",
            # "Stat_flavB_B0_eta_0p00toinf_pt_35to50",
            # "Stat_flavB_B0_eta_0p00toinf_pt_50to70",
            # "Stat_flavB_B0_eta_0p00toinf_pt_70to90",
            # "Stat_flavB_B0_eta_0p00toinf_pt_90to120",
            # "Stat_flavB_B0_eta_0p00toinf_pt_120toinf",
            # "Stat_flavB_B1_eta_0p00toinf_pt_25to35",
            # "Stat_flavB_B1_eta_0p00toinf_pt_35to50",
            # "Stat_flavB_B1_eta_0p00toinf_pt_50to70",
            # "Stat_flavB_B1_eta_0p00toinf_pt_70to90",
            # "Stat_flavB_B1_eta_0p00toinf_pt_90to120",
            # "Stat_flavB_B1_eta_0p00toinf_pt_120toinf",
            # "Stat_flavB_B2_eta_0p00toinf_pt_25to35",
            # "Stat_flavB_B2_eta_0p00toinf_pt_35to50",
            # "Stat_flavB_B2_eta_0p00toinf_pt_50to70",
            # "Stat_flavB_B2_eta_0p00toinf_pt_70to90",
            # "Stat_flavB_B2_eta_0p00toinf_pt_90to120",
            # "Stat_flavB_B2_eta_0p00toinf_pt_120toinf",
            # "Stat_flavB_B3_eta_0p00toinf_pt_25to35",
            # "Stat_flavB_B3_eta_0p00toinf_pt_35to50",
            # "Stat_flavB_B3_eta_0p00toinf_pt_50to70",
            # "Stat_flavB_B3_eta_0p00toinf_pt_70to90",
            # "Stat_flavB_B3_eta_0p00toinf_pt_90to120",
            # "Stat_flavB_B3_eta_0p00toinf_pt_120toinf",
            # "Stat_flavB_B4_eta_0p00toinf_pt_25to35",
            # "Stat_flavB_B4_eta_0p00toinf_pt_35to50",
            # "Stat_flavB_B4_eta_0p00toinf_pt_50to70",
            # "Stat_flavB_B4_eta_0p00toinf_pt_70to90",
            # "Stat_flavB_B4_eta_0p00toinf_pt_90to120",
            # "Stat_flavB_B4_eta_0p00toinf_pt_120toinf",
            # "Stat_flavB_C0_eta_0p00toinf_pt_25to35",
            # "Stat_flavB_C0_eta_0p00toinf_pt_35to50",
            # "Stat_flavB_C0_eta_0p00toinf_pt_50to70",
            # "Stat_flavB_C0_eta_0p00toinf_pt_70to90",
            # "Stat_flavB_C0_eta_0p00toinf_pt_90to120",
            # "Stat_flavB_C0_eta_0p00toinf_pt_120toinf",
            # "Stat_flavB_C1_eta_0p00toinf_pt_25to35",
            # "Stat_flavB_C1_eta_0p00toinf_pt_35to50",
            # "Stat_flavB_C1_eta_0p00toinf_pt_50to70",
            # "Stat_flavB_C1_eta_0p00toinf_pt_70to90",
            # "Stat_flavB_C1_eta_0p00toinf_pt_90to120",
            # "Stat_flavB_C1_eta_0p00toinf_pt_120toinf",
            # "Stat_flavB_C2_eta_0p00toinf_pt_25to35",
            # "Stat_flavB_C2_eta_0p00toinf_pt_35to50",
            # "Stat_flavB_C2_eta_0p00toinf_pt_50to70",
            # "Stat_flavB_C2_eta_0p00toinf_pt_70to90",
            # "Stat_flavB_C2_eta_0p00toinf_pt_90to120",
            # "Stat_flavB_C2_eta_0p00toinf_pt_120toinf",
            # "Stat_flavB_C3_eta_0p00toinf_pt_25to35",
            # "Stat_flavB_C3_eta_0p00toinf_pt_35to50",
            # "Stat_flavB_C3_eta_0p00toinf_pt_50to70",
            # "Stat_flavB_C3_eta_0p00toinf_pt_70to90",
            # "Stat_flavB_C3_eta_0p00toinf_pt_90to120",
            # "Stat_flavB_C3_eta_0p00toinf_pt_120toinf",
            # "Stat_flavB_C4_eta_0p00toinf_pt_25to35",
            # "Stat_flavB_C4_eta_0p00toinf_pt_35to50",
            # "Stat_flavB_C4_eta_0p00toinf_pt_50to70",
            # "Stat_flavB_C4_eta_0p00toinf_pt_70to90",
            # "Stat_flavB_C4_eta_0p00toinf_pt_90to120",
            # "Stat_flavB_C4_eta_0p00toinf_pt_120toinf",
            # "Stat_flavC_B0_eta_0p00toinf_pt_25to35",
            # "Stat_flavC_B0_eta_0p00toinf_pt_35to50",
            # "Stat_flavC_B0_eta_0p00toinf_pt_50to70",
            # "Stat_flavC_B0_eta_0p00toinf_pt_70to90",
            # "Stat_flavC_B0_eta_0p00toinf_pt_90to120",
            # "Stat_flavC_B0_eta_0p00toinf_pt_120toinf",
            # "Stat_flavC_B1_eta_0p00toinf_pt_25to35",
            # "Stat_flavC_B1_eta_0p00toinf_pt_35to50",
            # "Stat_flavC_B1_eta_0p00toinf_pt_50to70",
            # "Stat_flavC_B1_eta_0p00toinf_pt_70to90",
            # "Stat_flavC_B1_eta_0p00toinf_pt_90to120",
            # "Stat_flavC_B1_eta_0p00toinf_pt_120toinf",
            # "Stat_flavC_B2_eta_0p00toinf_pt_25to35",
            # "Stat_flavC_B2_eta_0p00toinf_pt_35to50",
            # "Stat_flavC_B2_eta_0p00toinf_pt_50to70",
            # "Stat_flavC_B2_eta_0p00toinf_pt_70to90",
            # "Stat_flavC_B2_eta_0p00toinf_pt_90to120",
            # "Stat_flavC_B2_eta_0p00toinf_pt_120toinf",
            # "Stat_flavC_B3_eta_0p00toinf_pt_25to35",
            # "Stat_flavC_B3_eta_0p00toinf_pt_35to50",
            # "Stat_flavC_B3_eta_0p00toinf_pt_50to70",
            # "Stat_flavC_B3_eta_0p00toinf_pt_70to90",
            # "Stat_flavC_B3_eta_0p00toinf_pt_90to120",
            # "Stat_flavC_B3_eta_0p00toinf_pt_120toinf",
            # "Stat_flavC_B4_eta_0p00toinf_pt_25to35",
            # "Stat_flavC_B4_eta_0p00toinf_pt_35to50",
            # "Stat_flavC_B4_eta_0p00toinf_pt_50to70",
            # "Stat_flavC_B4_eta_0p00toinf_pt_70to90",
            # "Stat_flavC_B4_eta_0p00toinf_pt_90to120",
            # "Stat_flavC_B4_eta_0p00toinf_pt_120toinf",
            # "Stat_flavC_C0_eta_0p00toinf_pt_25to35",
            # "Stat_flavC_C0_eta_0p00toinf_pt_35to50",
            # "Stat_flavC_C0_eta_0p00toinf_pt_50to70",
            # "Stat_flavC_C0_eta_0p00toinf_pt_70to90",
            # "Stat_flavC_C0_eta_0p00toinf_pt_90to120",
            # "Stat_flavC_C0_eta_0p00toinf_pt_120toinf",
            # "Stat_flavC_C1_eta_0p00toinf_pt_25to35",
            # "Stat_flavC_C1_eta_0p00toinf_pt_35to50",
            # "Stat_flavC_C1_eta_0p00toinf_pt_50to70",
            # "Stat_flavC_C1_eta_0p00toinf_pt_70to90",
            # "Stat_flavC_C1_eta_0p00toinf_pt_90to120",
            # "Stat_flavC_C1_eta_0p00toinf_pt_120toinf",
            # "Stat_flavC_C2_eta_0p00toinf_pt_25to35",
            # "Stat_flavC_C2_eta_0p00toinf_pt_35to50",
            # "Stat_flavC_C2_eta_0p00toinf_pt_50to70",
            # "Stat_flavC_C2_eta_0p00toinf_pt_70to90",
            # "Stat_flavC_C2_eta_0p00toinf_pt_90to120",
            # "Stat_flavC_C2_eta_0p00toinf_pt_120toinf",
            # "Stat_flavC_C3_eta_0p00toinf_pt_25to35",
            # "Stat_flavC_C3_eta_0p00toinf_pt_35to50",
            # "Stat_flavC_C3_eta_0p00toinf_pt_50to70",
            # "Stat_flavC_C3_eta_0p00toinf_pt_70to90",
            # "Stat_flavC_C3_eta_0p00toinf_pt_90to120",
            # "Stat_flavC_C3_eta_0p00toinf_pt_120toinf",
            # "Stat_flavC_C4_eta_0p00toinf_pt_25to35",
            # "Stat_flavC_C4_eta_0p00toinf_pt_35to50",
            # "Stat_flavC_C4_eta_0p00toinf_pt_50to70",
            # "Stat_flavC_C4_eta_0p00toinf_pt_70to90",
            # "Stat_flavC_C4_eta_0p00toinf_pt_90to120",
            # "Stat_flavC_C4_eta_0p00toinf_pt_120toinf",
            # "Stat_flavL_B0_eta_0p00toinf_pt_25to35",
            # "Stat_flavL_B0_eta_0p00toinf_pt_35to50",
            # "Stat_flavL_B0_eta_0p00toinf_pt_50to70",
            # "Stat_flavL_B0_eta_0p00toinf_pt_70to90",
            # "Stat_flavL_B0_eta_0p00toinf_pt_90to120",
            # "Stat_flavL_B0_eta_0p00toinf_pt_120toinf",
            # "Stat_flavL_B1_eta_0p00toinf_pt_25to35",
            # "Stat_flavL_B1_eta_0p00toinf_pt_35to50",
            # "Stat_flavL_B1_eta_0p00toinf_pt_50to70",
            # "Stat_flavL_B1_eta_0p00toinf_pt_70to90",
            # "Stat_flavL_B1_eta_0p00toinf_pt_90to120",
            # "Stat_flavL_B1_eta_0p00toinf_pt_120toinf",
            # "Stat_flavL_B2_eta_0p00toinf_pt_25to35",
            # "Stat_flavL_B2_eta_0p00toinf_pt_35to50",
            # "Stat_flavL_B2_eta_0p00toinf_pt_50to70",
            # "Stat_flavL_B2_eta_0p00toinf_pt_70to90",
            # "Stat_flavL_B2_eta_0p00toinf_pt_90to120",
            # "Stat_flavL_B2_eta_0p00toinf_pt_120toinf",
            # "Stat_flavL_B3_eta_0p00toinf_pt_25to35",
            # "Stat_flavL_B3_eta_0p00toinf_pt_35to50",
            # "Stat_flavL_B3_eta_0p00toinf_pt_50to70",
            # "Stat_flavL_B3_eta_0p00toinf_pt_70to90",
            # "Stat_flavL_B3_eta_0p00toinf_pt_90to120",
            # "Stat_flavL_B3_eta_0p00toinf_pt_120toinf",
            # "Stat_flavL_B4_eta_0p00toinf_pt_25to35",
            # "Stat_flavL_B4_eta_0p00toinf_pt_35to50",
            # "Stat_flavL_B4_eta_0p00toinf_pt_50to70",
            # "Stat_flavL_B4_eta_0p00toinf_pt_70to90",
            # "Stat_flavL_B4_eta_0p00toinf_pt_90to120",
            # "Stat_flavL_B4_eta_0p00toinf_pt_120toinf",
            # "Stat_flavL_C0_eta_0p00toinf_pt_25to35",
            # "Stat_flavL_C0_eta_0p00toinf_pt_35to50",
            # "Stat_flavL_C0_eta_0p00toinf_pt_50to70",
            # "Stat_flavL_C0_eta_0p00toinf_pt_70to90",
            # "Stat_flavL_C0_eta_0p00toinf_pt_90to120",
            # "Stat_flavL_C0_eta_0p00toinf_pt_120toinf",
            # "Stat_flavL_C1_eta_0p00toinf_pt_25to35",
            # "Stat_flavL_C1_eta_0p00toinf_pt_35to50",
            # "Stat_flavL_C1_eta_0p00toinf_pt_50to70",
            # "Stat_flavL_C1_eta_0p00toinf_pt_70to90",
            # "Stat_flavL_C1_eta_0p00toinf_pt_90to120",
            # "Stat_flavL_C1_eta_0p00toinf_pt_120toinf",
            # "Stat_flavL_C2_eta_0p00toinf_pt_25to35",
            # "Stat_flavL_C2_eta_0p00toinf_pt_35to50",
            # "Stat_flavL_C2_eta_0p00toinf_pt_50to70",
            # "Stat_flavL_C2_eta_0p00toinf_pt_70to90",
            # "Stat_flavL_C2_eta_0p00toinf_pt_90to120",
            # "Stat_flavL_C2_eta_0p00toinf_pt_120toinf",
            # "Stat_flavL_C3_eta_0p00toinf_pt_25to35",
            # "Stat_flavL_C3_eta_0p00toinf_pt_35to50",
            # "Stat_flavL_C3_eta_0p00toinf_pt_50to70",
            # "Stat_flavL_C3_eta_0p00toinf_pt_70to90",
            # "Stat_flavL_C3_eta_0p00toinf_pt_90to120",
            # "Stat_flavL_C3_eta_0p00toinf_pt_120toinf",
            # "Stat_flavL_C4_eta_0p00toinf_pt_25to35",
            # "Stat_flavL_C4_eta_0p00toinf_pt_35to50",
            # "Stat_flavL_C4_eta_0p00toinf_pt_50to70",
            # "Stat_flavL_C4_eta_0p00toinf_pt_70to90",
            # "Stat_flavL_C4_eta_0p00toinf_pt_90to120",
            # "Stat_flavL_C4_eta_0p00toinf_pt_120toinf",
            "Stat_flavB_B0",
            "Stat_flavB_B1",
            "Stat_flavB_B2",
            "Stat_flavB_B3",
            "Stat_flavB_B4",
            "Stat_flavB_C0",
            "Stat_flavB_C1",
            "Stat_flavB_C2",
            "Stat_flavB_C3",
            "Stat_flavB_C4",
            "Stat_flavB_L0",
            "Stat_flavC_B0",
            "Stat_flavC_B1",
            "Stat_flavC_B2",
            "Stat_flavC_B3",
            "Stat_flavC_B4",
            "Stat_flavC_C0",
            "Stat_flavC_C1",
            "Stat_flavC_C2",
            "Stat_flavC_C3",
            "Stat_flavC_C4",
            "Stat_flavC_L0",
            "Stat_flavL_B0",
            "Stat_flavL_B1",
            "Stat_flavL_B2",
            "Stat_flavL_B3",
            "Stat_flavL_B4",
            "Stat_flavL_C0",
            "Stat_flavL_C1",
            "Stat_flavL_C2",
            "Stat_flavL_C3",
            "Stat_flavL_C4",
            "Stat_flavL_L0",
            "Stat",
            "Total",
        ]
    elif SFtype.endswith("C"):
        systlist = [
            "Extrap",
            "Interp",
            "LHEScaleWeight_muF",
            "LHEScaleWeight_muR",
            "PSWeightFSR",
            "PSWeightISR",
            "PUWeight",
            "Stat",
            "XSec_BRUnc_DYJets_b",
            "XSec_BRUnc_DYJets_c",
            "XSec_BRUnc_WJets_c",
            "jer",
            "jesTotal",
        ]
    elif SFtype.endswith("B"):
        systlist = [
            "hf",
            "lf",
            "cferr1",
            "cferr2",
            "hfstat1",
            "hfstat2",
            "lfstats1",
            "lfstats2",
        ]
    # SelJet is jagged/unpadded (jet counts vary event-to-event within a chunk), so the SF
    # is evaluated once on the fully-flattened jet array per systematic variation (not per
    # jet "slot" bounded by the first event's jet count) and reduced back to a per-event
    # value via unflatten + product across that event's jets.
    #
    # masknone/fill_none note: the None-handling in the pre-flatten version of this function
    # existed to guard against the artificial None padding produced by per-slot slicing
    # (`alljet[:, nj]`) past an event's real jet count -- that padding no longer exists once
    # the jets are flattened, since every entry of flat_jet is a real jet (SelJet is built by
    # direct boolean/index selection, not ak.pad_none, and btagSFs is only ever called when
    # "hadronFlavour" is present, i.e. MC jets, where these fields are always filled). The
    # ak.fill_none calls are kept (harmless/cheap) as a defensive no-op; the np.where(masknone, ...)
    # wrapper is dropped since it was purely compensating for slot-padding.
    alljet = jet if jet.ndim > 1 else ak.singletons(jet)
    counts = ak.num(alljet.pt)
    flat_jet = ak.flatten(alljet)
    flat_jet.hadronFlavour = ak.fill_none(flat_jet.hadronFlavour, 0)

    def evaluate_flat(syst_key):
        jet = flat_jet
        if "ctag" in correct_map.keys() and "correctionlib" in str(
            type(correct_map["ctag"])
        ):
            if SFtype == "DeepJetC":
                jet.btagDeepFlavCvL = ak.fill_none(jet.btagDeepFlavCvL, 0.0)
                jet.btagDeepFlavCvB = ak.fill_none(jet.btagDeepFlavCvB, 0.0)
                tmp_sfs = correct_map["ctag"]["deepJet_shape"].evaluate(
                    syst_key,
                    jet.hadronFlavour,
                    jet.btagDeepFlavCvL,
                    jet.btagDeepFlavCvB,
                )
            elif SFtype == "DeepCSVC":
                jet.btagDeepCvL = ak.fill_none(jet.btagDeepCvL, 0.0)
                jet.btagDeepCvB = ak.fill_none(jet.btagDeepCvB, 0.0)
                tmp_sfs = correct_map["ctag"]["deepCSV_shape"].evaluate(
                    syst_key,
                    jet.hadronFlavour,
                    jet.btagDeepCvL,
                    jet.btagDeepCvB,
                )
            elif SFtype == "UParTAK4BC" or SFtype == "PNetBC":
                if SFtype == "UParTAK4BC":
                    jet_2Dbin = ak.fill_none(jet.btagUParTAK42Dbin, 0)
                elif SFtype == "PNetBC":
                    jet_2Dbin = ak.fill_none(jet.btagPNet2Dbin, 0)
                jet_eta = ak.fill_none(jet.eta, 0)
                jet_pt = ak.fill_none(jet.pt, 0)
                if SFtype == "PNetBC":
                    jet_2Dbin = ak.where(jet_2Dbin == -1, 0, jet_2Dbin)
                tmp_sfs = correct_map["ctag"]["UParTAK4_pseudocontinuous"].evaluate(
                    syst_key,
                    jet.hadronFlavour,
                    jet.btagUParTAK42Dbin,
                    jet_eta,
                    jet_pt,
                )
        elif "btag" in correct_map.keys() and "correctionlib" in str(
            type(correct_map["btag"])
        ):
            if SFtype == "DeepJetB":
                jet.btagDeepFlavCvL = ak.fill_none(jet.btagDeepFlavCvL, 0.0)
                jet.btagDeepFlavCvB = ak.fill_none(jet.btagDeepFlavCvB, 0.0)
                tmp_sfs = correct_map["btag"]["deepJet_shape"].evaluate(
                    syst_key,
                    jet.hadronFlavour,
                    jet.btagDeepFlavCvL,
                    jet.btagDeepFlavCvB,
                )
            elif SFtype == "DeepCSVB":
                jet.btagDeepCvL = ak.fill_none(jet.btagDeepCvL, 0.0)
                jet.btagDeepCvB = ak.fill_none(jet.btagDeepCvB, 0.0)
                tmp_sfs = correct_map["btag"]["deepCSV_shape"].evaluate(
                    syst_key,
                    jet.hadronFlavour,
                    jet.btagDeepCvL,
                    jet.btagDeepCvB,
                )
        return ak.to_numpy(ak.prod(ak.unflatten(tmp_sfs, counts), axis=1))

    # "central" does not depend on the systematic, so it is evaluated once here instead of
    # being recomputed (identically) on every one of the ~100 systlist iterations.
    central = evaluate_flat("central")
    if syst == False:
        weights.add(SFtype, central)
        return weights

    sfs_up_all, sfs_down_all = {}, {}
    for sys in systlist:
        sfs_up_all[sys] = evaluate_flat(f"up_{sys}")
        sfs_down_all[sys] = evaluate_flat(f"down_{sys}")

    weights.add_multivariation(
        SFtype,
        central,
        systlist,
        np.array(list(sfs_up_all.values())),
        np.array(list(sfs_down_all.values())),
    )
    return weights


def eleSFs(ele, correct_map, weights, syst=True, isHLT=False):
    allele = ele if ele.ndim > 1 else ak.singletons(ele)

    for sf in correct_map["EGM_cfg"].keys():
        ## Only apply SFs for lepton pass HLT filter
        if not isHLT and "Trig" in sf:
            continue
        sf_type = sf[: sf.find(" ")]
        for nele in range(ak.num(allele.pt)[0]):
            ele = allele[:, nele]
            ele_etaSC = ak.fill_none(ele.eta + ele.deltaEtaSC, -2.5)
            masknone = ak.is_none(ele.pt)
            sfs_alle, sfs_alle_up, sfs_alle_down = (
                np.ones_like(allele[:, 0].pt),
                np.ones_like(allele[:, 0].pt),
                np.ones_like(allele[:, 0].pt),
            )

            if "correctionlib" in str(type(correct_map["EGM"])):
                ## reco SFs, split by pT
                if "Reco" in sf:
                    ele_pt = np.clip(ele.pt, 20, 74.99999)
                    ele_pt_low = np.where(ele.pt >= 20, 19.99999, ele.pt)
                    ele_pt_high = np.clip(ele.pt, 75, 10000)
                    ## phi is used in Summer23
                    if "Summer23" in correct_map["campaign"]:
                        sfs_low = np.where(
                            (ele.pt < 20.0) & ~masknone,
                            correct_map["EGM"][sf.split(" ")[2]].evaluate(
                                sf.split(" ")[1],
                                "sf",
                                "RecoBelow20",
                                ele_etaSC,
                                ele_pt_low,
                                ele.phi,
                            ),
                            1.0,
                        )
                        sfs_high = np.where(
                            (ele.pt >= 75.0) & ~masknone,
                            correct_map["EGM"][sf.split(" ")[2]].evaluate(
                                sf.split(" ")[1],
                                "sf",
                                "RecoAbove75",
                                ele_etaSC,
                                ele_pt_high,
                                ele.phi,
                            ),
                            sfs_low,
                        )
                        sfs = np.where(
                            (ele.pt >= 20.0) & (ele.pt < 75.0) & ~masknone,
                            correct_map["EGM"][sf.split(" ")[2]].evaluate(
                                sf.split(" ")[1],
                                "sf",
                                "Reco20to75",
                                ele_etaSC,
                                ele_pt,
                                ele.phi,
                            ),
                            sfs_high,
                        )
                        sfs = np.where(masknone, 1.0, sfs)

                        if syst != False:
                            sfs_up_low = np.where(
                                (ele.pt < 20.0) & ~masknone,
                                correct_map["EGM"][sf.split(" ")[2]].evaluate(
                                    sf.split(" ")[1],
                                    "sfup",
                                    "RecoBelow20",
                                    ele_etaSC,
                                    ele_pt_low,
                                    ele.phi,
                                ),
                                0.0,
                            )
                            sfs_down_low = np.where(
                                (ele.pt < 20.0) & ~masknone,
                                correct_map["EGM"][sf.split(" ")[2]].evaluate(
                                    sf.split(" ")[1],
                                    "sfdown",
                                    "RecoBelow20",
                                    ele_etaSC,
                                    ele_pt_low,
                                    ele.phi,
                                ),
                                0.0,
                            )
                            sfs_up_high = np.where(
                                (ele.pt >= 75.0) & ~masknone,
                                correct_map["EGM"][sf.split(" ")[2]].evaluate(
                                    sf.split(" ")[1],
                                    "sfup",
                                    "RecoAbove75",
                                    ele_etaSC,
                                    ele_pt_high,
                                    ele.phi,
                                ),
                                sfs_up_low,
                            )
                            sfs_down_high = np.where(
                                (ele.pt >= 75.0) & ~masknone,
                                correct_map["EGM"][sf.split(" ")[2]].evaluate(
                                    sf.split(" ")[1],
                                    "sfdown",
                                    "RecoAbove75",
                                    ele_etaSC,
                                    ele_pt_high,
                                    ele.phi,
                                ),
                                sfs_down_low,
                            )
                            sfs_up = np.where(
                                (ele.pt >= 20.0) & (ele.pt < 75.0) & ~masknone,
                                correct_map["EGM"][sf.split(" ")[2]].evaluate(
                                    sf.split(" ")[1],
                                    "sfup",
                                    "Reco20to75",
                                    ele_etaSC,
                                    ele_pt,
                                    ele.phi,
                                ),
                                sfs_up_high,
                            )
                            sfs_down = np.where(
                                (ele.pt >= 20.0) & (ele.pt < 75.0) & ~masknone,
                                correct_map["EGM"][sf.split(" ")[2]].evaluate(
                                    sf.split(" ")[1],
                                    "sfdown",
                                    "Reco20to75",
                                    ele_etaSC,
                                    ele_pt,
                                    ele.phi,
                                ),
                                sfs_down_high,
                            )
                            sfs_up = np.where(masknone, 1.0, sfs_up)
                            sfs_down = np.where(masknone, 1.0, sfs_down)

                    elif sf.split(" ")[2].startswith("UL-"):
                        # UL uses a two-bin reco scheme (RecoBelow20 / RecoAbove20)
                        sf_id, sf_campaign = sf.split(" ")[2], sf.split(" ")[1]
                        sfs_low = np.where(
                            (ele.pt < 20.0) & ~masknone,
                            correct_map["EGM"][sf_id].evaluate(
                                sf_campaign, "sf", "RecoBelow20", ele_etaSC, ele_pt_low
                            ),
                            1.0,
                        )
                        sfs = np.where(
                            (ele.pt >= 20.0) & ~masknone,
                            correct_map["EGM"][sf_id].evaluate(
                                sf_campaign, "sf", "RecoAbove20", ele_etaSC, ele_pt
                            ),
                            sfs_low,
                        )
                        sfs = np.where(masknone, 1.0, sfs)

                        if syst:
                            sfs_up_low = np.where(
                                (ele.pt < 20.0) & ~masknone,
                                correct_map["EGM"][sf_id].evaluate(
                                    sf_campaign, "sfup", "RecoBelow20", ele_etaSC, ele_pt_low
                                ),
                                0.0,
                            )
                            sfs_down_low = np.where(
                                (ele.pt < 20.0) & ~masknone,
                                correct_map["EGM"][sf_id].evaluate(
                                    sf_campaign, "sfdown", "RecoBelow20", ele_etaSC, ele_pt_low
                                ),
                                0.0,
                            )
                            sfs_up = np.where(
                                (ele.pt >= 20.0) & ~masknone,
                                correct_map["EGM"][sf_id].evaluate(
                                    sf_campaign, "sfup", "RecoAbove20", ele_etaSC, ele_pt
                                ),
                                sfs_up_low,
                            )
                            sfs_down = np.where(
                                (ele.pt >= 20.0) & ~masknone,
                                correct_map["EGM"][sf_id].evaluate(
                                    sf_campaign, "sfdown", "RecoAbove20", ele_etaSC, ele_pt
                                ),
                                sfs_down_low,
                            )
                            sfs_up = np.where(masknone, 1.0, sfs_up)
                            sfs_down = np.where(masknone, 1.0, sfs_down)
                    else:
                        if correct_map["campaign"] in ["Prompt25", "CustomSummer25"]:
                            sfs_low = np.ones_like(ele_pt_low)
                        else:
                            sfs_low = np.where(
                                (ele.pt < 20.0) & ~masknone,
                                correct_map["EGM"][sf.split(" ")[2]].evaluate(
                                    sf.split(" ")[1],
                                    "sf",
                                    "RecoBelow20",
                                    ele_etaSC,
                                    ele_pt_low,
                                ),
                                1.0,
                            )
                        sfs_high = np.where(
                            (ele.pt >= 75.0) & ~masknone,
                            correct_map["EGM"][sf.split(" ")[2]].evaluate(
                                sf.split(" ")[1],
                                "sf",
                                "RecoAbove75",
                                ele_etaSC,
                                ele_pt_high,
                            ),
                            sfs_low,
                        )
                        sfs = np.where(
                            (ele.pt >= 20.0) & (ele.pt < 75.0) & ~masknone,
                            correct_map["EGM"][sf.split(" ")[2]].evaluate(
                                sf.split(" ")[1], "sf", "Reco20to75", ele_etaSC, ele_pt
                            ),
                            sfs_high,
                        )
                        sfs = np.where(masknone, 1.0, sfs)

                        if syst:
                            if correct_map["campaign"] in [
                                "Prompt25",
                                "CustomSummer25",
                            ]:
                                sfs_up_low = np.ones_like(ele_pt_low)
                                sfs_down_low = np.ones_like(ele_pt_low)
                            else:
                                sfs_up_low = np.where(
                                    (ele.pt < 20.0) & ~masknone,
                                    correct_map["EGM"][sf.split(" ")[2]].evaluate(
                                        sf.split(" ")[1],
                                        "sfup",
                                        "RecoBelow20",
                                        ele_etaSC,
                                        ele_pt_low,
                                    ),
                                    0.0,
                                )
                                sfs_down_low = np.where(
                                    (ele.pt < 20.0) & ~masknone,
                                    correct_map["EGM"][sf.split(" ")[2]].evaluate(
                                        sf.split(" ")[1],
                                        "sfdown",
                                        "RecoBelow20",
                                        ele_etaSC,
                                        ele_pt_low,
                                    ),
                                    0.0,
                                )
                            sfs_up_high = np.where(
                                (ele.pt >= 75.0) & ~masknone,
                                correct_map["EGM"][sf.split(" ")[2]].evaluate(
                                    sf.split(" ")[1],
                                    "sfup",
                                    "RecoAbove75",
                                    ele_etaSC,
                                    ele_pt_high,
                                ),
                                sfs_up_low,
                            )
                            sfs_down_high = np.where(
                                (ele.pt >= 75.0) & ~masknone,
                                correct_map["EGM"][sf.split(" ")[2]].evaluate(
                                    sf.split(" ")[1],
                                    "sfdown",
                                    "RecoAbove75",
                                    ele_etaSC,
                                    ele_pt_high,
                                ),
                                sfs_down_low,
                            )
                            sfs_up = np.where(
                                (ele.pt >= 20.0) & (ele.pt < 75.0) & ~masknone,
                                correct_map["EGM"][sf.split(" ")[2]].evaluate(
                                    sf.split(" ")[1],
                                    "sfup",
                                    "Reco20to75",
                                    ele_etaSC,
                                    ele_pt,
                                ),
                                sfs_up_high,
                            )
                            sfs_down = np.where(
                                (ele.pt >= 20.0) & (ele.pt < 75.0) & ~masknone,
                                correct_map["EGM"][sf.split(" ")[2]].evaluate(
                                    sf.split(" ")[1],
                                    "sfdown",
                                    "Reco20to75",
                                    ele_etaSC,
                                    ele_pt,
                                ),
                                sfs_down_high,
                            )
                            sfs_up = np.where(masknone, 1.0, sfs_up)
                            sfs_down = np.where(masknone, 1.0, sfs_down)

                else:
                    # trigger SFs
                    if "Trig" in sf and "correctionlib" in str(
                        type(correct_map["EGM_HLT"])
                    ):
                        _ele_map = "EGM_HLT"
                        ele_pt = ak.fill_none(ele.pt, 25.0)
                        ele_pt = np.clip(ele_pt, 25.0, None)
                    # ID SFs
                    else:
                        _ele_map = "EGM"
                        ele_pt = ak.fill_none(ele.pt, 10.0)
                        ele_pt = np.clip(ele_pt, 10.0, None)

                    if "Summer23" in correct_map["campaign"] and "ID" in sf:
                        sfs = np.where(
                            masknone,
                            1.0,
                            correct_map[_ele_map][sf.split(" ")[2]].evaluate(
                                sf.split(" ")[1],
                                "sf",
                                correct_map["EGM_cfg"][sf],
                                ele_etaSC,
                                ele_pt,
                                ele.phi,
                            ),
                        )

                        if syst:
                            sfs_up = np.where(
                                masknone,
                                1.0,
                                correct_map[_ele_map][sf.split(" ")[2]].evaluate(
                                    sf.split(" ")[1],
                                    "sfup",
                                    correct_map["EGM_cfg"][sf],
                                    ele_etaSC,
                                    ele_pt,
                                    ele.phi,
                                ),
                            )
                            sfs_down = np.where(
                                masknone,
                                1.0,
                                correct_map[_ele_map][sf.split(" ")[2]].evaluate(
                                    sf.split(" ")[1],
                                    "sfdown",
                                    correct_map["EGM_cfg"][sf],
                                    ele_etaSC,
                                    ele_pt,
                                    ele.phi,
                                ),
                            )
                    else:
                        sfs = np.where(
                            masknone,
                            1.0,
                            correct_map[_ele_map][sf.split(" ")[2]].evaluate(
                                sf.split(" ")[1],
                                "sf",
                                correct_map["EGM_cfg"][sf],
                                ele_etaSC,
                                ele_pt,
                            ),
                        )

                        if syst:
                            sfs_up = np.where(
                                masknone,
                                1.0,
                                correct_map[_ele_map][sf.split(" ")[2]].evaluate(
                                    sf.split(" ")[1],
                                    "sfup",
                                    correct_map["EGM_cfg"][sf],
                                    ele_etaSC,
                                    ele_pt,
                                ),
                            )
                            sfs_down = np.where(
                                masknone,
                                1.0,
                                correct_map[_ele_map][sf.split(" ")[2]].evaluate(
                                    sf.split(" ")[1],
                                    "sfdown",
                                    correct_map["EGM_cfg"][sf],
                                    ele_etaSC,
                                    ele_pt,
                                ),
                            )

            else:
                if "ele_Trig" in sf:
                    sfs = np.where(
                        masknone, 1.0, correct_map["EGM_custom"][sf_type](ele_pt)
                    )
                    if syst:
                        sfs_up = np.where(
                            masknone,
                            1.0,
                            correct_map["EGM_custom"][sf_type](ele_pt)
                            + correct_map["EGM_custom"][f"{sf_type}_error"](ele_pt),
                        )
                        sfs_down = np.where(
                            masknone,
                            1.0,
                            correct_map["EGM_custom"][sf_type](ele_pt)
                            - correct_map["EGM_custom"][f"{sf_type}_error"](ele_pt),
                        )
                elif "ele" in sf:
                    sfs = np.where(
                        masknone,
                        1.0,
                        correct_map["EGM_custom"][sf_type](ele_etaSC, ele_pt),
                    )
                    if syst:
                        sfs_up = np.where(
                            masknone,
                            1.0,
                            correct_map["EGM_custom"][sf_type](ele_etaSC, ele_pt)
                            + correct_map["EGM_custom"][f"{sf_type}_error"](
                                ele_etaSC, ele_pt
                            ),
                        )
                        sfs_down = np.where(
                            masknone,
                            1.0,
                            correct_map["EGM_custom"][sf_type](ele_etaSC, ele_pt)
                            - correct_map["EGM_custom"][f"{sf_type}_error"](
                                ele_etaSC, ele_pt
                            ),
                        )

            sfs_alle = sfs_alle * sfs
            if syst:
                sfs_alle_down = sfs_alle_down * sfs_down
                sfs_alle_up = sfs_alle_up * sfs_up

        sfname = sf.split(" ")[0]
        if syst:
            weights.add(sfname, sfs_alle, sfs_alle_up, sfs_alle_down)
        else:
            weights.add(sfname, sfs_alle)

    return weights


def muSFs(mu, correct_map, weights, syst=False, isHLT=False):
    allmu = mu if mu.ndim > 1 else ak.singletons(mu)
    for sf in correct_map["MUO_cfg"].keys():
        ## Only apply SFs for lepton pass HLT filter
        if not isHLT and "Trig" in sf:
            continue
        if "low" in sf:
            continue
        sfs_allmu, sfs_allmu_up, sfs_allmu_down = (
            np.ones_like(allmu[:, 0].pt),
            np.ones_like(allmu[:, 0].pt),
            np.ones_like(allmu[:, 0].pt),
        )
        sf_type = sf[: sf.find(" ")]
        for nmu in range(ak.num(allmu.pt)[0]):
            mu = allmu[:, nmu]
            masknone = ak.is_none(mu.pt)
            if "Trig" in sf:
                # lowest pT bin edge of the trigger SF: 29 GeV for IsoMu27, 26 GeV for IsoMu24
                trig_min = 29.0 if "IsoMu27" in correct_map["MUO_cfg"][sf] else 26.0
                mu_pt = np.clip(mu.pt, trig_min, None)
            else:
                pt_min = (
                    10.0
                    if correct_map["campaign"]
                    in [
                        "Summer24",
                        "CustomSummer24",
                        "CustomSummer24UParTPaper",
                        "CustomSummer25",
                        "Winter25",
                        "Prompt25",
                    ]
                    else 15.0
                )
                mu_pt = np.clip(mu.pt, pt_min, None)
            mu_eta = np.clip(mu.eta, -2.4, 2.399999)
            sfs = 1.0
            if "correctionlib" in str(type(correct_map["MUO"])):
                sfs = np.where(
                    masknone,
                    1.0,
                    correct_map["MUO"][correct_map["MUO_cfg"][sf]].evaluate(
                        mu_eta, mu_pt, "nominal"
                    ),
                )

                if syst:
                    sfs_up = np.where(
                        masknone,
                        1.0,
                        correct_map["MUO"][correct_map["MUO_cfg"][sf]].evaluate(
                            mu_eta, mu_pt, "systup"
                        ),
                    )
                    sfs_down = np.where(
                        masknone,
                        1.0,
                        correct_map["MUO"][correct_map["MUO_cfg"][sf]].evaluate(
                            mu_eta, mu_pt, "systdown"
                        ),
                    )
            else:
                if "mu" in sf:
                    sfs = np.where(
                        masknone, 1.0, correct_map["MUO_cfg"][sf_type](mu_eta, mu_pt)
                    )
                    if syst:
                        sfs_up = np.where(
                            masknone,
                            1.0,
                            sfs
                            + correct_map["MUO_custom"][f"{sf_type}_error"](
                                mu_eta, mu_pt
                            ),
                        )
                        sfs_down = np.where(
                            masknone,
                            1.0,
                            sfs
                            - correct_map["MUO_custom"][f"{sf_type}_error"](
                                mu_eta, mu_pt
                            ),
                        )

            sfs_allmu = sfs_allmu * sfs
            if syst:
                sfs_allmu_down = sfs_allmu_down * sfs_down
                sfs_allmu_up = sfs_allmu_up * sfs_up

        if syst:
            weights.add(sf.split(" ")[0], sfs_allmu, sfs_allmu_up, sfs_allmu_down)
        else:
            weights.add(sf.split(" ")[0], sfs_allmu)

    return weights


def jmar_sf(jet, correct_map, weights, syst=False):
    alljet = jet if jet.ndim > 1 else ak.singletons(jet)
    for sf in correct_map["JMAR_cfg"].keys():
        sfs_all, sfs_all_up, sfs_all_down = (
            np.ones_like(alljet[:, 0].pt),
            np.ones_like(alljet[:, 0].pt),
            np.ones_like(alljet[:, 0].pt),
        )
        for njet in range(ak.num(alljet.pt)[0]):
            jet = alljet[:, njet]
            jet_eta = ak.fill_none(jet.eta, -2.5)
            jet_pt = ak.fill_none(jet.pt, 20)
            masknone = ak.is_none(jet.pt)
            # PU Jet ID applied only jet pT<50GeV
            if sf == "PUJetID_eff":
                jet_pt = np.where(jet.pt > 50, 20, jet.pt)
                masknone = (ak.is_none(jet.pt)) & (jet.pt > 50)
            if "correctionlib" in str(type(correct_map["JMAR"])):
                sfs = np.where(
                    masknone,
                    1.0,
                    correct_map["JMAR"][sf].evaluate(
                        jet_eta,
                        jet_pt,
                        "nom",
                        correct_map["JMAR_cfg"][sf],
                    ),
                )
                if syst:
                    sfs_up = np.where(
                        masknone,
                        1.0,
                        correct_map["JMAR"][sf].evaluate(
                            jet_eta,
                            jet_pt,
                            "up",
                            correct_map["JMAR_cfg"][sf],
                        ),
                    )
                    sfs_down = np.where(
                        masknone,
                        1.0,
                        correct_map["JMAR"][sf].evaluate(
                            jet_eta,
                            jet_pt,
                            "down",
                            correct_map["JMAR_cfg"][sf],
                        ),
                    )
                    sfs_all_up = sfs_up * sfs_all_up
                    sfs_all_down = sfs_down * sfs_all_down
            sfs_all = sfs * sfs_all
        if syst:
            weights.add(sf, sfs_all, sfs_all_up, sfs_all_down)
        else:
            weights.add(sf, sfs_all)


# Per-event LHE weight ratios (scale, PDF, alpha_S) are capped in MAGNITUDE at
# this value, sign kept. The normalisation stays exact: every variation is
# divided downstream by its own sum over all events, built from the same capped
# weights (reweighting -> sumw_<v>). Measured on one file of each of the 35 MC
# samples of the lepton workflows: |ratio| > 10 hits at most 0.05 % of events,
# and the cap lifts the effective statistics of the worst scale variation from
# 0.39 to 0.82 (W+0j), 0.45 to 0.66 (DY 10-50), 0.62 to 0.77 (WZ->l3nu 4FS). A
# cap of 5 gains nothing further.
LHE_RATIO_CAP = 10.0


def _cap_lhe_ratio(r):
    r = np.asarray(r, dtype=np.float64)
    r = np.where(np.isfinite(r), r, 1.0)
    return np.clip(r, -LHE_RATIO_CAP, LHE_RATIO_CAP)


# PDF sets in these samples (member 0 central, 1..100 Hessian eigenvectors).
# value = whether members 101/102 are the alpha_S variations (0.116 / 0.120).
# 325300: 31 of the 35 lepton-workflow samples, 325500 (4FS, no alpha_S): 4.
# The PDF eigenvectors themselves are NOT booked (decided 2026-09-30): in the
# final fit the PDF nuisances explained at most 0.84 % of any SF's variance
# (median 0.07 %), and a correct Hessian treatment needs 100 extra slices per
# histogram; the former per-event quadrature was not a PDF variation at all.
_PDF_HESSIAN_SETS = {
    "306000 - 306102": True,
    "325300 - 325402": True,
    "325500 - 325600": False,
}


def _pdf_set(pdf_weights):
    doc = getattr(pdf_weights, "__doc__", "") or ""
    return next((k for k in _PDF_HESSIAN_SETS if k in doc), None)


def add_pdf_weight(weights, pdf_weights, isSyst=False):
    """alpha_S only (no PDF eigenvector variation, see _PDF_HESSIAN_SETS).
    Up = alpha_S 0.120 (member 102), Down = 0.116 (member 101), both relative to
    the central member and capped. Sets without alpha_S members book a variation
    equal to the nominal, so the label exists everywhere and the tensor writer
    drops it as zero response."""
    nom = np.ones(len(weights.weight()))
    if isSyst == False:
        weights.add("aS_weight", nom)
        return
    key = _pdf_set(pdf_weights) if pdf_weights is not None else None
    if key is None:
        warnings.warn(
            "PDF weights are not available"
            if pdf_weights is None
            else "unexpected PDF set used!"
        )
    if key is not None and _PDF_HESSIAN_SETS[key]:
        w = ak.to_numpy(pdf_weights).astype(np.float64)
        ref = np.where(w[:, 0] != 0, w[:, 0], 1.0)
        weights.add(
            "aS_weight",
            nom,
            _cap_lhe_ratio(w[:, 102] / ref),
            _cap_lhe_ratio(w[:, 101] / ref),
        )
    else:
        weights.add("aS_weight", nom, nom, nom)


VPT_EWK_CAMPAIGNS = ("CustomSummer24", "CustomSummer25")


def vpt_ewk_kfactor(vpt, boson):
    """NLO EWK k-factor vs LHE V pT (fit functions, valid 100 <= pT < 3000 GeV, 1 elsewhere)."""
    vpt = np.asarray(vpt, dtype=np.float64)
    if boson == "W":
        k = -0.830041 + 7.93714 * np.power(vpt + 877.978, -0.213831)
    else:
        k = -0.1808051 + 6.04146 * np.power(vpt + 759.098, -0.242556)
    return np.where((vpt >= 100) & (vpt < 3000), k, 1.0)


# https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopPtReweighting#TOP_PAG_corrections_based_on_the
def top_pT_sf_formula(pt):
    x = np.clip(pt, 0, 2000)
    """
    # From page 30 of AN v9 https://cms.cern.ch/iCMS/jsp/db_notes/noteInfo.jsp?cmsnoteid=CMS%20AN-2024/019
    w_13_to_13p6 = (
        0.991 + 0.000075 * x
    )  # Extrapolation of the number below (which was for CoM = 13 TeV) to CoM = 13.6 TeV
    return (0.103 * np.exp(-0.0118 * x) - 0.000134 * x + 0.973) * w_13_to_13p6
    """
    return 0.103 * np.exp(-0.0118 * x) - 0.000134 * x + 0.973


def top_pT_reweighting(gen):
    # Apply this SF only to TTbar datasets! Updated to latest suggestion
    # Documentation:
    #     - https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopPtReweighting
    #     - https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopPtReweighting#TOP_PAG_corrections_based_on_the
    top = gen[(gen.pdgId == 6) & gen.hasFlags(["isLastCopy"])]
    anti_top = gen[(gen.pdgId == -6) & gen.hasFlags(["isLastCopy"])]
    return np.sqrt(
        top_pT_sf_formula(ak.flatten(top.pt, axis=-1))
        * top_pT_sf_formula(ak.flatten(anti_top.pt, axis=-1))
    )


def _top_pt_weight_from_branch(ev, runs_sumw):
    """
    Central top-pT correction from the stored `TopPtWeight[1]` (13.6 TeV) branch,
    normalised by the per-file Runs sum. `None` when the branch is absent, so callers can
    fall back to `top_pT_reweighting`. Guards per event against non-finite/non-positive
    entries (the sibling `TOPMLWeight` branch uses -1 as a non-ttbar sentinel; `TopPtWeight`
    doesn't, but the guard is cheap and keeps this safe if ever reached off a ttbar sample).
    """
    if "TopPtWeight" not in ev.fields:
        return None
    w = ak.to_numpy(ev.TopPtWeight[:, 1])
    valid = np.isfinite(w) & (w > 0)
    sumw = runs_sumw_norm(runs_sumw or {}, "TopPtWeightSumw", 2)
    central = w / sumw[1] if sumw is not None else w
    return np.where(valid, central, 1.0)


def _get_hdamp_ml_sessions(campaign):
    cache_key = f"hdamp::{campaign}"
    if cache_key in _TTBAR_REWEIGHT_CACHE:
        return _TTBAR_REWEIGHT_CACHE[cache_key]

    if ort is None:
        warnings.warn(
            "onnxruntime not available; hdamp ML reweighting disabled.", stacklevel=2
        )
        _TTBAR_REWEIGHT_CACHE[cache_key] = None
        return None

    pkg = "BTVNanoCommissioning.data.TTBAR_REWEIGHT.hdamp_ml"
    suffix = (
        "13TeV"
        if (campaign and ("-UL" in campaign or campaign.startswith("UL")))
        else "13.6TeV"
    )
    try:
        with contextlib.ExitStack() as stack:
            up_path = stack.enter_context(
                importlib.resources.path(pkg, f"mymodel12_hdamp_up_{suffix}.onnx")
            )
            dn_path = stack.enter_context(
                importlib.resources.path(pkg, f"mymodel12_hdamp_down_{suffix}.onnx")
            )
            sess_up = ort.InferenceSession(
                str(up_path), providers=["CPUExecutionProvider"]
            )
            sess_dn = ort.InferenceSession(
                str(dn_path), providers=["CPUExecutionProvider"]
            )
            payload = {
                "up": (
                    sess_up,
                    sess_up.get_inputs()[0].name,
                    sess_up.get_outputs()[0].name,
                ),
                "dn": (
                    sess_dn,
                    sess_dn.get_inputs()[0].name,
                    sess_dn.get_outputs()[0].name,
                ),
            }
    except Exception as exc:
        warnings.warn(
            f"Could not load hdamp ML ONNX models: {exc}. Using nominal weights.",
            stacklevel=2,
        )
        payload = None

    _TTBAR_REWEIGHT_CACHE[cache_key] = payload
    return payload


# Mass normalisation constant used by the hdamp ML model during training
_HDAMP_MASS_NORM = 243.9517


def add_hdamp_ml_weight(weights, pruned_ev, campaign, isSyst=False, runs_sumw=None):
    nom = np.ones(len(weights.weight()), dtype=np.float64)
    n = len(nom)

    def _add(up=None, dn=None):
        if isSyst is not False and up is not None:
            weights.add("hdampML", nom, up, dn)
        else:
            weights.add("hdampML", nom)

    if "TOPMLWeight" in pruned_ev.fields:
        # TOPMLWeight[1]/[3] = hdamp up/down at 13.6 TeV. Non-ttbar samples fill this
        # branch with -1 sentinels (not NaN-safe), so a field-presence check alone isn't a
        # ttbar guard: reject any non-finite/non-positive entry per event before using it.
        topml = ak.to_numpy(pruned_ev.TOPMLWeight)
        raw_up, raw_dn = topml[:, 1], topml[:, 3]
        valid = np.isfinite(raw_up) & (raw_up > 0) & np.isfinite(raw_dn) & (raw_dn > 0)
        sumw = runs_sumw_norm(runs_sumw or {}, "TOPMLWeightSumw", 7)
        up = raw_up / sumw[1] if sumw is not None else raw_up
        dn = raw_dn / sumw[3] if sumw is not None else raw_dn
        return _add(np.where(valid, up, 1.0), np.where(valid, dn, 1.0))

    if "GenPart" not in pruned_ev.fields:
        warnings.warn("GenPart missing; hdamp ML weights set to nominal.", stacklevel=2)
        return _add()

    sessions = _get_hdamp_ml_sessions(campaign)
    if sessions is None:
        return _add()

    gp = pruned_ev.GenPart

    # Prefer first-copy tops; fall back to last copy. hasFlags is the NanoEvents API;
    # fall back to statusFlags bit test for plain awkward arrays (e.g. in unit tests).
    def _flag(gp_, name):
        if hasattr(gp_, "hasFlags"):
            return gp_.hasFlags([name])
        bit = {"isFirstCopy": _FIRST_COPY_BIT, "isLastCopy": _LAST_COPY_BIT}.get(name)
        return (
            (gp_.statusFlags & bit) != 0
            if bit and "statusFlags" in gp_.fields
            else ak.zeros_like(gp_.pdgId, dtype=bool)
        )

    def _tops(pid, flag):
        return ak.firsts(gp[(gp.pdgId == pid) & _flag(gp, flag)])

    top = _tops(6, "isFirstCopy")
    atop = _tops(-6, "isFirstCopy")
    if ak.all(ak.is_none(top)) or ak.all(ak.is_none(atop)):
        top = _tops(6, "isLastCopy")
        atop = _tops(-6, "isLastCopy")

    valid = (~ak.is_none(top)) & (~ak.is_none(atop))

    def _f(arr, fill):
        return ak.to_numpy(ak.fill_none(arr, fill))

    t_pt = np.maximum(_f(top.pt, 1.0), 1e-6)
    t_eta = _f(top.eta, 0.0)
    t_phi = _f(top.phi, 0.0)
    t_m = np.maximum(_f(top.mass, 172.5), 1e-6)
    at_pt = np.maximum(_f(atop.pt, 1.0), 1e-6)
    at_eta = _f(atop.eta, 0.0)
    at_phi = _f(atop.phi, 0.0)
    at_m = np.maximum(_f(atop.mass, 172.5), 1e-6)

    ttbar_pt = np.sqrt(
        (t_pt * np.cos(t_phi) + at_pt * np.cos(at_phi)) ** 2
        + (t_pt * np.sin(t_phi) + at_pt * np.sin(at_phi)) ** 2
    )

    # Build (n, 2, 6) feature tensor; rapidity = 0.5*ln((E+pz)/(E-pz))
    def _rap(pt, eta, m):
        E = np.sqrt((pt * np.cosh(eta)) ** 2 + m**2)
        pz = pt * np.sinh(eta)
        return 0.5 * np.log(np.maximum((E + pz) / np.maximum(E - pz, 1e-10), 1e-10))

    x = np.zeros((n, 2, 6), dtype=np.float32)
    x[:, 0] = np.stack(
        [
            np.log10(t_pt),
            _rap(t_pt, t_eta, t_m),
            t_phi,
            t_m / _HDAMP_MASS_NORM,
            np.full(n, 0.1),
            np.full(n, 1.379),
        ],
        axis=1,
    )
    x[:, 1] = np.stack(
        [
            np.log10(at_pt),
            _rap(at_pt, at_eta, at_m),
            at_phi,
            at_m / _HDAMP_MASS_NORM,
            np.full(n, 0.2),
            np.full(n, 1.379),
        ],
        axis=1,
    )

    apply_mask = (
        ak.to_numpy(valid)
        & np.all(np.isfinite(x.reshape(n, -1)), axis=1)
        & (ttbar_pt < 1000.0)
    )

    up, dn = nom.copy(), nom.copy()
    if np.any(apply_mask):
        x_sel = x[apply_mask]

        def _infer(sess, iname, oname):
            out = np.asarray(sess.run([oname], {iname: x_sel})[0])
            if out.ndim > 1 and out.shape[-1] >= 2:
                out = out[..., 0] / np.maximum(out[..., 1], 1e-8)
            else:
                out = out.reshape(-1)
            return np.where(
                np.isfinite(out) & (out > 0) & (out < 100), out, 1.0
            ).astype(np.float64)

        up[apply_mask] = _infer(*sessions["up"])
        dn[apply_mask] = _infer(*sessions["dn"])

    _add(up, dn)


_FIRST_COPY_BIT = 1 << 12
_LAST_COPY_BIT = 1 << 13
_LEPTON_IDS = {11, 13}
_B_HADRONS = {
    511,
    521,
    531,
    541,
    5122,
    5112,
    5132,
    5232,
    5332,
}
_C_HADRONS = {
    411,
    421,
    431,
    4122,
    4132,
    4232,
    4332,
}
# numba-friendly array forms of the sets above, for _find_top_hadron_pair's jitted kernel
_LEPTON_IDS_ARR = np.array(sorted(_LEPTON_IDS), dtype=np.int64)
_B_HADRONS_ARR = np.array(sorted(_B_HADRONS), dtype=np.int64)
_C_HADRONS_ARR = np.array(sorted(_C_HADRONS), dtype=np.int64)


def _load_frag_decay_maps():
    """Load TGraph weight files once and return a dict of interpolation callables."""
    cache_key = "frag_decay_maps"
    if cache_key in _TTBAR_REWEIGHT_CACHE:
        return _TTBAR_REWEIGHT_CACHE[cache_key]

    def _tgraph_fn(obj):
        """Return a callable f(values) -> np.ndarray that interpolates a TGraph."""
        try:
            x = np.asarray(obj.member("fX"), dtype=np.float64)
            y = np.asarray(obj.member("fY"), dtype=np.float64)
        except Exception:
            x, y = np.array([0.0, 1.0]), np.array([1.0, 1.0])
        if len(x) == 0:
            x, y = np.array([0.0, 1.0]), np.array([1.0, 1.0])
        order = np.argsort(x)
        xs, ys = x[order], y[order]

        def _eval(values):
            out = np.interp(
                np.asarray(values, dtype=np.float64), xs, ys, left=ys[0], right=ys[-1]
            )
            return np.where(np.isfinite(out), out, 1.0)

        return _eval

    pkg = "BTVNanoCommissioning.data.TTBAR_REWEIGHT.frag_decay"
    payload = None
    try:
        with contextlib.ExitStack() as stack:
            bfrag_path = stack.enter_context(
                importlib.resources.path(pkg, "bfragweights.root")
            )
            cfrag_path = stack.enter_context(
                importlib.resources.path(pkg, "cfragweights.root")
            )
            bdec_path = stack.enter_context(
                importlib.resources.path(pkg, "bdecayweights.root")
            )
            cdec_path = stack.enter_context(
                importlib.resources.path(pkg, "cdecayweights.root")
            )

            with uproot.open(str(bfrag_path)) as fb, uproot.open(str(cfrag_path)) as fc:
                with (
                    uproot.open(str(bdec_path)) as fbdec,
                    uproot.open(str(cdec_path)) as fcdec,
                ):
                    payload = {
                        "b_bl_up": _tgraph_fn(fb["fragCP5BLup_smooth"]),
                        "b_bl_dn": _tgraph_fn(fb["fragCP5BLdown_smooth"]),
                        "b_pet": _tgraph_fn(fb["fragCP5Peterson_smooth"]),
                        "c_bl_up": _tgraph_fn(fc["fragCP5BLup_smooth"]),
                        "c_bl_dn": _tgraph_fn(fc["fragCP5BLdown_smooth"]),
                        "c_pet": _tgraph_fn(fc["fragCP5Peterson_smooth"]),
                        "bdec_up": _tgraph_fn(fbdec["semilepbrup"]),
                        "bdec_dn": _tgraph_fn(fbdec["semilepbrdown"]),
                        "cdec_up": _tgraph_fn(fcdec["semilepbrup"]),
                        "cdec_dn": _tgraph_fn(fcdec["semilepbrdown"]),
                    }
    except Exception as exc:
        warnings.warn(
            f"Could not load fragmentation/decay weight files: {exc}. Falling back to nominal weights.",
            stacklevel=2,
        )

    _TTBAR_REWEIGHT_CACHE[cache_key] = payload
    return payload


@numba.njit(cache=True)
def _descends_from_numba(mother, idx, ancestor, n):
    cur, steps = idx, 0
    while cur >= 0 and steps < n + 5:
        if cur == ancestor:
            return True
        cur = mother[cur] if cur < n else -1
        steps += 1
    return False


@numba.njit(cache=True)
def _has_lepton_descendant_numba(pdg, mother, idx, lepton_ids, n):
    # Equivalent to "does idx have a lepton among its children/grandchildren/...";
    # reformulated as "is idx an ancestor of some lepton" so it needs only the
    # mother-pointer array, not a precomputed children map.
    for j in range(n):
        pid = pdg[j]
        if pid < 0:
            pid = -pid
        is_lepton = False
        for lid in lepton_ids:
            if pid == lid:
                is_lepton = True
                break
        if is_lepton and _descends_from_numba(mother, j, idx, n):
            return True
    return False


@numba.njit(cache=True)
def _best_hadron_numba(
    pdg, mother, status, pt, hadron_ids, root_idx, first_copy_bit, n
):
    # Heaviest first-copy hadron descended from root_idx; fall back to any copy.
    for require_first in (True, False):
        best_i, best_pt = -1, -1.0
        for i in range(n):
            apid = pdg[i]
            if apid < 0:
                apid = -apid
            is_hadron = False
            for hid in hadron_ids:
                if apid == hid:
                    is_hadron = True
                    break
            if not is_hadron:
                continue
            if require_first and (status[i] & first_copy_bit) == 0:
                continue
            if _descends_from_numba(mother, i, root_idx, n) and pt[i] > best_pt:
                best_i, best_pt = i, pt[i]
        if best_i != -1:
            return best_i
    return -1


@numba.njit(cache=True)
def _find_top_hadron_pair_numba(
    pdg, mother, status, pt, mass, hadron_ids, lepton_ids, first_copy_bit, last_copy_bit
):
    n = pdg.shape[0]

    # Last-copy top/antitop, with pdgId-only fallback
    tidx = -1
    for i in range(n):
        if pdg[i] == 6 and (status[i] & last_copy_bit) != 0:
            tidx = i
            break
    aidx = -1
    for i in range(n):
        if pdg[i] == -6 and (status[i] & last_copy_bit) != 0:
            aidx = i
            break
    if tidx == -1:
        for i in range(n):
            if pdg[i] == 6:
                tidx = i
                break
    if aidx == -1:
        for i in range(n):
            if pdg[i] == -6:
                aidx = i
                break
    if tidx == -1 or aidx == -1:
        return False, -1, 0.0, -1, False, -1, 0.0, -1, False

    # W boson mass from first W child of each top
    w_m, found_w = 0.0, False
    for i in range(n):
        if mother[i] == tidx:
            apid = pdg[i]
            if apid < 0:
                apid = -apid
            if apid == 24:
                w_m, found_w = mass[i], True
                break
    aw_m, found_aw = 0.0, False
    for i in range(n):
        if mother[i] == aidx:
            apid = pdg[i]
            if apid < 0:
                apid = -apid
            if apid == 24:
                aw_m, found_aw = mass[i], True
                break
    if not found_w or not found_aw:
        return False, -1, 0.0, -1, False, -1, 0.0, -1, False

    hid = _best_hadron_numba(
        pdg, mother, status, pt, hadron_ids, tidx, first_copy_bit, n
    )
    ahid = _best_hadron_numba(
        pdg, mother, status, pt, hadron_ids, aidx, first_copy_bit, n
    )
    if hid == -1 or ahid == -1:
        return False, -1, 0.0, -1, False, -1, 0.0, -1, False

    h_sl = _has_lepton_descendant_numba(pdg, mother, hid, lepton_ids, n)
    ah_sl = _has_lepton_descendant_numba(pdg, mother, ahid, lepton_ids, n)

    return True, tidx, w_m, hid, h_sl, aidx, aw_m, ahid, ah_sl


def _find_top_hadron_pair(pdg_a, mother_a, status_a, pt_a, mass_a, hadron_ids_arr):
    """
    For one event, find the heaviest B/C hadron pair descended from t and tbar.
    pdg_a/mother_a/status_a/pt_a/mass_a are per-event numpy arrays (see
    add_fragmentation_decay_weights for how they're built from GenPart fields).
    Returns (tidx, w_mass, hid, h_semilep, aidx, aw_mass, ahid, ah_semilep) or None.
    """
    found, tidx, w_m, hid, h_sl, aidx, aw_m, ahid, ah_sl = _find_top_hadron_pair_numba(
        pdg_a,
        mother_a,
        status_a,
        pt_a,
        mass_a,
        hadron_ids_arr,
        _LEPTON_IDS_ARR,
        _FIRST_COPY_BIT,
        _LAST_COPY_BIT,
    )
    if not found:
        return None
    return tidx, w_m, hid, h_sl, aidx, aw_m, ahid, ah_sl


def _safe_ratio(num, den):
    """num/den, guarded against non-finite/zero denominators and clamped to a sane range."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den
    return np.where(np.isfinite(r) & (r > 0) & (r < 100), r, 1.0)


def _add_fragdecay_from_branches(
    weights, pruned_ev, isSyst, runs_sumw, LABELS, nom, _commit
):
    """
    Central + variations from the stored `bFragAndDecayWeight`/`cFragAndDecayWeight`
    branches of the custom NanoTuples. Filled in GenWeightsTableProducer as a product
    over particle-level jets of the per-jet weights of BFragmentationWeightProducer
    (tthcc-run-3/BFragmentationAnalyzer):

        [0] fragCP5BLVsPt        central fragmentation tune  -> the nominal
        [1] fragCP5BLupVsPt      replaces [0]
        [2] fragCP5BLdownVsPt    replaces [0]
        [3] fragCP5PetersonVsPt  replaces [0]
        [4] semilepbrup          MULTIPLIES [0]; does not contain it
        [5] semilepbrdown        MULTIPLIES [0]; does not contain it

    So on top of the nominal the fragmentation variations are [k]/[0], while the
    decay variations are [4] and [5] themselves. Dividing those by [0] as well puts
    the INVERSE central fragmentation weight into both decay legs, which makes them
    move the same way (measured on TTto2L2Nu: 89 % of events same-sign with /[0],
    4 % without). The central entry and every variation are normalised with the
    per-file Runs-tree sums, so the systematics are acceptance-only.
    """
    b = ak.to_numpy(pruned_ev.bFragAndDecayWeight)
    c = ak.to_numpy(pruned_ev.cFragAndDecayWeight)

    b_sumw = runs_sumw_norm(runs_sumw or {}, "BFragAndDecayWeightSumw", 6)
    c_sumw = runs_sumw_norm(runs_sumw or {}, "CFragAndDecayWeightSumw", 6)
    b_central = b[:, 0] / b_sumw[0] if b_sumw is not None else b[:, 0]
    c_central = c[:, 0] / c_sumw[0] if c_sumw is not None else c[:, 0]
    b_central = np.where(np.isfinite(b_central) & (b_central > 0), b_central, 1.0)
    c_central = np.where(np.isfinite(c_central) & (c_central > 0), c_central, 1.0)
    weights.add("bFragDecayCentral", b_central)
    weights.add("cFragDecayCentral", c_central)

    # Normalise each variation to its own inclusive sum so the systematic is
    # acceptance-only. Fragmentation: [k]/[0] carries Sumw[k]/Sumw[0], divide it
    # out. Decay: w*[0]*[4] has inclusive sum Sumw[0]*Sumw[4] to 0.004 %
    # (measured on TTto2L2Nu; the two weights are independent), and the nominal
    # already divides by Sumw[0], so the decay factor is [4]/Sumw[4].
    bN = b_sumw[0] / b_sumw if b_sumw is not None else np.ones(6)
    cN = c_sumw[0] / c_sumw if c_sumw is not None else np.ones(6)
    bD = 1.0 / b_sumw if b_sumw is not None else np.ones(6)
    cD = 1.0 / c_sumw if c_sumw is not None else np.ones(6)
    bfu, bfd = (
        _safe_ratio(b[:, 1], b[:, 0]) * bN[1],
        _safe_ratio(b[:, 2], b[:, 0]) * bN[2],
    )
    # Peterson is ONE alternative model, not a +-1 sigma pair. The old down leg
    # 1/up was not normalised: E[1/x] > 1/E[x] gave +4.7 % inclusive yield for b
    # on TTto2L2Nu, with per-event weights up to 32. The down leg is now the
    # nominal; rabbit_write_tensor.py mirrors one-sided templates at histogram
    # level, where the bin sums are well behaved.
    bpu = _safe_ratio(b[:, 3], b[:, 0]) * bN[3]
    bpd = np.ones_like(bpu)
    bdu, bdd = (
        _safe_ratio(b[:, 4], 1.0) * bD[4],
        _safe_ratio(b[:, 5], 1.0) * bD[5],
    )
    cfu, cfd = (
        _safe_ratio(c[:, 1], c[:, 0]) * cN[1],
        _safe_ratio(c[:, 2], c[:, 0]) * cN[2],
    )
    cpu = _safe_ratio(c[:, 3], c[:, 0]) * cN[3]
    cpd = np.ones_like(cpu)
    cdu, cdd = (
        _safe_ratio(c[:, 4], 1.0) * cD[4],
        _safe_ratio(c[:, 5], 1.0) * cD[5],
    )

    _commit(
        [
            (bfu, bfd),
            (bpu, bpd),
            (cfu, cfd),
            (cpu, cpd),
            (bdu, bdd),
            (cdu, cdd),
        ]
    )
    if isSyst is not False:
        _add_decay_species_weights(weights, b[:, 4], b[:, 5], "b")
        _add_decay_species_weights(weights, c[:, 4], c[:, 5], "c")


# Hadron species of the semileptonic-BR weights (BFragmentationWeightProducer.cc
# l. 168-186 evaluates the graphs only for these; any other hadron gets 1.0).
_DECAY_SPECIES = {
    "b": (("B0", 511), ("Bplus", 521), ("Bs", 531), ("Lb", 5122)),
    "c": (("Dplus", 411), ("D0", 421), ("Ds", 431), ("Lc", 4122)),
}
# at most one weighted hadron per particle-level jet (the leading ghost hadron)
_DECAY_MAX_HADRONS = {"b": 6, "c": 8}
# The NanoTuple stores these weights with a reduced mantissa: 1.1182555 is read
# back as 1.11823. Measured on TTto2L2Nu (40k events): every event matches a
# table point within 5.3e-5 in log (L1 over the two legs), so 1.5e-4 accepts all.
# 19 of 40k b events have a second match within that distance, all with >= 3
# weighted hadrons (e.g. 2 B0 sl + 1 B0 vs 2 B+ sl + 2 B+); the fewest-hadrons
# assignment is taken.
_DECAY_LOG_TOL = 1e-4
_DECAY_LOG_ACCEPT = 1.5e-4


def _decay_decomposition_table(flav):
    """All hadron-count vectors over the 8 (species x semileptonic) categories
    with at most _DECAY_MAX_HADRONS hadrons, and their log products under the
    up and the down graph, keyed on the rounded pair of log products."""
    key = f"decay_table::{flav}"
    if key in _TTBAR_REWEIGHT_CACHE:
        return _TTBAR_REWEIGHT_CACHE[key]
    import itertools

    pkg = "BTVNanoCommissioning.data.TTBAR_REWEIGHT.frag_decay"
    with importlib.resources.path(pkg, f"{flav}decayweights.root") as p:
        with uproot.open(str(p)) as fh:
            graphs = {
                leg: dict(
                    zip(
                        fh[name].values("x").astype(int),
                        fh[name].values("y").astype(np.float64),
                    )
                )
                for leg, name in (("up", "semilepbrup"), ("dn", "semilepbrdown"))
            }
    cats = [(sp, pid, sl) for sp, pid in _DECAY_SPECIES[flav] for sl in (True, False)]
    # the producer stores the per-jet weights as float
    lu = np.log([np.float32(graphs["up"][pid if sl else -pid]) for _, pid, sl in cats])
    ld = np.log([np.float32(graphs["dn"][pid if sl else -pid]) for _, pid, sl in cats])
    counts = np.array(
        [
            np.bincount(np.array(combo, dtype=np.int64), minlength=len(cats))
            for tot in range(_DECAY_MAX_HADRONS[flav] + 1)
            for combo in itertools.combinations_with_replacement(range(len(cats)), tot)
        ],
        dtype=np.int64,
    )
    L4, L5 = counts @ lu, counts @ ld
    lookup = {}
    for i, (a, b) in enumerate(zip(L4, L5)):
        lookup.setdefault(
            (int(round(a / _DECAY_LOG_TOL)), int(round(b / _DECAY_LOG_TOL))), []
        ).append(i)
    table = (cats, counts, L4, L5, lu, ld, lookup)
    _TTBAR_REWEIGHT_CACHE[key] = table
    return table


def _add_decay_species_weights(weights, w_up, w_dn, flav):
    """One nuisance per hadron species, `<flav>decay_<species>`, IN ADDITION to
    the combined `<flav>decay`.

    The stored per-event `[4]`/`[5]` are products over jets of per-hadron graph
    values, a small discrete set, so each event decomposes exactly into counts
    of (species, semileptonic) hadrons. Every species then gets its own weight:
    the product of its hadrons' up-graph (down-graph) values. The graphs are
    one-sided per species (buildBRweights.py moves a species only where PDG +-
    sigma leaves the Pythia BR), so for B0, B+, Ds, Lambda_c only the down
    graph and for Lambda_b only the up graph moves; the missing leg is the
    per-event log mirror 1/w. Normalisation is done downstream with the
    per-variation sums (reweighting -> scaleSumW), so the mirror carries no
    yield bias. Events that do not decompose (more hadrons than the table, or a
    corrupted weight) keep 1.0 for every species; the rate is reported."""
    cats, counts, L4, L5, lu, ld, lookup = _decay_decomposition_table(flav)
    l4 = np.log(np.where(np.isfinite(w_up) & (w_up > 0), w_up, 1.0))
    l5 = np.log(np.where(np.isfinite(w_dn) & (w_dn > 0), w_dn, 1.0))
    n = np.zeros((len(l4), len(cats)), dtype=np.int64)
    bad = 0
    k4 = np.rint(l4 / _DECAY_LOG_TOL).astype(np.int64)
    k5 = np.rint(l5 / _DECAY_LOG_TOL).astype(np.int64)
    for i in range(len(l4)):
        best, bestkey = None, None
        for d4 in (-2, -1, 0, 1, 2):
            for d5 in (-2, -1, 0, 1, 2):
                for j in lookup.get((int(k4[i]) + d4, int(k5[i]) + d5), ()):
                    dist = abs(L4[j] - l4[i]) + abs(L5[j] - l5[i])
                    if dist > _DECAY_LOG_ACCEPT:
                        continue
                    key = (int(counts[j].sum()), dist)
                    if bestkey is None or key < bestkey:
                        best, bestkey = j, key
        if best is None:
            bad += 1
            continue
        n[i] = counts[best]
    if bad:
        warnings.warn(
            f"{flav}decay per-species split: {bad} of {len(l4)} events did not "
            "decompose; they keep 1.0 for every species",
            stacklevel=2,
        )
    ones = np.ones(len(l4), dtype=np.float64)
    for sp, _pid in _DECAY_SPECIES[flav]:
        idx = [k for k, c in enumerate(cats) if c[0] == sp]
        up = np.exp(n[:, idx] @ lu[idx])
        dn = np.exp(n[:, idx] @ ld[idx])
        up_moves, dn_moves = np.any(lu[idx] != 0), np.any(ld[idx] != 0)
        if up_moves and not dn_moves:
            dn = 1.0 / up
        elif dn_moves and not up_moves:
            up = 1.0 / dn
        weights.add(f"{flav}decay_{sp}", ones, up, dn)


def add_fragmentation_decay_weights(
    weights, pruned_ev, isSyst=False, runs_sumw=None, genpart_fallback=False
):
    LABELS = [
        "bfragBL",
        "bfragPeterson",
        "cfragBL",
        "cfragPeterson",
        "bdecay",
        "cdecay",
    ]
    nom = np.ones(len(weights.weight()), dtype=np.float64)
    n_evt = len(nom)

    def _commit(pairs=None):
        if isSyst is not False and pairs is not None:
            for lbl, (up, dn) in zip(LABELS, pairs):
                weights.add(lbl, nom, up, dn)
        else:
            for lbl in LABELS:
                weights.add(lbl, nom)

    if (
        "bFragAndDecayWeight" in pruned_ev.fields
        and "cFragAndDecayWeight" in pruned_ev.fields
    ):
        return _add_fragdecay_from_branches(
            weights, pruned_ev, isSyst, runs_sumw, LABELS, nom, _commit
        )

    if not genpart_fallback:
        return _commit()

    if "GenPart" not in pruned_ev.fields:
        warnings.warn(
            "GenPart missing; fragmentation/decay weights set to nominal.", stacklevel=2
        )
        return _commit()

    maps = _load_frag_decay_maps()
    if maps is None:
        return _commit()

    warnings.warn(
        "c-fragmentation reweighting uses approximate hadron-top ancestry reconstruction in NanoAOD.",
        stacklevel=2,
    )

    # Convert awkward arrays to plain lists once for all events
    gp = pruned_ev.GenPart
    pdg_all = ak.to_list(gp.pdgId)
    mother_all = ak.to_list(gp.genPartIdxMother)
    status_all = ak.to_list(gp.statusFlags)
    pt_all = ak.to_list(gp.pt)
    eta_all = ak.to_list(gp.eta)
    phi_all = ak.to_list(gp.phi)
    mass_all = ak.to_list(gp.mass)

    arrs = {
        k: np.ones(n_evt, dtype=np.float64)
        for k in [
            "bfu",
            "bfd",
            "bpu",
            "bpd",
            "bdu",
            "bdd",
            "cfu",
            "cfd",
            "cpu",
            "cpd",
            "cdu",
            "cdd",
        ]
    }

    # x_had = 2*(p_had · p_top) / (m_top^2 - m_W^2)  (Lorentz-invariant fragmentation variable)
    def _x_had(hpt, heta, hphi, hm, tpt, teta, tphi, tm, wm):
        denom = tm**2 - wm**2
        if abs(denom) < 1e-6:
            return None
        E_h = np.sqrt((hpt * np.cosh(heta)) ** 2 + hm**2)
        E_t = np.sqrt((tpt * np.cosh(teta)) ** 2 + tm**2)
        dot = (
            E_h * E_t
            - hpt * np.cos(hphi) * tpt * np.cos(tphi)
            - hpt * np.sin(hphi) * tpt * np.sin(tphi)
            - hpt * np.sinh(heta) * tpt * np.sinh(teta)
        )
        x = 2.0 * dot / denom
        return float(x) if np.isfinite(x) else None

    for ievt in range(n_evt):
        pdg = pdg_all[ievt]
        mother = mother_all[ievt]
        status = status_all[ievt]
        pt = pt_all[ievt]
        eta = eta_all[ievt]
        phi = phi_all[ievt]
        mass = mass_all[ievt]
        n_gp = len(pdg)
        pdg_a = np.asarray(pdg, dtype=np.int64)
        mother_a = np.array(
            [m if m is not None else -1 for m in mother], dtype=np.int64
        )
        status_a = np.asarray(status, dtype=np.int64)
        pt_a = np.asarray(pt, dtype=np.float64)
        mass_a = np.asarray(mass, dtype=np.float64)

        for hadron_ids_arr, pf in ((_B_HADRONS_ARR, "b"), (_C_HADRONS_ARR, "c")):
            res = _find_top_hadron_pair(
                pdg_a, mother_a, status_a, pt_a, mass_a, hadron_ids_arr
            )
            if res is None:
                continue
            tidx, w_m, hid, h_sl, aidx, aw_m, ahid, ah_sl = res

            x1 = _x_had(
                pt[hid],
                eta[hid],
                phi[hid],
                mass[hid],
                pt[tidx],
                eta[tidx],
                phi[tidx],
                mass[tidx],
                w_m,
            )
            x2 = _x_had(
                pt[ahid],
                eta[ahid],
                phi[ahid],
                mass[ahid],
                pt[aidx],
                eta[aidx],
                phi[aidx],
                mass[aidx],
                aw_m,
            )
            if x1 is None or x2 is None or x1 > 1.2 or x2 > 1.2:
                continue

            xv = np.array([x1, x2], dtype=np.float64)
            # sign of pdgId encodes semi-leptonic decay for the decay weight lookup
            pid1 = abs(int(pdg[hid])) if h_sl else -abs(int(pdg[hid]))
            pid2 = abs(int(pdg[ahid])) if ah_sl else -abs(int(pdg[ahid]))
            pids = np.array([pid1, pid2], dtype=np.float64)

            if pf == "b":
                arrs["bfu"][ievt] = float(np.prod(maps["b_bl_up"](xv)))
                arrs["bfd"][ievt] = float(np.prod(maps["b_bl_dn"](xv)))
                # Peterson down stays at the nominal (initialised to 1): one
                # alternative model, mirrored at template level, as in the
                # branch path
                arrs["bpu"][ievt] = float(np.prod(maps["b_pet"](xv)))
                arrs["bdu"][ievt] = float(np.prod(maps["bdec_up"](pids)))
                arrs["bdd"][ievt] = float(np.prod(maps["bdec_dn"](pids)))
            else:
                arrs["cfu"][ievt] = float(np.prod(maps["c_bl_up"](xv)))
                arrs["cfd"][ievt] = float(np.prod(maps["c_bl_dn"](xv)))
                arrs["cpu"][ievt] = float(np.prod(maps["c_pet"](xv)))
                arrs["cdu"][ievt] = float(np.prod(maps["cdec_up"](pids)))
                arrs["cdd"][ievt] = float(np.prod(maps["cdec_dn"](pids)))

    for a in arrs.values():
        a[:] = np.where(np.isfinite(a) & (a > 0) & (a < 100), a, 1.0)

    _commit(
        [
            (arrs["bfu"], arrs["bfd"]),
            (arrs["bpu"], arrs["bpd"]),
            (arrs["cfu"], arrs["cfd"]),
            (arrs["cpu"], arrs["cpd"]),
            (arrs["bdu"], arrs["bdd"]),
            (arrs["cdu"], arrs["cdd"]),
        ]
    )


_RUNS_SUMW_CACHE = {}

# `Runs`-tree branches holding, per stored-weight index, sum(genWeight * w_i) already
# divided by genEventSumw. So the per-file normalisation of variation i is 1 / Sumw[i].
_RUNS_SUMW_BRANCHES = (
    "PSSumw",
    "BFragAndDecayWeightSumw",
    "CFragAndDecayWeightSumw",
    "TOPMLWeightSumw",
    "TopPtWeightSumw",
)


def get_runs_sumw(filename):
    """Per-file `Runs`-tree weight sums, cached by filename. `{}` when unavailable."""
    if filename in _RUNS_SUMW_CACHE:
        return _RUNS_SUMW_CACHE[filename]

    sums = {}
    if filename:
        try:
            with uproot.open(filename) as fh:
                runs = fh["Runs"]
                keys = set(runs.keys())
                gsumw = np.asarray(
                    runs["genEventSumw"].array(library="np"), dtype=np.float64
                )
                for br in _RUNS_SUMW_BRANCHES:
                    if br not in keys:
                        continue
                    per_run = np.array(
                        [
                            np.asarray(v, dtype=np.float64)
                            for v in runs[br].array().to_list()
                        ]
                    )
                    # the stored sums are ratios to genEventSumw, so combining several
                    # Runs entries has to put that denominator back first
                    sums[br] = (per_run * gsumw[:, None]).sum(axis=0) / gsumw.sum()
        except Exception as exc:
            warnings.warn(
                f"Could not read Runs-tree sums from {filename}: {exc}", stacklevel=2
            )

    _RUNS_SUMW_CACHE[filename] = sums
    return sums


def runs_sumw_norm(runs_sumw, branch, n_expected):
    """Return the Sumw vector for `branch`, or None if it cannot be trusted."""
    s = runs_sumw.get(branch)
    if s is None:
        return None
    if len(s) != n_expected or not np.all(np.isfinite(s)) or np.any(s <= 0):
        warnings.warn(
            f"{branch} is not usable for normalisation (length {len(s)}, expected "
            f"{n_expected}; min {np.min(s)}); leaving those weights un-normalised.",
            stacklevel=2,
        )
        return None
    return s


# Jennet adds PS weights
# https://github.com/andrzejnovak/boostedhiggs/blob/master/boostedhiggs/corrections.py#L88-L108
def add_ps_weight(weights, ps_weights, isSyst=False, runs_sumw=None):
    nom = np.ones(len(weights.weight()))

    if ps_weights is None or isSyst == False:
        return

    n = len(ps_weights[0])
    if n == 4:
        up_isr = ps_weights[:, 0]
        down_isr = ps_weights[:, 2]
        up_fsr = ps_weights[:, 1]
        down_fsr = ps_weights[:, 3]
        weights.add("UEPS_ISR", nom, up_isr, down_isr)
        weights.add("UEPS_FSR", nom, up_fsr, down_fsr)
    elif n == 44:
        # 22 up/down pairs: mu_R x0.707/1.414, x0.5/2, x0.25/4, plus per-kernel
        # g2gg/g2qq/q2qg/x2xg mu_R and cns, separately for ISR and FSR.
        # FSR indices given; ISR = FSR + 22. "Up" is the larger murfac / cns=+2.0.
        pairs = [
            (None, 3, 2),
            ("muR_sqrt2", 1, 0),
            ("muR_4", 5, 4),
            ("g2gg_muR", 7, 6),
            ("g2qq_muR", 9, 8),
            ("q2qg_muR", 11, 10),
            ("x2xg_muR", 13, 12),
            ("g2gg_cns", 15, 14),
            ("g2qq_cns", 17, 16),
            ("q2qg_cns", 19, 18),
            ("x2xg_cns", 21, 20),
        ]
        norm = runs_sumw_norm(runs_sumw or {}, "PSSumw", 44)
        for suffix, fsr_up, fsr_down in pairs:
            isr_up, isr_down = fsr_up + 22, fsr_down + 22
            if suffix is None:
                fsr_name, isr_name = "UEPS_FSR", "UEPS_ISR"
            else:
                fsr_name, isr_name = f"PS_FSR_{suffix}", f"PS_ISR_{suffix}"
            for name, i_up, i_down in (
                (fsr_name, fsr_up, fsr_down),
                (isr_name, isr_up, isr_down),
            ):
                # The per-file Runs sums keep each factor near 1; the exact
                # acceptance-only normalisation is done downstream for EVERY
                # generator-level variation (reweighting -> scaleSumW), where
                # dividing by the same file constant again cancels.
                if norm is None:
                    up, down = ps_weights[:, i_up], ps_weights[:, i_down]
                else:
                    up = ps_weights[:, i_up] / norm[i_up]
                    down = ps_weights[:, i_down] / norm[i_down]
                weights.add(name, nom, up, down)
    else:
        warnings.warn(f"PS weight vector has length {n}")
        weights.add("UEPS_ISR", nom, nom, nom)
        weights.add("UEPS_FSR", nom, nom, nom)


def add_scalevar_weight(weights, lhe_weights, isSyst=False):
    """
    Twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopSystematics#Factorization_and_renormalizatio

    __doc__:
    ['LHE scale variation weights (w_var / w_nominal)',
    ' [0] is renscfact=0.5d0 facscfact=0.5d0 ',
    ' [1] is renscfact=0.5d0 facscfact=1d0 ',
    ' [2] is renscfact=0.5d0 facscfact=2d0 ',
    ' [3] is renscfact=1d0 facscfact=0.5d0 ',
    ' [4] is renscfact=1d0 facscfact=1d0 ',
    ' [5] is renscfact=1d0 facscfact=2d0 ',
    ' [6] is renscfact=2d0 facscfact=0.5d0 ',
    ' [7] is renscfact=2d0 facscfact=1d0 ',
    ' [8] is renscfact=2d0 facscfact=2d0 ']
    """

    nom = np.ones(len(weights.weight()))
    if isSyst != False:
        if len(lhe_weights) > 0:
            if len(lhe_weights[0]) == 9:
                # Relative to the nominal ([4], mu_R = mu_F = 1), which genWeight
                # already contains: the booked factor is 1 and each leg is
                # [k]/[4]. coffea's Weights takes weightUp/weightDown as the
                # ABSOLUTE factor and divides by the booked one, so booking
                # [4] as the factor would multiply the nominal by [4] once per
                # systematic and divide the legs by [4] twice.
                # "Up" is the larger scale, for all three: muR [7] (2, 1),
                # muF [5] (1, 2), muR_muF [8] (2, 2). Legs capped (LHE_RATIO_CAP).
                ref = lhe_weights[:, 4]

                def leg(k):
                    return _cap_lhe_ratio(lhe_weights[:, k] / ref)

                weights.add("scalevar_muR", nom, leg(7), leg(1))
                weights.add("scalevar_muF", nom, leg(5), leg(3))
                weights.add("scalevar_muR_muF", nom, leg(8), leg(0))
            elif len(lhe_weights[0]) == 8:
                # nominal member removed (e.g. TTLNu-1Jets): the 9-entry order without [4], ref = 1
                def leg8(k):
                    return _cap_lhe_ratio(lhe_weights[:, k])

                weights.add("scalevar_muR", nom, leg8(6), leg8(1))
                weights.add("scalevar_muF", nom, leg8(4), leg8(3))
                weights.add("scalevar_muR_muF", nom, leg8(7), leg8(0))
            elif len(lhe_weights[0]) > 1:
                print("Scale variation vector has length ", len(lhe_weights[0]))
        else:
            warnings.warn(
                "LHE scale variation weights are not available, put nominal weights"
            )
            weights.add("scalevar_muR", nom, nom, nom)
            weights.add("scalevar_muF", nom, nom, nom)
            weights.add("scalevar_muR_muF", nom, nom, nom)

    else:
        weights.add("scalevar_3pt", nom)


# JP calibration utility
class JPCalibHandler(object):
    def __init__(self, year, campaign, isRealData, dataset, isSyst=False):
        """
        A tool for calculating the track probability and jet probability
            campaign: campaign name
            isRealData: whether the dataset is real data
            dataset: dataset name from events.metadata["dataset"]
        """
        if "JPCalib" not in config[campaign]["default"].keys():
            templates = uproot.open(
                "src/BTVNanoCommissioning/data/JPCalib/Summer22Run3/calibeHistoWrite_MC2022_NANO130X_v2.root"
            )
        else:
            if isRealData:
                if isSyst is not False:
                    filename = config[campaign]["default"]["JPCalib"]["MC"]
                else:
                    filename = "default"
                    for key in config[campaign]["default"]["JPCalib"]:
                        if key in dataset:
                            filename = config[campaign]["default"]["JPCalib"][key]
                            break
                    if filename == "default":
                        raise ValueError(f"No JPCalib file found for dataset {dataset}")
            else:
                filename = config[campaign]["default"]["JPCalib"]["MC"]

            templates = uproot.open(
                f"src/BTVNanoCommissioning/data/JPCalib/{campaign}/{filename}"
            )
        self.ipsig_histo_val = np.array(
            [templates[f"histoCat{i}"].values() for i in range(10)]
        )
        self.ipsig_histo_tot = np.sum(self.ipsig_histo_val, axis=1)
        self.values_cumsum = np.cumsum(self.ipsig_histo_val[:, ::-1], axis=1)[:, ::-1]
        self.edges = templates["histoCat0"].axes[0].edges()

    def flatten(self, array):
        """
        Get the fully flattened array and its layout for each layer
        """
        layouts = []
        array_fl = array
        while str(ak.type(array_fl)).count("*") > 1:
            layouts.append(ak.num(array_fl))
            array_fl = ak.flatten(array_fl)
        return array_fl, layouts

    def unflatten(self, array_fl, layouts):
        """
        Recover a flattened array using the original layouts
        """

        array = array_fl
        for layout in layouts[::-1]:
            array = ak.unflatten(array, layout)
        return array

    def calc_track_proba(self, ipsig: ak.Array, cat: ak.Array):
        """
        Calculate the track probability from the integral of the track IPsig templates, given the IPsig and category.
        Reference code: https://github.com/cms-sw/cmssw/blob/CMSSW_13_0_X/RecoBTag/TrackProbability/src/HistogramProbabilityEstimator.cc
            ipsig: IP significance array
            cat: category array (0-9)
        """

        if ak.any(cat < 0) or ak.any(cat > 9):
            raise ValueError("Category out of range [0, 9]")

        # get the fully flattened array of the input while storing its layouts for later recovery
        ipsig_fl, layouts = self.flatten(ipsig)
        cat_fl = ak.flatten(cat, axis=None)

        # index of the IPsig bins
        ipsig_fl = abs(ipsig_fl)
        ipsig_fl_index = np.minimum(
            np.searchsorted(self.edges, ipsig_fl), self.ipsig_histo_val.shape[1] - 1
        )

        # retrieve the cumsum value (\int_{ipsig}^{inf} p(ipsig') d(ipsig')) from the correct template
        ipsig_cumsum_fl = self.values_cumsum[cat_fl, ipsig_fl_index]

        # calculate the track probability as (\int_{ipsig}^{inf} ..) / (\int_{0}^{inf} ..) * sign(IPsig)
        proba_fl = (ipsig_cumsum_fl / self.ipsig_histo_tot[cat_fl]) * np.sign(ipsig_fl)

        # recover the original layout
        proba = self.unflatten(proba_fl, layouts)
        return proba

    def calc_jet_proba(self, proba):
        # Calculate jet probability (JP)
        # according to jetProbability func in https://github.com/cms-sw/cmssw/blob/CMSSW_13_0_X/RecoBTag/ImpactParameter/interface/TemplatedJetProbabilityComputer.h

        # minium proba = 0.5%
        proba = np.maximum(proba, 0.005)  # dim: (evt, jet, trk)

        ntrk = ak.num(proba, axis=-1)  # dim: (evt, jet), the number of tracks in a jet
        prodproba_log = ak.sum(
            np.log(proba), axis=-1
        )  # dim: (evt, jet), the log(Π(proba)) of all tracks in a jet
        prodproba_log_m_log = ak.where(
            (ak.num(proba, axis=-1) >= 2) & (prodproba_log < 0),
            np.log(-prodproba_log),
            0,
        )  # log(-logΠ), if >=2 tracks in a jet

        # now calculating Σ_tr{0..N-1} ((-logΠ)^tr / tr!)
        trk_index = ak.local_index(proba)

        # Handle fully-empty track collections: ak.max(...) can return None
        # (e.g. no tracks pass JP selection in a chunk).
        max_val = ak.max(trk_index, axis=None)
        max_trk_index = 0 if max_val is None else int(ak.to_numpy(max_val))
        fact_array = ak.concatenate(
            [
                [1.0],
                np.arange(1, max(5, max_trk_index + 1), dtype=np.float64).cumprod(),
            ]
        )  # construct a factorial array
        trk_index_fl, _layouts = self.flatten(trk_index)
        lfact = self.unflatten(
            fact_array[trk_index_fl], _layouts
        )  # dim: (evt, jet, trk), nested factorial array given the track index

        prob = ak.sum(
            np.exp(trk_index * prodproba_log_m_log - np.log(lfact)), axis=-1
        )  # dim: (evt, jet), Σ_tr{0..N-1} ((-logΠ)^tr / tr!)

        prob_jet = np.minimum(
            np.exp(np.maximum(np.log(np.maximum(prob, 1e-30)) + prodproba_log, -30.0)),
            1.0,
        )  # dim: (evt, jet), calculating Π * Σ_tr{0..N-1} ((-logΠ)^tr / tr!)

        prob_jet = ak.where(prodproba_log < 0, prob_jet, 1.0)
        prob_jet = np.maximum(prob_jet, 1e-30)

        return prob_jet


def _get_jpcalib_handler(year, campaign, isRealData, dataset, isSyst):
    """Cached construction of JPCalibHandler.

    JPCalibHandler only reads its constructor args to pick a template file and
    precompute derived arrays (see class docstring); it has no other mutable
    state, so a single instance can be reused across every shift/chunk that
    shares the same (year, campaign, isRealData, dataset, isSyst).
    """
    cache_key = (year, campaign, isRealData, dataset, isSyst)
    if cache_key in _JPCALIB_CACHE:
        return _JPCALIB_CACHE[cache_key]

    handler = JPCalibHandler(year, campaign, isRealData, dataset, isSyst)
    _JPCALIB_CACHE[cache_key] = handler
    return handler


def common_shifts(self, events):
    """
    Apply common shifts to a events(mostly affect energy resolution/scale of objects).

    This function applies common shifts to the input DataFrame based on the specified shift type.
    It modifies the DataFrame in place to reflect the systematic variations/dedicated corrections.
    This includes JERC, Rochester (Run 2), muon scale & smearing (SS), and electron SS corrections.

    Scale/Resolution Corrections:
    Construct a shift list with tuples of (obj_dict, shift_name).
    These corrections are applied independently on all objects by updating the contents of the branch.
    Normally done before selection to apply updated objects.
    Uncertainties are handled by updating object collections with up/down variations.

    Example for Shift List:
    ```python
    # nominal correction
    shift = [({"Jet": jets, "MET": met, "Muon": muon, "Electron": ele}, None)]
    # add variations
    shifts += [
                    (
                        {
                            "Jet": jets.JES_Total.up,
                            "MET": met.JES_Total.up,
                        },
                        "JESUp",
                    )]
    shifts += [
                    (
                        {
                            "Jet": jets.JES_Total.down,
                            "MET": met.JES_Total.down,
                        },
                        "JESDown",
                    )]
    ```

    Different treatment for weights and scale/resolution shifts is necessary
    to ensure accurate corrections and uncertainties are applied to the data.

    Parameters:
    self (dict): The configuration dictionary from SF_map containing the scale factors and other settings.
    events (events): The input events containing the data to be shifted.

    Returns:
    pandas.DataFrame: The DataFrame with the applied systematic shifts.
    """

    isRealData = not hasattr(events, "genWeight")
    dataset = events.metadata["dataset"]

    shifts = []

    if "JME" in self.SF_map.keys():
        shifts = JME_shifts(
            shifts,
            self.SF_map,
            events,
            self._year,
            self._campaign,
            isRealData,
            self.isSyst,
        )
    else:
        ## Use PuppiMET if available (NanoAODv15), otherwise fall back to PFMET
        if hasattr(events, "PuppiMET"):
            shifts = [
                (
                    {
                        "Jet": events.Jet,
                        "MET": events.PuppiMET,
                    },
                    None,
                )
            ]
        else:
            shifts = [
                (
                    {
                        "Jet": events.Jet,
                        "MET": events.MET,
                    },
                    None,
                )
            ]

    if "roccor" in self.SF_map.keys():
        shifts = Roccor_shifts(shifts, self.SF_map, events, isRealData, False)
    elif "muonSS" in self.SF_map.keys():
        shifts = MUO_shifts(shifts, self.SF_map, events, isRealData, True)
    else:
        for shift in shifts:
            shift[0]["Muon"] = events.Muon

    if "electronSS" in self.SF_map.keys():
        shifts = EGM_shifts(shifts, self.SF_map, events, isRealData, True)
    else:
        for shift in shifts:
            shift[0]["Electron"] = events.Electron

    # Apply jet veto. Run 3 JME recipe: reject the event if any jet with pT > 15 GeV,
    # tightLepVeto ID and (chEmEF + neEmEF) < 0.9 lies in the map (nominal jets for all shifts)
    if "jetveto" in self.SF_map.keys():
        jet_veto = jetveto(events.Jet, self.SF_map) > 0
        if not self._campaign.endswith("-UL"):
            jets = shifts[0][0]["Jet"]
            jet_veto = (
                jet_veto
                & jet_id(
                    SimpleNamespace(Jet=jets), self._campaign, max_eta=5.2, min_pt=15
                )
                & ((jets.chEmEF + jets.neEmEF) < 0.9)
            )
        event_veto = ak.any(jet_veto, axis=1)
        vetoed_events = events[~event_veto]
        for collections, _ in shifts:
            for key in collections:
                collections[key] = collections[key][~event_veto]
    else:
        vetoed_events = events

    return vetoed_events, shifts


def generator_weights(
    pruned_ev, isSyst, ttbar_reweights="none", campaign=None, nlo=None
):
    """Every generator-level factor and variation: genWeight, PS, PDF, LHE scale,
    top pT, hdamp ML, b/c fragmentation and decay. Shared by `weight_manager`
    (selected events) and `reweighting` (all events of the chunk), so the sums
    that normalise the variations are built from exactly the same weights as the
    histograms. Detector-level SFs are NOT here: they are not renormalised."""
    weights = Weights(len(pruned_ev), storeIndividual=True)
    if "genWeight" in pruned_ev.fields:
        weights.add("genweight", pruned_ev.genWeight)
    runs_sumw = (
        get_runs_sumw(pruned_ev.metadata.get("filename"))
        if "genWeight" in pruned_ev.fields
        else {}
    )
    if "PSWeight" in pruned_ev.fields:
        # PS ISR/FSR weights
        add_ps_weight(weights, pruned_ev.PSWeight, isSyst, runs_sumw=runs_sumw)
    if "LHEPdfWeight" in pruned_ev.fields:
        add_pdf_weight(weights, pruned_ev.LHEPdfWeight, isSyst)
    if nlo is not None:
        add_nlo_reweight(weights, nlo, isSyst)
    elif "LHEScaleWeight" in pruned_ev.fields:
        add_scalevar_weight(weights, pruned_ev.LHEScaleWeight, isSyst)

    is_mc = "genWeight" in pruned_ev.fields
    has_genpart = "GenPart" in pruned_ev.fields
    dataset_name = pruned_ev.metadata.get("dataset", "")
    is_ttbar_mc = is_mc and has_genpart and ("TT" in dataset_name)

    if is_ttbar_mc:
        nom = _top_pt_weight_from_branch(pruned_ev, runs_sumw)
        if nom is None:
            nom = top_pT_reweighting(pruned_ev.GenPart)
    else:
        nom = ak.ones_like(weights.weight())

    if isSyst != False and is_ttbar_mc:
        weights.add(
            "ttbar_weight",
            nom,
            nom + np.abs(ak.ones_like(nom) - nom),
            nom - np.abs(ak.ones_like(nom) - nom),
        )
    else:
        weights.add("ttbar_weight", nom)

    # NLO EWK k-factor in LHE V pT for W+jets / DY (nominal = corrected; the variation is the
    # symmetrised "no correction", as for top pT). Generator-level, so `reweighting` keeps it
    # shape-only (scaleSumW). Restricted to the campaigns that asked for it.
    vpt_boson = None
    has_lhe_vpt = "LHE" in pruned_ev.fields and "Vpt" in pruned_ev.LHE.fields
    if is_mc and campaign in VPT_EWK_CAMPAIGNS and has_lhe_vpt:
        if dataset_name.startswith("Wto") and "Nu-" in dataset_name:
            vpt_boson = "W"
        elif dataset_name.startswith("DYto"):
            vpt_boson = "Z"
    if vpt_boson is not None:
        k = vpt_ewk_kfactor(ak.to_numpy(pruned_ev.LHE.Vpt), vpt_boson)
        name = f"{vpt_boson.lower()}pt_ewk"
        if isSyst != False:
            weights.add(name, k, k + np.abs(1.0 - k), k - np.abs(1.0 - k))
        else:
            weights.add(name, k)

    # Additional ttbar reweighting hooks (enabled via runner flag).
    # Apply only to MC ttbar samples.
    if is_ttbar_mc and ttbar_reweights in ("hdamp_ml", "full"):
        add_hdamp_ml_weight(
            weights, pruned_ev, campaign=campaign, isSyst=isSyst, runs_sumw=runs_sumw
        )
    has_frag_decay_branches = (
        "bFragAndDecayWeight" in pruned_ev.fields
        and "cFragAndDecayWeight" in pruned_ev.fields
    )
    if is_mc and (
        has_frag_decay_branches or (is_ttbar_mc and ttbar_reweights == "full")
    ):
        add_fragmentation_decay_weights(
            weights,
            pruned_ev,
            isSyst=isSyst,
            runs_sumw=runs_sumw,
            genpart_fallback=(is_ttbar_mc and ttbar_reweights == "full"),
        )
    return weights


# Generator weights of all events of the current chunk (built in `reweighting`), reused by
# `weight_manager` for the selected events of every JES/JER shift instead of recomputing them.
_GEN_CACHE = {"chunk": None}


def _chunk_id(ev):
    m = ev.metadata
    return (
        m.get("dataset"),
        m.get("filename"),
        m.get("entrystart"),
        m.get("entrystop"),
    )


def _event_keys(ev):
    """uint64 key per event from (luminosityBlock, event); None if not unique-safe."""
    lumi = ak.to_numpy(ev.luminosityBlock).astype(np.uint64)
    evt = ak.to_numpy(ev.event).astype(np.uint64)
    run = ak.to_numpy(ev.run)
    if len(run) and (
        run.min() != run.max() or lumi.max() >= 2**30 or evt.max() >= 2**34
    ):
        return None
    return lumi << np.uint64(34) | evt


def _store_generator_weights(events, gen, args):
    _GEN_CACHE["chunk"] = None
    if not gen._storeIndividual or not isinstance(gen._weight, np.ndarray):
        return
    keys = _event_keys(events)
    if keys is None:
        return
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    if len(sk) > 1 and np.any(sk[1:] == sk[:-1]):
        return
    _GEN_CACHE.update(
        chunk=(_chunk_id(events), args), gen=gen, sorted_keys=sk, order=order
    )


def _cached_generator_weights(pruned_ev, args):
    """Rows of the chunk generator weights for the selected events; None if not available."""
    if _GEN_CACHE["chunk"] != (_chunk_id(pruned_ev), args):
        return None
    keys = _event_keys(pruned_ev)
    if keys is None:
        return None
    sk = _GEN_CACHE["sorted_keys"]
    pos = np.searchsorted(sk, keys)
    if len(keys) and (pos.max() >= len(sk) or np.any(sk[pos] != keys)):
        return None
    rows = _GEN_CACHE["order"][pos]
    gen = _GEN_CACHE["gen"]
    out = Weights(len(rows), storeIndividual=True)
    out._weight = gen._weight[rows]
    out._weights = {k: v[rows] for k, v in gen._weights.items()}
    out._modifiers = {k: v[rows] for k, v in gen._modifiers.items()}
    out._names = list(gen._names)
    for k, v in out._weights.items():  # as coffea Weights.add does
        if v.size:
            mn, mx = v.min(), v.max()
        elif np.issubdtype(v.dtype, np.integer):
            mn, mx = np.iinfo(v.dtype).max, np.iinfo(v.dtype).min
        else:
            mn, mx = np.inf, -np.inf
        out._weightStats[k] = WeightStatistics(v.sum(), (v**2).sum(), mn, mx, v.size)
    return out


# common weights
_STITCH = {}
_NLO_CACHE = {"chunk": None, "rows": {}}


def _nlo_process(dataset):
    if "amcatnloFXFX" not in dataset:
        return None
    if dataset.startswith("DYto2") and "MLL-10to50" not in dataset:
        return "dyll"
    if dataset.startswith(("WtoENu", "WtoMuNu", "WtoTauNu", "WtoLNu")):
        return "wlnu"
    return None


def nlo_reweight(pruned_ev):
    """PRISM-PCA negative-weight reweighting of the FxFx W/DY (MLL>50) samples
    (runner.py --nlo-reweight; CMS AN-2026/042). Selected events only; the
    normalisation keeps the genWeight sumw. Cached by (lumi, event) so the model
    runs once per event of a chunk, not once per JES/JER shift."""
    if not os.environ.get("BTV_NLO_REWEIGHT") or "GenPart" not in pruned_ev.fields:
        return None
    process = _nlo_process(pruned_ev.metadata.get("dataset", ""))
    if process is None:
        return None
    from BTVNanoCommissioning.helpers.nlo_reweight import calculate_nlo_reweighting

    chunk = (
        pruned_ev.metadata.get("filename"),
        pruned_ev.metadata.get("entrystart"),
    )
    if _NLO_CACHE["chunk"] != chunk:
        _NLO_CACHE["chunk"], _NLO_CACHE["rows"] = chunk, {}
    rows = _NLO_CACHE["rows"]
    keys = list(
        zip(
            ak.to_numpy(pruned_ev.luminosityBlock).tolist(),
            ak.to_numpy(pruned_ev.event).tolist(),
        )
    )
    new = np.array([k not in rows for k in keys], dtype=bool)
    if new.any():
        data = importlib.resources.files("BTVNanoCommissioning.data") / "NLORW"
        r = calculate_nlo_reweighting(
            pruned_ev[new],
            model_path=str(data / "ensemble_batch64.onnx"),
            process=process,
            era="run3_24",  # Summer24 MC (also used for 2025)
            network_libdir=str(data),
        )
        for j, k in enumerate(np.array(keys, dtype=object)[new]):
            rows[tuple(k)] = (
                r["nloReweight"][j],
                r["nloReweightUp"][j],
                r["nloReweightDown"][j],
            )
    nom = np.array([rows[k][0] for k in keys], dtype=np.float64)
    up = np.array([rows[k][1] for k in keys], dtype=np.float64).reshape(-1, 14)
    down = np.array([rows[k][2] for k in keys], dtype=np.float64).reshape(-1, 14)
    return nom, up, down


def add_nlo_reweight(weights, nlo, isSyst):
    """Nominal factor (sign(genWeight) x (2 P_pos - 1)), the 10 bootstrap pairs
    (nlo_vjet_0..9, for the PCA in the fit) and the reweighted muR/muF variations,
    which replace the LHE scale weights for these samples."""
    nom, up, down = nlo
    nom = np.where(nom == 0, 1e-9, nom)  # coffea divides the variations by it
    if isSyst == False:
        weights.add("nlo_vjet", nom)
        return
    weights.add_multivariation(
        "nlo_vjet",
        nom,
        [str(i) for i in range(10)],
        [up[:, i] for i in range(10)],
        [down[:, i] for i in range(10)],
    )
    one = np.ones(len(nom))
    # ML on/off: Up = without the ML factor, Down = mirrored (factor applied twice)
    weights.add("nlo_ml_onoff", one, 1.0 / nom, nom)
    for name, i in (
        ("scalevar_muF", 10),
        ("scalevar_muR", 11),
        ("scalevar_muR_muF", 12),
    ):
        weights.add(
            name,
            one,
            _cap_lhe_ratio(up[:, i] / nom),
            _cap_lhe_ratio(down[:, i] / nom),
        )


def vjets_stitch(pruned_ev, weights):
    """LHE-level FxFx W/DY stitching factor (runner.py --vjets-stitch). Not in
    generator_weights on purpose: sumw must stay the plain genWeight sum."""
    path = os.environ.get("BTV_VJETS_STITCH")
    if not path or "LHE" not in pruned_ev.fields:
        return
    if not os.path.exists(
        path
    ):  # file name only: the copy in data/STITCH (condor sandbox)
        path = str(
            importlib.resources.files("BTVNanoCommissioning.data")
            / "STITCH"
            / os.path.basename(path)
        )
    if path not in _STITCH:
        corr = correctionlib_compat.from_file(path)["vjets_fxfx_stitch"]
        with open(path) as f:
            keys = {c["key"] for c in json.load(f)["corrections"][0]["data"]["content"]}
        _STITCH[path] = (corr, keys)
    corr, keys = _STITCH[path]
    dataset = pruned_ev.metadata.get("dataset", "")
    if dataset not in keys:
        return
    pid = abs(pruned_ev.LHEPart.pdgId)
    lep = ak.fill_none(
        ak.max(ak.where((pid == 11) | (pid == 13) | (pid == 15), pid, 0), axis=1), 0
    )
    weights.add(
        "vjets_stitch",
        corr.evaluate(
            dataset,
            np.minimum(ak.to_numpy(pruned_ev.LHE.NpNLO), 3).astype(int),
            ak.to_numpy(lep).astype(int),
            ak.to_numpy(pruned_ev.LHE.Vpt).astype(float),
        ),
    )


def weight_manager(pruned_ev, SF_map, isSyst, ttbar_reweights="none", campaign=None):
    """
    Example for Scaling Factors (SFs):
    ```python
    # evaluation depends on file types...
    ## add SFs & uncertainties to weight function
    weights.add(sf.split(" ")[0], sfs_alle, sfs_alle_up, sfs_alle_down)
    """
    nlo = nlo_reweight(pruned_ev)
    weights = (
        None
        if nlo is not None
        else _cached_generator_weights(pruned_ev, (isSyst, ttbar_reweights, campaign))
    )
    if weights is None:
        weights = generator_weights(
            pruned_ev, isSyst, ttbar_reweights, campaign, nlo=nlo
        )
    vjets_stitch(pruned_ev, weights)

    if "hadronFlavour" in pruned_ev.Jet.fields:
        syst_wei = True if isSyst != False else False
        if "LUM" in SF_map.keys():
            puwei(
                pruned_ev.Pileup.nTrueInt,
                SF_map,
                weights,
                syst_wei,
            )
        if "MUO" in SF_map.keys() and "SelMuon" in pruned_ev.fields:
            muSFs(pruned_ev.SelMuon, SF_map, weights, syst_wei, True)
        if "EGM" in SF_map.keys() and "SelElectron" in pruned_ev.fields:
            eleSFs(pruned_ev.SelElectron, SF_map, weights, syst_wei, True)
        if (
            "ctag" in SF_map.keys() or "btag" in SF_map.keys()
        ) and "SelJet" in pruned_ev.fields:
            btagSFs(pruned_ev, SF_map, weights, "UParTAK4BC", syst_wei)
            # btagSFs(pruned_ev, SF_map, weights, "PNetBC", syst_wei)
            # btagSFs(pruned_ev, SF_map, weights, "DeepJetC", syst_wei)
            # btagSFs(pruned_ev, SF_map, weights, "DeepJetB", syst_wei)
            # btagSFs(pruned_ev, SF_map, weights, "DeepCSVB", syst_wei)
            # btagSFs(pruned_ev, SF_map, weights, "DeepCSVC", syst_wei)

    return weights


def reweighting(events, isSyst, ttbar_reweights="none", campaign=None):
    """Sums of weights over ALL events of the chunk (before any selection).

    `sumw` is the sum of the full generator-level NOMINAL weight (genWeight x top
    pT x b/c fragmentation central x ...); `sumw_<v>` the same sum with
    generator-level variation `v` applied, for every variation that
    `generator_weights` books. Flat keys, not a nested dict: xs_scaler detects
    merged outputs by dict depth. `scaleSumW` divides each variation by its own
    sum, so every theory systematic is acceptance-only relative to the nominal
    actually used.

    Normalising each factor on its own (per-file Runs sums against genWeight
    alone) is NOT enough: the factors are correlated. Measured on TTto2L2Nu, the
    b-fragmentation central weight (x_b dependent) against the FSR variations
    (which move x_b): PS_FSR_muR_4 down changed the inclusive yield by +4.9 %,
    UEPS_FSR down by +1.9 %, PS_FSR_x2xg_muR down by +1.6 %, cfragPeterson up by
    +0.2 %, although each factor alone averages to 1.0000."""
    sumws = {}
    if "genWeight" not in events.fields:
        sumws["sumw"] = len(events)
        return sumws
    gen = generator_weights(events, isSyst, ttbar_reweights, campaign)
    _store_generator_weights(events, gen, (isSyst, ttbar_reweights, campaign))
    sumws["sumw"] = float(np.sum(np.asarray(gen.weight(), dtype=np.float64)))
    if isSyst != False:
        for v in gen.variations:
            sumws[f"sumw_{v}"] = float(
                np.sum(np.asarray(gen.weight(v), dtype=np.float64))
            )
    return sumws
