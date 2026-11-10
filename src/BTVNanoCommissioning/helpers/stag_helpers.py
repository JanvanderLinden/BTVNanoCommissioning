"""
Helpers for the s-jet calibration workflows (``stag_*``).

Everything in this module operates on plain numpy arrays (plus a few thin awkward
wrappers) so that it is independent of the awkward / coffea version and can be unit
tested without a NanoAOD file.

Contents
--------
- ``STAG_CONFIG``                  per-campaign settings (triggers, lepton SFs, external files, cuts)
- ``TaggingBin``                   s-tagging WP bins and scores from the sfs-3d WP json
- ``add_stag_fields``              s-tag bin, scores, discriminants and flavour of jets
- ``spanet_inputs``/``onnx_feed``  padded SPANet inputs
- ``extract_best_assignment``      SPANet greedy best-overall jet assignment
- ``SpanetEvaluator``              lazy ONNX runtime wrapper
- ``gen_ttsl_partons``/``match_targets``  truth targets for SPANet training
- ``reconstruct_ttsl``             reconstructed objects / variables from the assignment
- ``TTSL_CATEGORIES``              event categories
"""

from __future__ import annotations

import os

import numpy as np
import awkward as ak

# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------

_DATA_DIR = "src/BTVNanoCommissioning/data/BTV"

# Per-campaign configuration. Paths can be overridden with environment variables:
#   STAG_TAGGING_WPS    -> sfs-3d WP json ("tagging_bin", "score_<axis>")
#   STAG_SPANET_MODEL -> SPANet ONNX model
_RUN3_2024 = {
    # single-lepton triggers per channel
    "triggers": {"mu": ["IsoMu24"], "e": ["Ele30_WPTight_Gsf"]},
    # correction_config entries (AK4_parameters) with the lepton SFs of the
    # prompt-MVA lepton selection, per channel
    "lepton_sf_selmod": {"mu": "semittM_2D", "e": "semittE_2D"},
    # sfs-3d WP json (wps.json of B_derive_wps.sh)
    "tagging_wps": f"{_DATA_DIR}/Summer24/stag/stag-wp.json",
    # SPANet ONNX model(s): one file, or the k-fold models in fold order (model k
    # evaluates the events with event % k_folds == k)
    "spanet_model": [
        f"{_DATA_DIR}/Summer24/stag/spanet_model_fold0.onnx",
        f"{_DATA_DIR}/Summer24/stag/spanet_model_fold1.onnx",
    ],
    # cut on the (raw, un-doubled) SPANet W assignment probability -> pass_spanet_w
    "spanet_w_score_cut": 0.3,
    # tagID bins (0-7: BT, BL, CT, CL, L, SL, SM, ST) that count as b tagged (applied
    # to the assigned b jets) and as c tagged (applied to the W daughters)
    "btag_bins": [0],
    "ctag_bins": [2],
    # event selection: minimum PuppiMET pT and W transverse mass (0: no cut)
    "met_pt_min": 0.0,
    "mtw_min": 0.0,
}
# private NanoAODv15 tuples (CustomSummer24 MC, CustomSummer25 data): NoPIDParTAK4 WPs
_RUN3_2024_NOPID = dict(
    _RUN3_2024,
    # B_derive_wps.sh with wp/configs/stag_2024_UParTNoPID.yml: pt-inclusive WPs, svudg
    # reweighted (ud/g)
    tagging_wps=f"{_DATA_DIR}/CustomSummer24/stag/wp_v1_ptincl_rew_svudg.json",
)
STAG_CONFIG = {
    "Summer24": _RUN3_2024,
    "CustomSummer24": _RUN3_2024_NOPID,
    "CustomSummer24UParTPaper": _RUN3_2024,
    # 2025: same triggers; WPs and SPANet model of 2024
    "CustomSummer25": _RUN3_2024_NOPID,
    "Prompt25": _RUN3_2024,
}


def get_stag_config(campaign: str) -> dict:
    if campaign not in STAG_CONFIG:
        raise KeyError(
            f"No s-tagging (stag) configuration for campaign '{campaign}'. "
            f"Available: {list(STAG_CONFIG)} -- add one to "
            "BTVNanoCommissioning/helpers/stag_helpers.py:STAG_CONFIG"
        )
    cfg = dict(STAG_CONFIG[campaign])
    for key, env in (("met_pt_min", "STAG_MET_PT_MIN"), ("mtw_min", "STAG_MTW_MIN")):
        if env in os.environ:
            cfg[key] = float(os.environ[env])
    cfg["tagging_wps"] = os.environ.get("STAG_TAGGING_WPS", cfg.get("tagging_wps"))
    cfg["spanet_model"] = os.environ.get("STAG_SPANET_MODEL", cfg.get("spanet_model"))
    if isinstance(cfg["spanet_model"], str) and "," in cfg["spanet_model"]:
        cfg["spanet_model"] = [f for f in cfg["spanet_model"].split(",") if f]
    if "STAG_SPANET_W_SCORE_CUT" in os.environ:
        cfg["spanet_w_score_cut"] = float(os.environ["STAG_SPANET_W_SCORE_CUT"])
    for key, env in (("btag_bins", "STAG_BTAG_BINS"), ("ctag_bins", "STAG_CTAG_BINS")):
        if env in os.environ:
            cfg[key] = parse_tag_bins(os.environ[env])
        cfg[key] = [int(b) for b in cfg[key]]
    return cfg


def parse_tag_bins(spec: str) -> list:
    """
    Parse a comma-separated list of tagID bins, given as numbers or WP names,
    e.g. "0", "BT,BL" or "2,3".
    """
    bins = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in TAG_BIN_NAMES:
            bins.append(TAG_BIN_NAMES.index(tok))
        elif tok.isdigit():
            bins.append(int(tok))
        else:
            bins.append(-1)
    if not bins or any(b < 0 or b >= len(TAG_BIN_NAMES) for b in bins):
        raise ValueError(
            f"stag: invalid tag bin specification '{spec}' (use bin numbers 0-7 or "
            f"names {TAG_BIN_NAMES})"
        )
    return bins


# --------------------------------------------------------------------------------------
# small numpy / awkward utilities
# --------------------------------------------------------------------------------------


def pad_np(jagged, n: int, fill=0.0, dtype=np.float32) -> np.ndarray:
    """Pad/clip a jagged [event, obj] array to a regular (n_events, n) numpy array."""
    padded = ak.fill_none(ak.pad_none(jagged, n, axis=1, clip=True), fill)
    out = ak.to_numpy(padded)
    if out.ndim != 2:  # empty input
        out = np.zeros((len(jagged), n))
    return np.asarray(out, dtype=dtype)


def first_np(jagged, fill=0.0, dtype=np.float32) -> np.ndarray:
    return np.asarray(ak.to_numpy(ak.fill_none(ak.firsts(jagged), fill)), dtype=dtype)


def delta_phi(phi1, phi2):
    dphi = np.abs(phi1 - phi2)
    return np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)


def p4(pt, eta, phi, mass):
    """(px, py, pz, E) from pt/eta/phi/mass numpy arrays."""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(np.maximum(px**2 + py**2 + pz**2 + mass**2, 0.0))
    return px, py, pz, e


def p4_mass(px, py, pz, e):
    return np.sqrt(np.maximum(e**2 - px**2 - py**2 - pz**2, 0.0))


def p4_pt(px, py, *_):
    return np.sqrt(px**2 + py**2)


def custom_flav(parton_flavour):
    """partonFlavour with gluons (21) mapped to 0."""
    return np.where(parton_flavour == 21, 0, parton_flavour)


def flav_class(flav):
    """
    Coarse flavour class of a (signed, custom_flav) parton flavour:
    0 = g/unmatched, 1 = u/d, 3 = s, 4 = c, 5 = b
    """
    a = np.abs(flav)
    return np.where(a == 2, 1, np.where(a > 5, 0, a))


# --------------------------------------------------------------------------------------
# s-tagging working points
# --------------------------------------------------------------------------------------

TAG_BIN_NAMES = ["BT", "BL", "CT", "CL", "L", "SL", "SM", "ST"]

# Inputs of the WP files of the earlier UParT derivation (scores as inputs), keyed by
# correction input name. The sfs-3d WP files take the raw NanoAOD Jet branches instead
# (input name = branch name) and compute the scores inside the correction.
LEGACY_INPUTS = {
    "bvall": "btagUParTAK4B",
    "cvnotb": "btagUParTAK4CvNotB",
    "svudg": "btagUParTAK4SvUDG",
}

# s-tagging jet flavours (as in the sfs-3d WP derivation), stagFlav = index
STAG_FLAVOURS = ["b", "bb", "c", "cc", "s", "ud", "g", "PU"]

# NoPIDParTAK4 class probabilities and the discriminants built from them
NOPID_PROBS = ["probb", "probbb", "probc", "probcc", "probs", "probud", "probg"]
NOPID_DISCRIMINANTS = ["BvAll", "CvNotB", "SvUDG", "SvUD"]


class TaggingBin:
    """
    s-tagging WP bin per jet (0-7: BT, BL, CT, CL, L, SL, SM, ST) and the scores of the
    cascade, from the WP json of the sfs-3d derivation (``wps.json``):

    - ``tagging_bin``(pt, raw tagger outputs...) -> bin index
    - ``score_<axis>``(raw tagger outputs...) -> score (reweighted where the WPs are)

    The correction inputs are NanoAOD Jet branch names (``pt``,
    ``btagNoPIDParTAK4probb``, ...). Older WP files with the scores as inputs
    (``bvall``, ``cvnotb``, ``svudg``) are read with ``LEGACY_INPUTS``.
    Picklable: the correctionlib objects are loaded lazily.
    """

    def __init__(self, wps_path: str):
        self.wps_path = wps_path
        self._cset = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_cset"] = None
        return state

    def _load(self):
        if self._cset is not None:
            return
        import correctionlib

        if self.wps_path is None or not os.path.exists(self.wps_path):
            raise FileNotFoundError(
                f"stag: tagging WP file not found at '{self.wps_path}'. Provide the "
                "sfs-3d wps.json there or point the STAG_TAGGING_WPS environment "
                "variable at it."
            )
        self._cset = correctionlib.CorrectionSet.from_file(self.wps_path)
        self.corrector = self._cset["tagging_bin"]
        self.input_names = [i.name for i in self.corrector.inputs]
        self.score_correctors = {
            name[len("score_") :]: self._cset[name]
            for name in self._cset.keys()
            if name.startswith("score_")
        }
        legacy = [n for n in self.input_names if n in LEGACY_INPUTS]
        if legacy and not self.wps_path.endswith(".gz"):
            # score inputs are only meaningful as the UParT discriminants
            import json

            with open(self.wps_path) as fh:
                desc = json.load(fh).get("description", "") or ""
            if "UParT" not in desc and ("NoPID" in desc or "ParTAK4" in desc):
                raise ValueError(
                    f"stag: '{self.wps_path}' takes the scores {legacy} as inputs and was "
                    f"not derived for the UParT discriminants ({desc[:80]}...). Regenerate "
                    "it with the current sfs-3d derive_wps.py (raw tagger outputs as inputs)."
                )

    def check(self):
        """Load the file once (on the submission side), then drop it again."""
        self._load()
        self._cset = None

    @property
    def branches(self) -> list:
        """Jet branches needed by ``tagging_bin`` (incl. pt)."""
        self._load()
        return [LEGACY_INPUTS.get(n, n) for n in self.input_names]

    @property
    def axes(self) -> list:
        self._load()
        if self.score_correctors:
            return list(self.score_correctors)
        return [n for n in self.input_names if n in LEGACY_INPUTS]

    def _args(self, fields, names):
        return [
            np.asarray(fields[LEGACY_INPUTS.get(n, n)], dtype=np.float64) for n in names
        ]

    def __call__(self, fields: dict) -> np.ndarray:
        """Bin index from flat numpy arrays keyed by Jet branch name."""
        self._load()
        if len(fields["pt"]) == 0:
            return np.zeros(0, dtype=np.int32)
        args = self._args(fields, self.input_names)
        return np.asarray(self.corrector.evaluate(*args)).astype(np.int32)

    def scores(self, fields: dict) -> dict:
        """Scores of the cascade {axis: array} from flat arrays keyed by branch name."""
        self._load()
        out = {}
        for axis in self.axes:
            if axis in self.score_correctors:
                corr = self.score_correctors[axis]
                args = self._args(fields, [i.name for i in corr.inputs])
                if len(fields["pt"]) == 0:
                    out[axis] = np.zeros(0)
                else:
                    out[axis] = np.asarray(corr.evaluate(*args), dtype=np.float64)
            else:
                out[axis] = np.asarray(fields[LEGACY_INPUTS[axis]], dtype=np.float64)
        return out


def _flat(jets):
    """(flat jets, counts or None) for a jagged or a one-per-event jet collection."""
    if jets.ndim > 1:
        return ak.flatten(jets), ak.to_numpy(ak.num(jets, axis=1))
    return jets, None


def _unflat(values, counts):
    return values if counts is None else ak.unflatten(values, counts)


def jet_tagging_bin(jets, tagging: TaggingBin):
    """``TaggingBin`` for a coffea Jet collection (jagged or one jet per event)."""
    missing = set(tagging.branches) - set(jets.fields)
    if missing:
        raise KeyError(f"stag: jet branches missing from input: {sorted(missing)}")
    flat, counts = _flat(jets)
    fields = {b: ak.to_numpy(flat[b]) for b in tagging.branches}
    return _unflat(tagging(fields), counts)


def nopid_discriminants(p: dict) -> dict:
    """NoPIDParTAK4 discriminants from the class probabilities (unreweighted)."""
    b, c = p["probb"] + p["probbb"], p["probc"] + p["probcc"]
    s, ud, g = p["probs"], p["probud"], p["probg"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {
            "BvAll": b / (b + c + s + ud + g),
            "CvNotB": c / (c + s + ud + g),
            "SvUDG": s / (s + ud + g),
            "SvUD": s / (s + ud),
        }
    return {k: np.nan_to_num(v, nan=-1.0) for k, v in out.items()}


def stag_jet_fields(jets, tagging: TaggingBin, isRealData: bool) -> dict:
    """
    s-tagging fields of a coffea Jet collection (jagged or one jet per event), keyed by
    field name: ``stagBin``, ``stagScore_<axis>``, the NoPIDParTAK4 discriminants
    ``stagBvAll``, ``stagCvNotB``, ``stagSvUDG``, ``stagSvUD`` (if the class probabilities
    are present) and, for MC, ``stagFlav`` (``stag_flavour``).
    """
    from BTVNanoCommissioning.utils.selection import stagflav

    flat, counts = _flat(jets)
    names = set(tagging.branches)
    for corr in tagging.score_correctors.values():
        names |= {i.name for i in corr.inputs}
    missing = {LEGACY_INPUTS.get(n, n) for n in names} - set(jets.fields)
    if missing:
        raise KeyError(f"stag: jet branches missing from input: {sorted(missing)}")
    fields = {LEGACY_INPUTS.get(n, n): ak.to_numpy(flat[LEGACY_INPUTS.get(n, n)]) for n in names}
    out = {"stagBin": _unflat(tagging(fields), counts)}
    for axis, v in tagging.scores(fields).items():
        out[f"stagScore_{axis}"] = _unflat(v, counts)
    probs = [f"btagNoPIDParTAK4{p}" for p in NOPID_PROBS]
    if set(probs) <= set(jets.fields):
        p = {k: ak.to_numpy(flat[b]).astype(np.float64) for k, b in zip(NOPID_PROBS, probs)}
        for k, v in nopid_discriminants(p).items():
            out[f"stag{k}"] = _unflat(v, counts)
    if not isRealData and "nBHadrons" in jets.fields:
        out["stagFlav"] = stagflav(jets)
    return out


def add_stag_fields(pruned_ev, collection: str, tagging: TaggingBin, isRealData: bool):
    """Add the ``stag_jet_fields`` to ``pruned_ev[collection]`` (in place)."""
    for name, values in stag_jet_fields(
        pruned_ev[collection], tagging, isRealData
    ).items():
        pruned_ev[collection, name] = values


# --------------------------------------------------------------------------------------
# SPANet inputs
# --------------------------------------------------------------------------------------

# number of jets each event is padded/truncated to for the SPANet sequential jet input
MAX_JETS = 6
# deltaR threshold for truth <-> reco jet matching
MATCH_DR = 0.4

# event-particle names, in the order declared in the SPANet event_info.yaml
EVENT_PARTICLES = ("lt", "htb", "W")
MOMENTA_FEATURES = ("mass", "pt", "eta", "sin_phi", "cos_phi")
LEPTON_FEATURES = ("pt", "eta", "sin_phi", "cos_phi")
MET_FEATURES = ("pt", "sin_phi", "cos_phi")
TARGET_NAMES = ("blep", "bhad", "q1", "q2")


def spanet_inputs(jet: dict, lepton: dict, met: dict) -> dict:
    """
    Build the SPANet input arrays.

    :param jet: dict of (n_events, MAX_JETS) numpy arrays ``pt, eta, phi, mass`` (zero padded,
        pt sorted) and the boolean ``mask``.
    :param lepton: dict of (n_events,) arrays ``pt, eta, phi``.
    :param met: dict of (n_events,) arrays ``pt, phi``.
    :returns: dict with the flat feature columns ``SpanetJet_{pt,eta,sin_phi,cos_phi,mass,mask}``,
        ``spanet_lepton_*``, ``spanet_met_*``).
    """
    mask = jet["mask"].astype(bool)
    out = {
        "SpanetJet_pt": np.where(mask, jet["pt"], 0.0).astype(np.float32),
        "SpanetJet_eta": np.where(mask, jet["eta"], 0.0).astype(np.float32),
        "SpanetJet_sin_phi": np.where(mask, np.sin(jet["phi"]), 0.0).astype(np.float32),
        "SpanetJet_cos_phi": np.where(mask, np.cos(jet["phi"]), 1.0).astype(np.float32),
        "SpanetJet_mass": np.where(mask, jet["mass"], 0.0).astype(np.float32),
        "SpanetJet_mask": mask,
        "spanet_lepton_pt": lepton["pt"].astype(np.float32),
        "spanet_lepton_eta": lepton["eta"].astype(np.float32),
        "spanet_lepton_sin_phi": np.sin(lepton["phi"]).astype(np.float32),
        "spanet_lepton_cos_phi": np.cos(lepton["phi"]).astype(np.float32),
        "spanet_met_pt": met["pt"].astype(np.float32),
        "spanet_met_sin_phi": np.sin(met["phi"]).astype(np.float32),
        "spanet_met_cos_phi": np.cos(met["phi"]).astype(np.float32),
    }
    # padded jets have phi = 0 -> sin = 0, cos = 1
    return out


def onnx_feed(inputs: dict) -> dict:
    """Arrange ``spanet_inputs`` output into the ONNX graph inputs (feature order matters)."""
    n = len(inputs["spanet_lepton_pt"])
    momenta = np.stack(
        [inputs[f"SpanetJet_{f}"] for f in MOMENTA_FEATURES], axis=-1
    ).astype(np.float32)
    lep = np.stack(
        [inputs[f"spanet_lepton_{f}"] for f in LEPTON_FEATURES], axis=-1
    ).astype(np.float32)[:, None, :]
    met = np.stack([inputs[f"spanet_met_{f}"] for f in MET_FEATURES], axis=-1).astype(
        np.float32
    )[:, None, :]
    return {
        "Momenta_data": momenta,
        "Momenta_mask": inputs["SpanetJet_mask"].astype(bool),
        "Lepton_data": lep,
        "Lepton_mask": np.ones((n, 1), dtype=bool),
        "Met_data": met,
        "Met_mask": np.ones((n, 1), dtype=bool),
    }


def extract_best_assignment(lt_prob, htb_prob, w_prob) -> dict:
    """
    Vectorized reimplementation of SPANet's greedy "best overall assignment" selection
    (``spanet.network.prediction_selection.extract_predictions``).

    :param lt_prob: assignment probabilities, shape ``(n_events, n_jets)``.
    :param htb_prob: same shape as ``lt_prob``.
    :param w_prob: shape ``(n_events, n_jets, n_jets)`` (symmetric in the last two axes).
    :returns: dict of per-event arrays ``lt_idx, htb_idx, w_idx1, w_idx2`` (-1 if not
        resolved) and ``lt_prob, htb_prob, w_prob`` (raw value at the chosen point).
    """
    n_events, n_jets = lt_prob.shape
    lt = np.asarray(lt_prob, dtype=np.float64).copy()
    htb = np.asarray(htb_prob, dtype=np.float64).copy()
    w = np.asarray(w_prob, dtype=np.float64).reshape(n_events, n_jets * n_jets).copy()

    EXCLUDED = -1.0
    ev = np.arange(n_events)

    result = {
        "lt_idx": np.full(n_events, -1, dtype=np.int64),
        "htb_idx": np.full(n_events, -1, dtype=np.int64),
        "w_idx1": np.full(n_events, -1, dtype=np.int64),
        "w_idx2": np.full(n_events, -1, dtype=np.int64),
        "lt_prob": np.zeros(n_events, dtype=np.float64),
        "htb_prob": np.zeros(n_events, dtype=np.float64),
        "w_prob": np.zeros(n_events, dtype=np.float64),
    }
    if n_events == 0:
        return result

    for _round in range(len(EVENT_PARTICLES)):
        lt_best_val, lt_best_idx = lt.max(axis=1), lt.argmax(axis=1)
        htb_best_val, htb_best_idx = htb.max(axis=1), htb.argmax(axis=1)
        w_best_val, w_best_idx = w.max(axis=1), w.argmax(axis=1)

        # the symmetric W head spreads each pair over two mirrored entries -> compare 2x
        w_best_val_scaled = w_best_val * 2.0

        stacked = np.stack([lt_best_val, htb_best_val, w_best_val_scaled], axis=1)
        winner = np.argmax(stacked, axis=1)
        winner_val = stacked[ev, winner]

        active = winner_val > EXCLUDED
        is_lt = active & (winner == 0)
        is_htb = active & (winner == 1)
        is_w = active & (winner == 2)

        jet1 = np.full(n_events, -1, dtype=np.int64)
        jet2 = np.full(n_events, -1, dtype=np.int64)
        jet1[is_lt] = lt_best_idx[is_lt]
        jet1[is_htb] = htb_best_idx[is_htb]
        jet1[is_w] = w_best_idx[is_w] // n_jets
        jet2[is_w] = w_best_idx[is_w] % n_jets

        result["lt_idx"][is_lt] = jet1[is_lt]
        result["lt_prob"][is_lt] = lt_best_val[is_lt]
        result["htb_idx"][is_htb] = jet1[is_htb]
        result["htb_prob"][is_htb] = htb_best_val[is_htb]
        result["w_idx1"][is_w] = jet1[is_w]
        result["w_idx2"][is_w] = jet2[is_w]
        result["w_prob"][is_w] = w_best_val[is_w]

        lt[is_lt, :] = EXCLUDED
        htb[is_htb, :] = EXCLUDED
        w[is_w, :] = EXCLUDED

        w_view = w.reshape(n_events, n_jets, n_jets)
        if active.any():
            lt[active, jet1[active]] = EXCLUDED
            htb[active, jet1[active]] = EXCLUDED
            w_view[active, jet1[active], :] = EXCLUDED
            w_view[active, :, jet1[active]] = EXCLUDED
        if is_w.any():
            lt[is_w, jet2[is_w]] = EXCLUDED
            htb[is_w, jet2[is_w]] = EXCLUDED
            w_view[is_w, jet2[is_w], :] = EXCLUDED
            w_view[is_w, :, jet2[is_w]] = EXCLUDED

    return result


class SpanetModel:
    """
    Thin, picklable wrapper around an onnxruntime session (the session itself is created
    lazily on the worker and never pickled).
    """

    def __init__(self, model_file: str, batch_size: int = 50000):
        self.model_file = model_file
        self.batch_size = batch_size
        self._session = None
        self._output_names = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_session"] = None
        state["_output_names"] = None
        return state

    def _load(self):
        if self._session is not None:
            return
        if self.model_file is None or not os.path.exists(self.model_file):
            raise FileNotFoundError(
                f"stag: SPANet ONNX model not found at '{self.model_file}'. Provide the "
                "model there or point the STAG_SPANET_MODEL environment variable "
                "at it (comma-separated for k-fold models). To only produce SPANet "
                "training inputs use the "
                "'stag_{mu,e}_tt1L_s_spanet_data' workflows."
            )
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            self.model_file, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._output_names = [o.name for o in self._session.get_outputs()]
        expected = {f"{p}_assignment_probability" for p in EVENT_PARTICLES} | {
            f"{p}_detection_probability" for p in EVENT_PARTICLES
        }
        missing = expected - set(self._output_names)
        if missing:
            raise RuntimeError(
                f"ONNX model {self.model_file} misses output(s) {sorted(missing)} -- was it "
                "exported from an event_info.yaml with event-particles lt/htb/W?"
            )
        input_names = {i.name for i in self._session.get_inputs()}
        expected_in = {
            "Momenta_data",
            "Momenta_mask",
            "Lepton_data",
            "Lepton_mask",
            "Met_data",
            "Met_mask",
        }
        if not expected_in.issubset(input_names):
            raise RuntimeError(
                f"ONNX model {self.model_file} has inputs {sorted(input_names)}, expected "
                f"{sorted(expected_in)}"
            )
        self._input_names = input_names

    def __call__(self, feed: dict) -> dict:
        """Run the model in batches; returns dict output_name -> numpy array."""
        self._load()
        n = len(feed["Momenta_data"])
        outs = {name: [] for name in self._output_names}
        for start in range(0, max(n, 1), self.batch_size):
            stop = min(start + self.batch_size, n)
            if stop <= start:
                break
            batch = {
                k: v[start:stop] for k, v in feed.items() if k in self._input_names
            }
            res = self._session.run(None, batch)
            for name, arr in zip(self._output_names, res):
                outs[name].append(np.asarray(arr))
        return {
            k: (np.concatenate(v, axis=0) if len(v) else np.zeros((0,)))
            for k, v in outs.items()
        }


class SpanetEvaluator:
    """
    SPANet evaluation with one model or k-fold models.

    ``model_files``: one ONNX file, or a list of N files in fold order. With N models,
    an event with ``event % N == k`` is evaluated with model k, which was trained on the
    events of the other folds (sfs-3d ``spanet/make_dataset.py --folds N``).
    """

    def __init__(self, model_files, batch_size: int = 50000):
        if isinstance(model_files, (str, os.PathLike)) or model_files is None:
            model_files = [model_files]
        self.model_files = list(model_files)
        self.models = [SpanetModel(f, batch_size) for f in self.model_files]

    @property
    def n_folds(self) -> int:
        return len(self.models)

    def check(self):
        """Load every model once (file, input and output names), then drop the sessions."""
        for m in self.models:
            m._load()
            m._session = None
            m._output_names = None

    def __call__(self, feed: dict, event=None) -> dict:
        """Outputs per event; ``event`` (event numbers) selects the fold model."""
        if self.n_folds == 1:
            return self.models[0](feed)
        if event is None:
            raise ValueError("stag: k-fold SPANet evaluation needs the event numbers")
        fold = np.asarray(event, dtype=np.uint64) % np.uint64(self.n_folds)
        order, results = [], []
        for k, model in enumerate(self.models):
            idx = np.flatnonzero(fold == np.uint64(k))
            if len(idx) == 0:
                continue
            results.append(model({key: v[idx] for key, v in feed.items()}))
            order.append(idx)
        if not results:
            return self.models[0](feed)
        order = np.concatenate(order)
        out = {}
        for name in results[0]:
            values = np.concatenate([r[name] for r in results], axis=0)
            arr = np.empty_like(values)
            arr[order] = values
            out[name] = arr
        return out


# --------------------------------------------------------------------------------------
# truth information for SPANet targets
# --------------------------------------------------------------------------------------


def gen_ttsl_partons(genpart) -> dict:
    """
    Identify the hard-process partons of a semileptonic ttbar decay.

    Returns per-event records (option type, ``None`` for non-semileptonic events) under the
    keys ``blep``, ``bhad``, ``q1`` (up-type quark from the hadronic W) and ``q2``
    (down-type quark), plus the boolean numpy array ``is_sl``.
    Taus count as leptons.
    """
    hard = genpart[genpart.hasFlags("isHardProcess")]
    pdg = hard.pdgId
    parent_pdg = abs(ak.fill_none(hard.distinctParent.pdgId, 0))

    w_dau = hard[parent_pdg == 24]
    abs_dau = abs(w_dau.pdgId)
    quarks = w_dau[abs_dau <= 5]
    leptons = w_dau[(abs_dau == 11) | (abs_dau == 13) | (abs_dau == 15)]
    b_from_t = hard[(abs(pdg) == 5) & (parent_pdg == 6)]

    is_sl = (
        (ak.num(quarks, axis=1) == 2)
        & (ak.num(leptons, axis=1) == 1)
        & (ak.num(b_from_t, axis=1) == 2)
    )

    lep_sign = np.sign(ak.fill_none(ak.firsts(leptons.pdgId), 0))
    # l- (pdg > 0) comes from a W-, i.e. from the anti-top -> leptonic b is a b-bar (-5)
    blep = ak.firsts(b_from_t[b_from_t.pdgId == -5 * lep_sign])
    bhad = ak.firsts(b_from_t[b_from_t.pdgId == 5 * lep_sign])
    q1 = ak.firsts(quarks[abs(quarks.pdgId) % 2 == 0])
    q2 = ak.firsts(quarks[abs(quarks.pdgId) % 2 == 1])

    is_sl_np = np.asarray(ak.to_numpy(is_sl), dtype=bool)
    out = {"is_sl": is_sl_np}
    for name, parton in (("blep", blep), ("bhad", bhad), ("q1", q1), ("q2", q2)):
        out[name] = {
            "eta": np.asarray(
                ak.to_numpy(ak.fill_none(parton.eta, -999.0)), dtype=np.float64
            ),
            "phi": np.asarray(
                ak.to_numpy(ak.fill_none(parton.phi, 0.0)), dtype=np.float64
            ),
        }
        out[name]["valid"] = is_sl_np & (out[name]["eta"] > -998.0)
    return out


def match_targets(
    jet_eta, jet_phi, jet_valid, partons: dict, max_jets=MAX_JETS, dr_max=MATCH_DR
):
    """
    Truth assignment targets: index of the closest jet (dR < dr_max) to each parton among
    *all* selected jets, -1 if none or if beyond the ``max_jets`` truncation. Targets that
    resolve to the same jet are invalidated (only the colliding ones).

    :param jet_eta, jet_phi, jet_valid: (n_events, n_max) numpy arrays of the selected,
        pt-sorted jets (padded; ``jet_valid`` marks real jets).
    :returns: dict ``spanet_target_{blep,bhad,q1,q2}`` -> int32 arrays.
    """
    idxs, founds = {}, {}
    for name in TARGET_NAMES:
        par = partons[name]
        deta = jet_eta - par["eta"][:, None]
        dphi = delta_phi(jet_phi, par["phi"][:, None])
        dr = np.sqrt(deta**2 + dphi**2)
        dr = np.where(jet_valid, dr, np.inf)
        if dr.shape[1] == 0:
            idx = np.full(len(dr), -1)
            min_dr = np.full(len(dr), np.inf)
        else:
            idx = np.argmin(dr, axis=1)
            min_dr = dr[np.arange(len(dr)), idx]
        found = par["valid"] & (min_dr < dr_max) & (idx < max_jets)
        idxs[name], founds[name] = idx, found

    collision = {n: np.zeros(len(jet_eta), dtype=bool) for n in TARGET_NAMES}
    for i, ni in enumerate(TARGET_NAMES):
        for nj in TARGET_NAMES[i + 1 :]:
            same = (idxs[ni] == idxs[nj]) & founds[ni] & founds[nj]
            collision[ni] |= same
            collision[nj] |= same

    return {
        f"spanet_target_{n}": np.where(founds[n] & ~collision[n], idxs[n], -1).astype(
            np.int32
        )
        for n in TARGET_NAMES
    }


# --------------------------------------------------------------------------------------
# reconstruction from the SPANet assignment
# --------------------------------------------------------------------------------------


def reconstruct_ttsl(
    jets: dict,
    assignment: dict,
    outputs: dict,
    w_score_cut: float,
    is_mc: bool,
    btag_bins=(0,),
    ctag_bins=(2,),
) -> dict:
    """
    Define the reconstructed-object variables from the SPANet assignment.

    :param jets: dict of (n_events, MAX_JETS) numpy arrays ``pt, eta, phi, mass, tagID``
        (+ ``flav`` for MC, already customFlav'ed).
    :param assignment: output of ``extract_best_assignment``.
    :param outputs: raw ONNX outputs (for the detection probabilities).
    :returns: flat dict of per-event numpy arrays.
    """
    n = len(assignment["lt_idx"])
    ev = np.arange(n)

    def gather(field, idx):
        return jets[field][ev, np.where(idx >= 0, idx, 0)]

    def vec(idx):
        return p4(
            gather("pt", idx),
            gather("eta", idx),
            gather("phi", idx),
            gather("mass", idx),
        )

    out = {}
    lt_idx, htb_idx = assignment["lt_idx"], assignment["htb_idx"]
    has_full = (lt_idx >= 0) & (htb_idx >= 0) & (assignment["w_idx1"] >= 0)
    out["spanet_has_full_assignment"] = has_full
    out["spanet_lt_assignment_prob"] = assignment["lt_prob"]
    out["spanet_htb_assignment_prob"] = assignment["htb_prob"]
    out["spanet_w_assignment_prob"] = assignment["w_prob"]
    for p, key in (("lt", "lt"), ("htb", "htb"), ("w", "W")):
        out[f"spanet_{p}_detection_prob"] = np.asarray(
            outputs[f"{key}_detection_probability"], dtype=np.float64
        ).reshape(n)
    out["pass_spanet_w"] = has_full & (assignment["w_prob"] > w_score_cut)

    # b jets
    blep, bhad = vec(lt_idx), vec(htb_idx)
    out["blep_tagID"] = gather("tagID", lt_idx)
    out["bhad_tagID"] = gather("tagID", htb_idx)
    out["blep_pt"], out["bhad_pt"] = gather("pt", lt_idx), gather("pt", htb_idx)
    out["spanet_lt_jet_idx"] = lt_idx.astype(np.int32)
    out["spanet_htb_jet_idx"] = htb_idx.astype(np.int32)
    blep_btag = np.isin(out["blep_tagID"], btag_bins)
    bhad_btag = np.isin(out["bhad_tagID"], btag_bins)
    out["bb_tagged"] = has_full & blep_btag & bhad_btag
    out["bhad_tagged"] = has_full & bhad_btag
    out["blep_tagged"] = has_full & blep_btag
    out["dphi_bb"] = delta_phi(gather("phi", htb_idx), gather("phi", lt_idx))
    out["bb_ptRel"] = out["bhad_pt"] / (out["blep_pt"] + 1e-10)
    bb = tuple(a + b for a, b in zip(bhad, blep))
    out["bb_m"] = p4_mass(*bb)
    out["bb_pt"] = p4_pt(*bb)
    out["_blep_p4"] = blep  # used for lepton-b variables, not histogrammed directly

    # W decay jets, pt sorted (q1 = leading)
    w1, w2 = assignment["w_idx1"], assignment["w_idx2"]
    lead1 = gather("pt", w1) >= gather("pt", w2)
    q1_idx = np.where(lead1, w1, w2)
    q2_idx = np.where(lead1, w2, w1)
    q1, q2 = vec(q1_idx), vec(q2_idx)
    whad = tuple(a + b for a, b in zip(q1, q2))
    out["spanet_whadq1_jet_idx"] = q1_idx.astype(np.int32)
    out["spanet_whadq2_jet_idx"] = q2_idx.astype(np.int32)
    out["whad_m"] = p4_mass(*whad)
    out["whad_pt"] = p4_pt(*whad)
    out["whad_m_diff"] = np.abs(out["whad_m"] - 80.4)
    out["whadq1_pt"], out["whadq2_pt"] = gather("pt", q1_idx), gather("pt", q2_idx)
    out["whadq1_tagID"], out["whadq2_tagID"] = gather("tagID", q1_idx), gather(
        "tagID", q2_idx
    )
    out["dphi_whad"] = delta_phi(gather("phi", q1_idx), gather("phi", q2_idx))
    out["whad_ptRel"] = out["whadq1_pt"] / (out["whadq2_pt"] + 1e-10)

    out["wq1_ctagged"] = np.isin(out["whadq1_tagID"], ctag_bins)
    out["wq2_ctagged"] = np.isin(out["whadq2_tagID"], ctag_bins)
    # cJet: the c-tagged W daughter (q1 if both/neither are), sJet: the other one
    use_q2_as_c = out["wq2_ctagged"] & ~out["wq1_ctagged"]
    wc_idx = np.where(use_q2_as_c, q2_idx, q1_idx)
    sig_idx = np.where(use_q2_as_c, q1_idx, q2_idx)
    for prefix, idx in (("cJet", wc_idx), ("sJet", sig_idx)):
        out[f"{prefix}_pt"] = gather("pt", idx)
        out[f"{prefix}_eta"] = gather("eta", idx)
        out[f"{prefix}_tagID"] = gather("tagID", idx)

    if is_mc:
        out["blep_flav"] = gather("flav", lt_idx)
        out["bhad_flav"] = gather("flav", htb_idx)
        out["whadq1_flav"] = gather("flav", q1_idx)
        out["whadq2_flav"] = gather("flav", q2_idx)
        out["cJet_flav"] = gather("flav", wc_idx)
        out["sJet_flav"] = gather("flav", sig_idx)
    else:
        for k in ("blep", "bhad", "whadq1", "whadq2", "cJet", "sJet"):
            out[f"{k}_flav"] = np.zeros(n, dtype=np.int32)

    return out


# --------------------------------------------------------------------------------------
# categories (all on top of 1 lepton & >= 4 jets)
# --------------------------------------------------------------------------------------


def _c(v):
    return v["wq1_ctagged"] | v["wq2_ctagged"]


# Categories (all on top of 1 lepton & >= 4 jets), ordered as a step-by-step purification
# chain: inclusive -> hadronic b tag -> + c-tagged W daughter -> + leptonic b tag, and the
# final W-quality cut (SPANet W detection probability > 0.5) on the s-enriched selections.
TTSL_CATEGORIES = {
    # inclusive control region: data/MC and SPANet output validation
    "incl_ttsl": lambda v: np.ones(len(v["n_jet"]), dtype=bool),
    # hadronic b tagged: removes the b from the W candidates on the hadronic side
    "incl_1bh_ttsl": lambda v: v["bhad_tagged"],
    # + c-tagged W daughter: W -> cs enrichment, sJet is s-enriched
    "incl_1bh1c_ttsl": lambda v: v["bhad_tagged"] & _c(v),
    # + leptonic b tagged
    "incl_2b1c_ttsl": lambda v: v["bb_tagged"] & _c(v),
    # nominal s-enriched signal region
    "wdet_1bh1c_ttsl": lambda v: v["bhad_tagged"]
    & _c(v)
    & (v["spanet_w_detection_prob"] > 0.5),
    # tighter variant (2 b tags) for purity vs statistics checks
    "wdet_2b1c_ttsl": lambda v: v["bb_tagged"]
    & _c(v)
    & (v["spanet_w_detection_prob"] > 0.5),
}
