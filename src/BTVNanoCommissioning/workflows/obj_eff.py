import importlib.resources
import json
import gzip
import os
import types

import awkward as ak
import correctionlib
import hist
import numpy as np
from coffea import processor
from coffea.analysis_tools import Weights
from coffea.lumi_tools import LumiMask

from BTVNanoCommissioning.helpers.func import dump_lumi, update
from BTVNanoCommissioning.helpers.update_branch import missing_branch
from BTVNanoCommissioning.utils.AK4_parameters import correction_config
from BTVNanoCommissioning.utils.correction import (
    common_shifts,
    generator_weights,
    jetveto,
    load_lumi,
    load_SF,
    muSFs,
    puwei,
    reweighting,
)
from BTVNanoCommissioning.utils.selection import (
    MET_filters,
    ele_promptmvaid,
    jet_id,
    mu_promptmvaid,
)

# Lepton tag-and-probe in Z->ll, Z counting per run and jet checks in Z events, all with
# loose selections. step axis: raw = generator weight, pu = + pileup,
# sf = + lepton SFs (tag: all; probe: the SFs of the levels it passes). Data: raw only.
# MC with --isSyst (e.g. JEC_total) adds sf_up/down (all lepton SFs together), pu_up/down,
# and jes_up/down, jer_up/down (jet histograms only, weight as sf).
# Events with a jet in the veto map are KEPT; the veto is an axis.

LSYST = ["sf_up", "sf_down", "pu_up", "pu_down", "prefire_up", "prefire_down"]
JSHIFT = {
    "JESTotalUp": "jes_up",
    "JESTotalDown": "jes_down",
    "JERTotalUp": "jer_up",
    "JERTotalDown": "jer_down",
}
# lepton energy scale / resolution (MC with --isSyst): TnP and Z-mass histograms only, weight as sf
LSHIFT = {
    "MuonScaleUp": "lscale_up",
    "MuonScaleDown": "lscale_down",
    "MuonResolUp": "lres_up",
    "MuonResolDown": "lres_down",
    "ElectronScaleUp": "lscale_up",
    "ElectronScaleDown": "lscale_down",
    "ElectronSmearUp": "lres_up",
    "ElectronSmearDown": "lres_down",
}
LKEYS = ["tnp", "tnp_map", "tnp_npv", "lep_var", "z_mass"]
STEPS = ["raw", "pu", "sf"] + LSYST + list(JSHIFT.values())
STEPS += ["lscale_up", "lscale_down", "lres_up", "lres_down"]


def sfw(s, wpu, st, sp, L):
    """Weight for step s: raw/pu event weight, else pileup weight x tag SF x probe SF."""
    if s in ("raw", "pu"):
        return wpu[s]
    pu = wpu[s] if s in wpu else wpu["pu"]
    v = s[2:] if s.startswith("sf_") else ""
    return pu * st["all" + v] * sp[L + v]


CH = {
    "M": dict(
        hlt="IsoMu24",
        pdg=13,
        bits=2 | 8,  # Iso, 1mu
        trig_pt=24.0,
        tag_pt=27.0,
        # mvaonly = MUO NUM_promptMVA_WP64ID_DEN_TightID numerator (no miniIso/sip3d/dxy/dz);
        # tightIso = TightID + PF iso, centrally calibrated in all years
        levels=["probe", "tightID", "promptMVA", "trig", "mvaonly", "tightIso"],
        eta=hist.axis.Regular(16, -2.4, 2.4, name="eta", label="probe #eta"),
        # cumulative probe SFs per level
        sfs={
            "probe": [],
            "tightID": ["mu_ID"],
            "promptMVA": ["mu_ID", "mu_Iso"],
            "mvaonly": ["mu_ID", "mu_Iso"],
            "tightIso": ["mu_ID"],
        },
        trig_sf="mu_Trig",
    ),
    "E": dict(
        hlt="Ele30_WPTight_Gsf",
        pdg=11,
        bits=2,  # 1e WPTight
        trig_pt=30.0,
        tag_pt=32.0,
        # mvaonly / presel / promptMVA_lh: parts of the promptMVA ID, to compare with EGM
        levels=[
            "probe",
            "wp80iso",
            "promptMVA",
            "trig",
            "mvaonly",
            "presel",
            "promptMVA_lh",
        ],
        eta=hist.axis.Variable(
            [-2.5, -2.0, -1.566, -1.4442, -0.8, 0, 0.8, 1.4442, 1.566, 2.0, 2.5],
            name="eta",
            label="probe SC #eta",
        ),
        # no wp80iso SF in the config: reco SF only
        sfs={
            "probe": ["ele_Reco"],
            "wp80iso": ["ele_Reco"],
            "promptMVA": ["ele_Reco", "ele_ID"],
            "mvaonly": ["ele_Reco"],
            "presel": ["ele_Reco"],
            "promptMVA_lh": ["ele_Reco", "ele_ID"],
        },
        trig_sf="ele_Trig",
    ),
}
RUN0 = {"2017": 297000, "2024": 378900, "2025": 391600}
MU_ISO = {
    "2017": "NUM_TightRelIso_DEN_TightIDandIPCut",
    "2024": "NUM_TightPFIso_DEN_TightID",
    "2025": "NUM_TightPFIso_DEN_TightID",
}
_EGM = "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM"
_E24 = f"{_EGM}/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/latest/electron.json.gz"
_H24 = f"{_EGM}/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/latest/electronHlt.json.gz"
_E25 = f"{_EGM}/Run3-25Prompt-Summer24-NanoAODv15/latest/electron.json.gz"
_E17 = f"{_EGM}/Run2-2017-UL-NanoAODv15/latest/electron.json.gz"
_I24, _I25, _I17 = (
    (_E24, "Electron-ID-SF", "2024Prompt"),
    (_E25, "Electron-ID-SF", "2025Prompt"),
    (_E17, "UL-Electron-ID-SF", "2017"),
)
_HLT24 = (_H24, "Electron-HLT-SF", "2024Prompt", "HLT_SF_Ele30_MVAiso80ID", 25, 1e9)
# electron SFs: name -> [(file, correction, year, working point / path, pT low, pT high)]
# 2025: no EGM PromptMVA or HLT SF and no reco SF below 20 GeV -> 2024 SF as stand-in.
# 2017: no EGM HLT SF.
ELE_SF = {
    "2024": {
        "reco": [
            (*_I24, "RecoBelow20", 0, 20),
            (*_I24, "Reco20to75", 20, 75),
            (*_I24, "RecoAbove75", 75, 1e9),
        ],
        "wp80": [(*_I24, "wp80iso", 0, 1e9)],
        "pmva": [(*_I24, "PromptMVA-Tight", 0, 1e9)],
        "hlt": [_HLT24],
    },
    "2025": {
        "reco": [
            (*_I24, "RecoBelow20", 0, 20),
            (*_I25, "Reco20to75", 20, 75),
            (*_I25, "RecoAbove75", 75, 1e9),
        ],
        "wp80": [(*_I25, "wp80iso", 0, 1e9)],
        "pmva": [(*_I24, "PromptMVA-Tight", 0, 1e9)],
        "hlt": [_HLT24],
    },
    "2017": {
        "reco": [(*_I17, "RecoBelow20", 0, 20), (*_I17, "RecoAbove20", 20, 1e9)],
        "wp80": [(*_I17, "wp80iso", 0, 1e9)],
    },
}


def pt_range(path, corr, keys):
    """(min, max) of the pt edges in a correctionlib JSON, below the given category keys."""
    node = next(c for c in json.load(gzip.open(path))["corrections"] if c["name"] == corr)["data"]
    lo, hi = np.inf, -np.inf
    stack = [node]
    while stack:
        n = stack.pop()
        if not isinstance(n, dict):
            continue
        if n.get("nodetype") == "category":
            sub = [x["value"] for x in n["content"] if str(x["key"]) in keys]
            stack += sub or [x["value"] for x in n["content"]]
        elif n.get("nodetype") in ("multibinning", "binning"):
            inp = n.get("inputs", [n.get("input")])
            edges = n["edges"] if n["nodetype"] == "multibinning" else [n["edges"]]
            if "pt" in inp:
                e = edges[inp.index("pt")]
                lo, hi = min(lo, float(e[0])), max(hi, float(e[-1]))
            stack += n["content"] if isinstance(n["content"], list) else [n["content"]]
    return lo, hi


# SF names per probe level (cumulative); "all" (tag) = "trig"
_LEV_RUN3 = {
    "probe": ["reco"],
    "wp80iso": ["reco", "wp80"],
    "promptMVA": ["reco", "pmva"],
    "mvaonly": ["reco", "pmva"],
    "presel": ["reco"],
    "promptMVA_lh": ["reco", "pmva"],
    "trig": ["reco", "pmva", "hlt"],
}
ELE_LEV = {
    "2024": _LEV_RUN3,
    "2025": _LEV_RUN3,
    # 2017 analysis ID (promptMVA level) = wp80iso; no promptMVA SF
    "2017": {
        **_LEV_RUN3,
        "promptMVA": ["reco", "wp80"],
        "mvaonly": ["reco"],
        "promptMVA_lh": ["reco", "wp80"],
        "trig": ["reco", "wp80"],
    },
}
# UL2017: unprescaled single-lepton triggers; analysis ID = tight ID + PF iso (mu), MVA wp80iso (e)
CH2017 = {
    "M": dict(hlt="IsoMu27", trig_pt=27.0, tag_pt=30.0),
    "E": dict(hlt="Ele35_WPTight_Gsf", trig_pt=35.0, tag_pt=38.0),
}

a_step = hist.axis.StrCategory(STEPS, name="step")
a_pt = hist.axis.Variable(
    [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 120, 200], name="pt", label="probe pT"
)
a_mass = hist.axis.Regular(30, 60, 120, name="mass", label="m(tag,probe)")
a_mass_pog = hist.axis.Regular(40, 60, 140, name="mass", label="m(tag,probe)")
a_phi = hist.axis.Regular(36, -np.pi, np.pi, name="phi")
a_npv = hist.axis.Regular(25, 0, 100, name="npv", label="good PV")
a_zpt = hist.axis.Variable(
    [0, 15, 20, 30, 40, 50, 70, 100, 150, 250, 500], name="zpt", label="Z pT"
)
a_bool = lambda n: hist.axis.Boolean(name=n)
a_npv5 = hist.axis.Variable([0, 25, 35, 45, 55, 100], name="npv", label="good PV")
# lepton ID input variables (probe, all tag-probe masses): value mapped to [0, 1] with (lo, hi)
LVAR = {
    "miniPFRelIso_all": (0, 0.4),
    "miniPFRelIso_chg": (0, 0.4),
    "miniIso_neu": (0, 0.4),
    "jetRelIso": (0, 2),
    "jetPtRelv2": (0, 20),
    "sip3d": (0, 8),
    "promptMVA": (-1, 1),
}
LVAR_CH = {
    "M": {"pfRelIso04_all": (0, 0.4)},
    "E": {"hoe": (0, 0.15), "mvaIso": (-1, 1)},
}
W = hist.storage.Weight()


def trig_match(obj, events, c):
    to = events.TrigObj
    to = to[(to.id == c["pdg"]) & ((to.filterBits & c["bits"]) == c["bits"])]
    to = to[to.pt > c["trig_pt"]]
    return ak.fill_none(ak.any(obj.metric_table(to) < 0.1, axis=2), False)


def jet_vars(jets, gen):
    """flat numpy arrays of the jet variables an SF can be binned in, and the gen match mask."""
    fj = ak.flatten(jets)
    var = {
        "pt": fj.pt,
        "abseta": abs(fj.eta),
        "qvg": fj.btagUParTAK4QvG,
        "nch": fj.chMultiplicity,
        "nneu": fj.nConstituents - fj.chMultiplicity,
    }
    gidx = jets.genJetIdx
    matched = (gidx >= 0) & (gidx < ak.num(gen))
    gm = gen[ak.mask(gidx, matched)]
    var["ptgen"] = ak.flatten(ak.fill_none(gm.pt, -1))
    var["absetagen"] = ak.flatten(ak.fill_none(abs(gm.eta), 99))
    return {k: ak.to_numpy(v) for k, v in var.items()}, matched


def keep_prob(d, var, gmask, gc):
    """p = min(SF, 1) for the jets in the SF's mask (gen-matched for "ptgen" SFs), else 1."""
    if d is None:
        return np.ones(len(gc))
    idx = tuple(
        np.clip(np.searchsorted(e, var[v], side="right") - 1, 0, len(e) - 2)
        for v, e in zip(d["vars"], d["edges"])
    )
    m = gmask if "ptgen" in d["vars"] else gc
    return np.where(m, np.minimum(np.asarray(d["sf"])[idx], 1), 1)


def pbinom(q):
    """Poisson-binomial P(k found) per event from padded keep probabilities q [events, K]."""
    P = np.zeros((q.shape[0], q.shape[1] + 1))
    P[:, 0] = 1
    for i in range(q.shape[1]):
        P[:, 1:] = P[:, 1:] * (1 - q[:, i, None]) + P[:, :-1] * q[:, i, None]
        P[:, 0] *= 1 - q[:, i]
    return P


def jet_sf_closure(out, path, jets, good_c, jid, gen, lepz, z, w):
    """test hook BTV_JET_SF=<json>: keep prob. p = min(SF, 1) per central Tight jet (other jets p = 1);
    an SF binned in "ptgen" applies to Tight jets matched to a gen jet with |eta_gen| < 2.5.
    Fills N jets (pT > 30) and the leading jet as event-weight expectations; no jet is removed.
    """
    n = np.asarray(ak.fill_none(ak.num(jets), 0))
    var, matched = jet_vars(jets, gen)
    gc = ak.to_numpy(ak.flatten(good_c))
    gmask = ak.to_numpy(ak.flatten(jid & matched)) & (var["absetagen"] < 2.5)
    genc = gen[
        ak.fill_none(
            ak.all(gen.metric_table(lepz) > 0.4, axis=2, mask_identity=True), True
        )
        & (abs(gen.eta) < 4.7)
    ]
    lg = ak.firsts(genc)
    out["gen_recoil"].fill(
        zpt=z.pt,
        ptgen=ak.fill_none(lg.pt, -1),
        absetagen=ak.fill_none(abs(lg.eta), 0),
        weight=w,
    )
    c30 = gc & (var["pt"] > 30)
    K = max(1, min(int(n.max(initial=0)), 12))  # ponytail: jets beyond the 12th ignored

    def pad(x):
        return ak.to_numpy(
            ak.fill_none(ak.pad_none(ak.unflatten(x, n), K, clip=True), 0)
        ).astype(float)

    zpt = np.asarray(ak.fill_none(z.pt, 0))
    ae = pad(var["abseta"])
    bal = np.where(
        pad(ak.to_numpy(ak.flatten(abs(jets.delta_phi(z)) > 2.7))) > 0,
        pad(var["pt"]) / zpt[:, None],
        -1.0,
    )
    for name, d in [("none", None)] + list(json.load(open(path)).items()):
        p = keep_prob(d, var, gmask, gc)
        tj = ak.to_numpy(ak.flatten(jid)) & (var["abseta"] < 2.5)
        out["clos_jet"].fill(
            sf=name,
            pt=var["pt"][tj],
            abseta=var["abseta"][tj],
            weight=(np.repeat(w, n) * p)[tj],
        )
        P = pbinom(pad(np.where(c30, p, 0)))
        for k in range(min(K + 1, 16)):
            out["clos_njet"].fill(sf=name, n=np.full(len(zpt), k), weight=w * P[:, k])
        pp = pad(p) * (np.arange(K)[None, :] < n[:, None])
        before = np.cumprod(np.c_[np.ones(len(zpt)), 1 - pp[:, :-1]], axis=1)
        # alpha cut: every found jet after the leading one has pT < max(15, 0.3 pT(Z))
        f = np.where(pad(var["pt"]) >= np.maximum(15, 0.3 * zpt)[:, None], 1 - pp, 1)
        after = np.c_[np.cumprod(f[:, ::-1], axis=1)[:, ::-1][:, 1:], np.ones(len(zpt))]
        for j in range(K):
            out["clos_lead"].fill(
                sf=name,
                zpt=zpt,
                abseta=ae[:, j],
                bal=bal[:, j],
                weight=w * pp[:, j] * before[:, j] * after[:, j],
            )
        out["clos_lead"].fill(
            sf=name,
            zpt=zpt,
            abseta=np.full(len(zpt), -1.0),
            bal=np.full(len(zpt), -1.0),
            weight=w * np.prod(1 - pp, axis=1),
        )


def emu_fill(out, jets, jid, gen, w, variants):
    """e-mu (ttbar) test: Tight jets, |eta| < 2.5, pT > 20, selection >= 1 such jet with pT > 50.
    Per SF variant the expected N(jets pT > 30) and jets per pT bin, given the selection
    (event weights from the keep probabilities, no jet removed). gen None (data): p = 1.
    """
    jets = jets[jid & (abs(jets.eta) < 2.5) & (jets.pt > 20)]
    n = np.asarray(ak.fill_none(ak.num(jets), 0))
    if gen is None:
        fj = ak.flatten(jets)
        var = {"pt": ak.to_numpy(fj.pt), "abseta": ak.to_numpy(abs(fj.eta))}
        gmask, flav = np.zeros(len(var["pt"]), bool), np.zeros(len(var["pt"]), int)
    else:
        var, matched = jet_vars(jets, gen)
        gmask = ak.to_numpy(ak.flatten(matched)) & (var["absetagen"] < 2.5)
        pf = np.abs(ak.to_numpy(ak.flatten(jets.partonFlavour)))
        flav = np.select(
            [pf == 5, pf == 4, pf == 21, (pf >= 1) & (pf <= 3)], [5, 4, 21, 1], 0
        )
    gc = np.ones(len(var["pt"]), bool)
    K = max(1, min(int(n.max(initial=0)), 12))  # ponytail: jets beyond the 12th ignored

    def pad(x):
        return ak.to_numpy(
            ak.fill_none(ak.pad_none(ak.unflatten(x, n), K, clip=True), 0)
        ).astype(float)

    hi, c30 = var["pt"] > 50, var["pt"] > 30
    for name, d in variants:
        p = keep_prob(d, var, gmask, gc)
        lost_hi = np.prod(
            1 - pad(np.where(hi, p, 0)), axis=1
        )  # all jets above 50 GeV lost
        P = pbinom(pad(np.where(c30, p, 0))) - lost_hi[:, None] * pbinom(
            pad(np.where(c30 & ~hi, p, 0))
        )
        for k in range(min(K + 1, 16)):
            out["emu_njet"].fill(sf=name, n=np.full(len(n), k), weight=w * P[:, k])
        sel = np.repeat(1 - lost_hi, n)
        out["emu_jet"].fill(
            sf=name,
            pt=var["pt"],
            abseta=var["abseta"],
            flav=flav,
            weight=np.repeat(w, n) * p * np.where(hi, 1, sel),
        )


class NanoProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year="2024",
        campaign="Summer24",
        name="",
        isSyst=False,
        isArray=False,
        noHist=False,
        chunksize=75000,
        ttbar_reweights="none",
        selectionModifier="M",
    ):
        self._year = year
        self._campaign = campaign
        self.isSyst = isSyst
        self.ch = selectionModifier
        self.c = {**CH[self.ch], **(CH2017[self.ch] if year == "2017" else {})}
        if year == "2017":
            with importlib.resources.path(
                "BTVNanoCommissioning.data.DC",
                "Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt",
            ) as fn:
                self.lumiMask = LumiMask(fn)
        else:
            self.lumiMask = load_lumi(campaign)
        selmod = "ObjEff" + selectionModifier
        if selmod not in correction_config.get(campaign, {}):
            selmod = "WcM_2D" if selectionModifier == "M" else "WcE_2D"
        self.SF_map = load_SF(year, campaign, selmod)
        # test hook BTV_PU_FILE=<json, may contain {year}>: other pileup weights (e.g. 75.3 mb)
        self.egm = (
            {
                f: correctionlib.CorrectionSet.from_file(f)
                for specs in ELE_SF[year].values()
                for f, *_ in specs
            }
            if selectionModifier == "E"
            else {}
        )
        self.egm_pt = {
            (f, corr, wp): pt_range(f, corr, {yr, wp})
            for specs in (ELE_SF[year].values() if selectionModifier == "E" else [])
            for f, corr, yr, wp, *_ in specs
        }
        # own attribute: SF_map is rebuilt from the config when pickled to workers
        self.pu_alt = (
            {"LUM": correctionlib.CorrectionSet.from_file(os.environ["BTV_PU_FILE"].format(year=year))}
            if os.environ.get("BTV_PU_FILE")
            else None
        )

    def process(self, events):
        events = missing_branch(events, f"{self._year}_{self._campaign}")
        sumws = reweighting(events, False, campaign=self._campaign)
        lumi = {}
        if not hasattr(events, "genWeight"):
            events = events[self.lumiMask(events.run, events.luminosityBlock)]
            lumi = dump_lumi(events, {})
        # cheap preselection before the corrections (they act object by object)
        lep = events.Muon if self.ch == "M" else events.Electron
        nlep = ak.sum(lep.pt > 9, axis=1) >= 2
        if self.ch == "M" and self._year != "2017":
            # e-mu events for the ttbar jet test
            nlep = nlep | (
                (ak.sum(events.Muon.pt > 9, axis=1) >= 1)
                & (ak.sum(events.Electron.pt > 9, axis=1) >= 1)
            )
        events = events[events.HLT[self.c["hlt"]] & nlep]
        # common_shifts without the event-level jet veto: the veto is studied here
        noveto = types.SimpleNamespace(
            SF_map={k: self.SF_map[k] for k in self.SF_map if k != "jetveto"},
            _year=self._year,
            _campaign=self._campaign,
            isSyst=self.isSyst if hasattr(events, "genWeight") else False,
        )
        veto_any = ak.any(jetveto(events.Jet, self.SF_map) > 0, axis=1)
        _, shifts = common_shifts(noveto, events)
        collections, _ = shifts[0]
        for coll, name in shifts[1:]:
            if name in JSHIFT:
                collections["Jet" + JSHIFT[name].replace("_", "")] = coll["Jet"]
        res = self.process_nominal(update(events, collections), sumws, veto_any, lumi)
        if self.isSyst is not False and hasattr(events, "genWeight"):
            pre = "Muon" if self.ch == "M" else "Electron"
            for coll, name in shifts[1:]:
                if name in LSHIFT and name.startswith(pre):
                    o = self.process_nominal(
                        update(events, coll), sumws, veto_any, lumi, LSHIFT[name]
                    )
                    for ds, h in o.items():
                        for k in LKEYS:
                            res[ds][k] += h[k]
        return res

    def process_nominal(self, events, sumws, veto_any, lumi, lshift=None):
        dataset = events.metadata["dataset"]
        isData = not hasattr(events, "genWeight")
        syst = not isData and self.isSyst is not False
        c = self.c
        lv = hist.axis.StrCategory(c["levels"], name="level")
        a_run = hist.axis.Regular(
            8100, RUN0[self._year], RUN0[self._year] + 8100, name="run"
        )
        out = {
            "sumw": sumws["sumw"],
            "tnp": hist.Hist(a_step, lv, a_pt, c["eta"], a_mass, storage=W),
            "tnp_map": hist.Hist(
                a_step,
                lv,
                hist.axis.Regular(50, -2.5, 2.5, name="eta"),
                a_phi,
                storage=W,
            ),
            "tnp_npv": hist.Hist(
                a_step,
                lv,
                a_npv,
                hist.axis.Variable([10, 20, 30, 50, 200], name="pt"),
                storage=W,
            ),
            "tnp_run": hist.Hist(lv, a_run, storage=W),
            "lep_var": hist.Hist(
                a_step,
                hist.axis.StrCategory([*LVAR, *LVAR_CH[self.ch]], name="var"),
                hist.axis.Variable([0, 1.479, 2.5], name="abseta"),
                hist.axis.Regular(40, 0, 1, name="value"),
                a_mass,
                storage=W,
            ),
            "jet_pv": hist.Hist(
                a_step,
                a_bool("jetid"),
                hist.axis.Variable([15, 20, 25, 30, 40, 50, 70, 100, 500], name="pt"),
                hist.axis.Variable([0, 1.3, 2.0, 2.5, 3.0, 4.7], name="abseta"),
                a_npv5,
                hist.axis.Variable(
                    [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0], name="nehef"
                ),
                storage=W,
            ),
            "bal_pv": hist.Hist(
                a_step,
                a_zpt,
                a_npv5,
                hist.axis.Variable([0, 1.3, 2.5, 4.7], name="abseta"),
                hist.axis.Regular(30, 0, 3, name="bal", label="pT(jet)/pT(Z)"),
                storage=W,
            ),
            "bal_nehef": hist.Hist(
                a_step,
                a_zpt,
                hist.axis.Variable([0, 1.3, 2.5], name="abseta"),
                hist.axis.Variable(
                    [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0], name="nehef"
                ),
                hist.axis.Regular(60, 0, 3, name="bal", label="pT(jet)/pT(Z)"),
                storage=W,
            ),
            "jet_tag": hist.Hist(
                a_step,
                hist.axis.IntCategory([0, 4, 5], name="flav"),
                hist.axis.Variable([20, 30, 50, 100, 500], name="pt"),
                hist.axis.Variable([0, 1.3, 2.5], name="abseta"),
                hist.axis.StrCategory(["btagUParTAK4B", "btagUParTAK4QvG"], name="var"),
                hist.axis.Regular(20, 0, 1, name="value"),
                hist.axis.Variable([0, 0.05, 0.15, 0.3, 1.0], name="nehef"),
                storage=W,
            ),
            "jet_sfvar": hist.Hist(
                a_step,
                hist.axis.Variable([20, 25, 30, 40, 50, 70, 100, 500], name="pt"),
                hist.axis.Variable([0, 1.3, 2.5], name="abseta"),
                hist.axis.Regular(10, 0, 1, name="qvg"),
                hist.axis.Variable([0, 3, 5, 7, 10, 15, 100], name="nch"),
                hist.axis.Variable([0, 2, 4, 6, 9, 13, 100], name="nneu"),
                storage=W,
            ),
            "jet_raw": hist.Hist(
                a_step,
                hist.axis.Variable(
                    [5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 70, 100, 500], name="ptraw"
                ),
                hist.axis.Variable([0, 1.3, 2.5], name="abseta"),
                hist.axis.Variable([0, 0.05, 0.1, 0.2, 0.3, 1.0], name="nehef"),
                storage=W,
            ),
            "gen_recoil": hist.Hist(
                a_zpt,
                hist.axis.Variable(
                    [0, 10, 15, 20, 25, 30, 40, 50, 70, 100, 150, 250, 500],
                    name="ptgen",
                ),
                hist.axis.Variable([0, 1.3, 2.5, 4.7], name="absetagen"),
                storage=W,
            ),
            "emu_njet": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="sf"),
                hist.axis.Integer(0, 16, name="n"),
                storage=W,
            ),
            "emu_jet": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="sf"),
                hist.axis.Variable(
                    [20, 25, 30, 40, 50, 70, 100, 150, 250, 500], name="pt"
                ),
                hist.axis.Variable([0, 1.3, 2.5], name="abseta"),
                hist.axis.IntCategory([0, 1, 4, 5, 21], name="flav"),
                storage=W,
            ),
            "clos_jet": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="sf"),
                hist.axis.Variable(
                    [15, 20, 25, 30, 40, 50, 70, 100, 150, 250, 500], name="pt"
                ),
                hist.axis.Variable([0, 1.3, 2.5], name="abseta"),
                storage=W,
            ),
            "clos_njet": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="sf"),
                hist.axis.Integer(0, 16, name="n"),
                storage=W,
            ),
            "clos_lead": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="sf"),
                a_zpt,
                hist.axis.Variable([0, 1.3, 2, 2.5, 3, 5], name="abseta"),
                hist.axis.Regular(60, 0, 3, name="bal"),
                storage=W,
            ),
            "tnp_alttag": hist.Hist(a_step, lv, a_pt, c["eta"], a_mass, storage=W),
            "tnp_pog": hist.Hist(a_step, lv, a_pt, c["eta"], a_mass_pog, storage=W),
            # MC only: tnp_pog with a prompt (genPartFlav == 1) probe, for MC eff. and templates
            "tnp_pog_gen": hist.Hist(a_step, lv, a_pt, c["eta"], a_mass_pog, storage=W),
            "z_run": hist.Hist(a_run, storage=W),
            "z": hist.Hist(
                a_step,
                a_bool("veto_any"),
                a_bool("veto_recipe"),
                a_zpt,
                hist.axis.Regular(10, -2.5, 2.5, name="zy"),
                a_npv,
                storage=W,
            ),
            "z_mass": hist.Hist(
                a_step, hist.axis.Regular(60, 60, 120, name="mass"), storage=W
            ),
            "evt": hist.Hist(
                a_step,
                hist.axis.StrCategory(["rho", "njet30", "met"], name="var"),
                hist.axis.Regular(100, 0, 100, name="value"),
                storage=W,
            ),
            "jet": hist.Hist(
                a_step,
                a_bool("jetid"),
                a_bool("inmap"),
                hist.axis.Variable(
                    [15, 20, 25, 30, 40, 50, 70, 100, 150, 250, 500], name="pt"
                ),
                hist.axis.Regular(47, -4.7, 4.7, name="eta"),
                storage=W,
            ),
            "jet_map": hist.Hist(
                a_step,
                a_bool("jetid"),
                hist.axis.Regular(50, -5, 5, name="eta"),
                a_phi,
                storage=W,
            ),
            "bal": hist.Hist(
                a_step,
                a_bool("jetid"),
                a_zpt,
                hist.axis.Variable([0, 1.3, 2.0, 2.5, 3.0, 5.0], name="abseta"),
                hist.axis.Regular(60, 0, 3, name="bal", label="pT(jet)/pT(Z)"),
                storage=W,
            ),
            "jet_frac": hist.Hist(
                a_step,
                hist.axis.StrCategory(
                    ["chHEF", "neHEF", "chEmEF", "neEmEF", "muEF"], name="var"
                ),
                hist.axis.Variable([20, 30, 50, 100, 1000], name="pt"),
                hist.axis.Variable([0, 1.3, 2.5], name="abseta"),
                hist.axis.Regular(50, 0, 1, name="value"),
                storage=W,
            ),
            "jet_count": hist.Hist(
                a_step,
                hist.axis.StrCategory(
                    ["nConstituents", "chMultiplicity", "nSVs"], name="var"
                ),
                hist.axis.Variable([20, 30, 50, 100, 1000], name="pt"),
                hist.axis.Variable([0, 1.3, 2.5], name="abseta"),
                hist.axis.Integer(0, 80, name="value"),
                storage=W,
            ),
        }

        # event preselection
        sel = MET_filters(events, self._campaign) & (events.PV.npvsGood > 0)
        out.update(lumi)
        events, veto_any = events[sel], veto_any[sel]

        # leptons
        if self.ch == "M":
            lep = events.Muon
            ana = (
                (lep.tightId & (lep.pfRelIso04_all < 0.15) & (abs(lep.eta) < 2.4))
                if self._year == "2017"
                else mu_promptmvaid(events)
            )
            probe = lep.isTracker & (lep.pt > 10) & (abs(lep.eta) < 2.4)
            pass_lv = {
                "tightID": lep.tightId,
                "promptMVA": ana,
                "mvaonly": lep.tightId & (lep.promptMVA > 0.64),
                "tightIso": lep.tightId & (lep.pfRelIso04_all < 0.15),
            }
            lep_eta = lep.eta
        else:
            lep = events.Electron
            ana = (
                (lep.mvaIso_WP80 & (abs(lep.eta + lep.deltaEtaSC) < 2.5))
                if self._year == "2017"
                else ele_promptmvaid(events, self._campaign)
            )
            lep_eta = lep.eta + lep.deltaEtaSC
            probe = (lep.pt > 10) & (abs(lep_eta) < 2.5)
            pass_lv = {
                "wp80iso": lep.mvaIso_WP80,
                "promptMVA": ana,
                "mvaonly": (lep.promptMVA >= 0.9) & (abs(lep.eta) < 2.5),
                "presel": (lep.miniPFRelIso_all < 0.4)
                & (lep.sip3d < 8)
                & (abs(lep.dxy) < 0.05)
                & (abs(lep.dz) < 0.1),
                "promptMVA_lh": ana & (lep.lostHits < 2),
            }
        matched = trig_match(lep, events, c)
        pass_lv["probe"] = probe
        pass_lv["trig"] = ana & matched
        lep = ak.with_field(lep, ak.local_index(lep.pt), "idx")
        lep = ak.with_field(lep, lep_eta, "eta_sel")
        for k, v in pass_lv.items():
            lep = ak.with_field(lep, v, f"p_{k}")
        tags = lep[ana & matched & (lep.pt > c["tag_pt"])]
        pairs = ak.cartesian({"t": tags, "p": lep[probe]})
        mll = (pairs.t + pairs.p).mass
        pairs = pairs[
            (pairs.t.idx != pairs.p.idx)
            & (pairs.t.charge != pairs.p.charge)
            & (mll > 60)
            & (mll < 120)
        ]
        # alternative tag (EGM: cut-based Tight; muons: tight ID + PF iso), for tnp_alttag only
        alt = (
            (lep.cutBased == 4)
            if self.ch == "E"
            else (lep.tightId & (lep.pfRelIso04_all < 0.15))
        )
        pairs_a = ak.cartesian(
            {"t": lep[alt & matched & (lep.pt > c["tag_pt"])], "p": lep[probe]}
        )
        mll_a = (pairs_a.t + pairs_a.p).mass
        pairs_a = pairs_a[
            (pairs_a.t.idx != pairs_a.p.idx)
            & (pairs_a.t.charge != pairs_a.p.charge)
            & (mll_a > 60)
            & (mll_a < 120)
        ]
        # POG-like TnP (tnp_pog): 60-140 GeV; muons: MUO tag iso < 0.2, |dxy| < 0.2, |dz| < 0.5 on tag and probe
        if self.ch == "M":
            ip = (abs(lep.dxy) < 0.2) & (abs(lep.dz) < 0.5)
            tag_g = lep.tightId & (lep.pfRelIso04_all < 0.2) & ip
            probe_g = probe & ip
        else:
            tag_g, probe_g = alt, probe
        pairs_g = ak.cartesian(
            {"t": lep[tag_g & matched & (lep.pt > c["tag_pt"])], "p": lep[probe_g]}
        )
        mll_g = (pairs_g.t + pairs_g.p).mass
        pairs_g = pairs_g[
            (pairs_g.t.idx != pairs_g.p.idx)
            & (pairs_g.t.charge != pairs_g.p.charge)
            & (mll_g > 60)
            & (mll_g < 140)
        ]

        # event weights: gen, gen x PU
        if isData:
            w_evt = {"raw": np.ones(len(events))}
        else:
            gw = generator_weights(events, False, "none", self._campaign)
            puwei(events.Pileup.nTrueInt, self.pu_alt or self.SF_map, gw, syst)
            if self._year == "2017":
                pf = events.L1PreFiringWeight
                gw.add("prefire", pf.Nom, pf.Up, pf.Dn)
            w_evt = {
                "raw": gw.partial_weight(exclude=["puweight", "prefire"]),
                "pu": gw.weight(),
            }
            if syst:
                w_evt["pu_up"] = gw.weight("puweightUp")
                w_evt["pu_down"] = gw.weight("puweightDown")
                pre = self._year == "2017"
                w_evt["prefire_up"] = gw.weight("prefireUp") if pre else w_evt["pu"]
                w_evt["prefire_down"] = gw.weight("prefireDown") if pre else w_evt["pu"]
        steps = ["raw"] if isData else ["raw", "pu", "sf"] + (LSYST if syst else [])
        if lshift:
            steps = [lshift]

        # --- e-mu (ttbar) jet test: one trigger-matched muon + one electron, opposite charge
        if self.ch == "M" and self._year != "2017" and not lshift:
            tm = lep[ana & matched & (lep.pt > c["tag_pt"])]
            el = events.Electron
            te = el[
                ele_promptmvaid(events, self._campaign)
                & (el.pt > 20)
                & (abs(el.eta + el.deltaEtaSC) < 2.5)
            ]
            m1, e1 = ak.firsts(tm), ak.firsts(te)
            ok = (ak.num(tm) == 1) & (ak.num(lep[ana]) == 1) & (ak.num(te) == 1)
            ok = ak.to_numpy(
                ak.fill_none(
                    ok & (m1.charge != e1.charge) & ((m1 + e1).mass > 20), False
                )
            )
            if ok.any():
                ev_e, m1, e1 = events[ok], m1[ok], e1[ok]
                we = (
                    np.ones(ok.sum())
                    if isData
                    else np.asarray(w_evt["pu"])[ok] * self.lep_sfs(m1)["all"]
                )
                je = ev_e.Jet
                for lo in (m1, e1):
                    je = je[
                        ak.fill_none(
                            ak.all(
                                je.metric_table(ak.singletons(lo)) > 0.4,
                                axis=2,
                                mask_identity=True,
                            ),
                            True,
                        )
                    ]
                jide = ak.fill_none(
                    jet_id(
                        types.SimpleNamespace(Jet=je),
                        self._campaign,
                        max_eta=5.2,
                        min_pt=0,
                    ),
                    False,
                )
                variants = [("none", None)]
                sf_file = os.environ.get("BTV_JET_SF")
                if sf_file and not isData:
                    variants += list(json.load(open(sf_file)).items())
                emu_fill(out, je, jide, None if isData else ev_e.GenJet, we, variants)

        # --- tag-and-probe: one entry per (tag, probe) pair
        npair = ak.num(pairs)
        t, p = ak.flatten(pairs.t), ak.flatten(pairs.p)
        mass = (t + p).mass
        npv = np.repeat(ak.to_numpy(events.PV.npvsGood), npair)
        run = np.repeat(ak.to_numpy(events.run), npair)
        w_pair = {s: np.repeat(np.asarray(w), npair) for s, w in w_evt.items()}
        sf_t, sf_p = (None, None) if isData else (self.lep_sfs(t), self.lep_sfs(p))
        for L in c["levels"]:
            m = ak.to_numpy(p[f"p_{L}"])
            for s in steps:
                w = sfw(s, w_pair, sf_t, sf_p, L)
                win = m & (mass > 81) & (mass < 101)
                out["tnp"].fill(
                    step=s,
                    level=L,
                    pt=p.pt[m],
                    eta=p.eta_sel[m],
                    mass=mass[m],
                    weight=w[m],
                )
                out["tnp_map"].fill(
                    step=s,
                    level=L,
                    eta=p.eta_sel[win & (p.pt > 20)],
                    phi=p.phi[win & (p.pt > 20)],
                    weight=w[win & (p.pt > 20)],
                )
                out["tnp_npv"].fill(
                    step=s, level=L, npv=npv[win], pt=p.pt[win], weight=w[win]
                )
            if isData:
                win = m & (mass > 81) & (mass < 101) & ak.to_numpy(p.pt > 20)
                out["tnp_run"].fill(level=L, run=run[win])
        pa = ak.flatten(pairs_a.p)
        mass_a = ak.flatten((pairs_a.t + pairs_a.p).mass)
        for s in [x for x in steps if x in ("raw", "pu")]:
            wa = np.repeat(np.asarray(w_evt[s]), ak.num(pairs_a))
            for L in c["levels"]:
                m = ak.to_numpy(pa[f"p_{L}"])
                out["tnp_alttag"].fill(
                    step=s,
                    level=L,
                    pt=pa.pt[m],
                    eta=pa.eta_sel[m],
                    mass=mass_a[m],
                    weight=wa[m],
                )
        pg = ak.flatten(pairs_g.p)
        mass_g = ak.flatten((pairs_g.t + pairs_g.p).mass)
        for s in [x for x in steps if x in ("raw", "pu")]:
            wg = np.repeat(np.asarray(w_evt[s]), ak.num(pairs_g))
            for L in c["levels"]:
                m = ak.to_numpy(pg[f"p_{L}"])
                out["tnp_pog"].fill(
                    step=s,
                    level=L,
                    pt=pg.pt[m],
                    eta=pg.eta_sel[m],
                    mass=mass_g[m],
                    weight=wg[m],
                )
                if not isData:
                    mg = m & ak.to_numpy(pg.genPartFlav == 1)
                    out["tnp_pog_gen"].fill(
                        step=s,
                        level=L,
                        pt=pg.pt[mg],
                        eta=pg.eta_sel[mg],
                        mass=mass_g[mg],
                        weight=wg[mg],
                    )
        pv = {k: p[k] for k in [*LVAR, *LVAR_CH[self.ch]] if k != "miniIso_neu"}
        pv["miniIso_neu"] = p.miniPFRelIso_all - p.miniPFRelIso_chg
        m20 = ak.to_numpy(p.pt > 20)
        for s in steps:
            w = sfw(s, w_pair, sf_t, sf_p, "probe")
            for k, (lo, hi) in {**LVAR, **LVAR_CH[self.ch]}.items():
                out["lep_var"].fill(
                    step=s,
                    var=k,
                    abseta=abs(p.eta_sel[m20]),
                    value=(pv[k][m20] - lo) / (hi - lo),
                    mass=mass[m20],
                    weight=w[m20],
                )

        # --- Z events: best pair with both legs at analysis ID (tag is trigger-matched)
        zp = pairs[pairs.p.p_promptMVA]
        best = ak.argmin(abs((zp.t + zp.p).mass - 91.19), axis=1, keepdims=True)
        zp = ak.firsts(zp[best])
        hasz = ~ak.is_none(zp)
        zp, ev, veto_any = zp[hasz], events[hasz], veto_any[hasz]
        z = zp.t + zp.p
        zpu = {s: np.asarray(w)[ak.to_numpy(hasz)] for s, w in w_evt.items()}
        zs_t, zs_p = (
            (None, None) if isData else (self.lep_sfs(zp.t), self.lep_sfs(zp.p))
        )
        zw = {s: sfw(s, zpu, zs_t, zs_p, "promptMVA") for s in steps}
        for s in steps:
            out["z_mass"].fill(step=s, mass=z.mass, weight=zw[s])
        if lshift:
            return {dataset: out}
        good = ak.to_numpy((z.mass > 76) & (z.mass < 106))
        zp, ev, veto_any, z = zp[good], ev[good], veto_any[good], z[good]
        zw = {s: w[good] for s, w in zw.items()}
        if isData:
            out["z_run"].fill(run=ev.run)

        # jets: clean against the two Z legs, pT > 15 GeV (JEC applied)
        jcols = [(ev.Jet, steps)]
        if syst:
            jcols += [(ev["Jet" + v.replace("_", "")], [v]) for v in JSHIFT.values()]
        nehef_k = float(os.environ.get("BTV_NEHEF_SCALE", "1"))
        for jets, jsteps in jcols:
            if nehef_k != 1 and not isData and jsteps is steps:
                # test hook BTV_NEHEF_SCALE=k: neutral-hadron energy x k in MC, mean response kept
                # (the data L2L3Res restores it). ponytail: fixed <neHEF> = 0.14, per-jet pT-dependence ignored
                r = 1 - jets.neHEF * (1 - nehef_k)
                jets = ak.with_field(
                    jets, jets.pt * r / (1 - 0.14 * (1 - nehef_k)), "pt"
                )
                for v in ["chHEF", "chEmEF", "neEmEF", "muEF"]:
                    jets = ak.with_field(jets, jets[v] / r, v)
                jets = ak.with_field(jets, jets.neHEF * nehef_k / r, "neHEF")
                jets = jets[ak.argsort(jets.pt, ascending=False)]
            lepz = ak.concatenate([ak.singletons(zp.t), ak.singletons(zp.p)], axis=1)
            clean = ak.fill_none(
                ak.all(jets.metric_table(lepz) > 0.4, axis=2, mask_identity=True), True
            )
            jid = ak.fill_none(
                jet_id(
                    types.SimpleNamespace(Jet=jets),
                    self._campaign,
                    max_eta=5.2,
                    min_pt=0,
                ),
                False,
            )
            inmap = jetveto(jets, self.SF_map) > 0
            # Run 3 JME recipe: pT > 15 GeV, tightLepVeto ID, EM fraction < 0.9
            rec = (jets.pt > 15) & jid & ((jets.chEmEF + jets.neEmEF) < 0.9) & inmap
            veto_recipe = ak.any(rec, axis=1)
            if jsteps is steps:
                # raw (uncorrected) pT of all stored jets, Tight ID, clean, |eta| < 2.5
                rj = jets[clean & jid & (abs(jets.eta) < 2.5)]
                wr = {
                    s: np.repeat(np.asarray(zw[s]), ak.to_numpy(ak.num(rj)))
                    for s in ("raw", "sf")
                    if s in zw
                }
                frj = ak.flatten(rj)
                for s_, w_ in wr.items():
                    out["jet_raw"].fill(
                        step=s_,
                        ptraw=frj.pt_raw,
                        abseta=abs(frj.eta),
                        nehef=frj.neHEF,
                        weight=w_,
                    )
            sel_j = clean & (jets.pt > 15) & (abs(jets.eta) < 4.7)
            jets, jid, inmap = jets[sel_j], jid[sel_j], inmap[sel_j]
            nj = ak.num(jets)
            good_c = jid & (jets.pt > 20) & (abs(jets.eta) < 2.5)
            bb = jets[good_c]
            dphi = abs(jets.delta_phi(z))
            lead = ak.firsts(jets)
            sub_pt = ak.fill_none(ak.pad_none(jets.pt, 2)[:, 1], 0)
            recoil = ak.fill_none(abs(lead.delta_phi(z)) > 2.7, False)
            alpha_ok = sub_pt < np.maximum(15, 0.3 * z.pt)
            bal = ak.where(recoil, ak.fill_none(lead.pt, 0) / z.pt, -1.0)
            ljid = ak.fill_none(ak.firsts(jid), False)
            sf_file = os.environ.get("BTV_JET_SF")
            if sf_file and not isData and jsteps is steps:
                jet_sf_closure(
                    out, sf_file, jets, good_c, jid, ev.GenJet, lepz, z, zw["sf"]
                )
            for s in jsteps:
                w = zw["sf" if s in JSHIFT.values() else s]
                wj = np.repeat(w, ak.to_numpy(nj))
                out["z"].fill(
                    step=s,
                    veto_any=veto_any,
                    veto_recipe=veto_recipe,
                    zpt=z.pt,
                    zy=0.5 * np.log((z.energy + z.pz) / (z.energy - z.pz)),
                    npv=ev.PV.npvsGood,
                    weight=w,
                )
                out["evt"].fill(
                    step=s, var="rho", value=ev.Rho.fixedGridRhoFastjetAll, weight=w
                )
                out["evt"].fill(
                    step=s,
                    var="njet30",
                    value=ak.sum(good_c & (jets.pt > 30), axis=1),
                    weight=w,
                )
                out["evt"].fill(step=s, var="met", value=ev.MET.pt, weight=w)
                fj = ak.flatten(jets)
                out["jet"].fill(
                    step=s,
                    jetid=ak.flatten(jid),
                    inmap=ak.flatten(inmap),
                    pt=fj.pt,
                    eta=fj.eta,
                    weight=wj,
                )
                m30 = ak.to_numpy(fj.pt > 30)
                out["jet_map"].fill(
                    step=s,
                    jetid=ak.flatten(jid)[m30],
                    eta=fj.eta[m30],
                    phi=fj.phi[m30],
                    weight=wj[m30],
                )
                a = ak.to_numpy(alpha_ok)
                out["bal_pv"].fill(
                    step=s,
                    zpt=z.pt[a],
                    npv=ev.PV.npvsGood[a],
                    abseta=ak.fill_none(abs(lead.eta), -1)[a],
                    bal=bal[a],
                    weight=w[a],
                )
                out["jet_pv"].fill(
                    step=s,
                    jetid=ak.flatten(jid),
                    pt=fj.pt,
                    abseta=abs(fj.eta),
                    npv=np.repeat(ak.to_numpy(ev.PV.npvsGood), ak.to_numpy(nj)),
                    nehef=fj.neHEF,
                    weight=wj,
                )
                out["bal_nehef"].fill(
                    step=s,
                    zpt=z.pt[a],
                    abseta=ak.fill_none(abs(lead.eta), -1)[a],
                    nehef=ak.fill_none(lead.neHEF, -1)[a],
                    bal=bal[a],
                    weight=w[a],
                )
                out["bal"].fill(
                    step=s,
                    jetid=ljid[a],
                    zpt=z.pt[a],
                    abseta=ak.fill_none(abs(lead.eta), -1)[a],
                    bal=bal[a],
                    weight=w[a],
                )
                fb = ak.flatten(bb)
                wb = np.repeat(w, ak.to_numpy(ak.num(bb)))
                flav = np.zeros(len(fb), int) if isData else fb.hadronFlavour
                for v in ["btagUParTAK4B", "btagUParTAK4QvG"]:
                    out["jet_tag"].fill(
                        step=s,
                        flav=flav,
                        pt=fb.pt,
                        abseta=abs(fb.eta),
                        var=v,
                        value=fb[v],
                        nehef=fb.neHEF,
                        weight=wb,
                    )
                if s in ("raw", "sf"):
                    out["jet_sfvar"].fill(
                        step=s,
                        pt=fb.pt,
                        abseta=abs(fb.eta),
                        qvg=fb.btagUParTAK4QvG,
                        nch=fb.chMultiplicity,
                        nneu=fb.nConstituents - fb.chMultiplicity,
                        weight=wb,
                    )
                for v in ["chHEF", "neHEF", "chEmEF", "neEmEF", "muEF"]:
                    out["jet_frac"].fill(
                        step=s,
                        var=v,
                        pt=fb.pt,
                        abseta=abs(fb.eta),
                        value=fb[v],
                        weight=wb,
                    )
                for v in ["nConstituents", "chMultiplicity", "nSVs"]:
                    out["jet_count"].fill(
                        step=s,
                        var=v,
                        pt=fb.pt,
                        abseta=abs(fb.eta),
                        value=fb[v],
                        weight=wb,
                    )
        return {dataset: out}

    def lep_sfs(self, lep):
        """Per-object SF products: one array per probe level plus 'all' (incl. trigger)."""
        n = len(lep)
        c = self.c
        syst = self.isSyst is not False
        keys = c["levels"] + ["all"]
        res = {k + v: np.ones(n) for k in keys for v in ["", "_up", "_down"]}
        if n == 0:
            return res
        if self.ch == "E":
            return self.ele_level_sfs(lep, res, syst)
        w = Weights(n, storeIndividual=True)
        muSFs(lep, self.SF_map, w, syst, True)
        have = set(w.weightStatistics)
        part = lambda names: (
            np.asarray(w.partial_weight(include=[x for x in names if x in have]))
            if have.intersection(names)
            else np.ones(n)
        )
        for L, names in c["sfs"].items():
            res[L] = part(names)
        res["trig"] = part(c["sfs"]["promptMVA"] + [c["trig_sf"]])
        res["all"] = res["trig"]
        if syst:
            nom = np.asarray(w.weight())
            for v, mod in [("_up", "Up"), ("_down", "Down")]:
                ratio = lambda names: np.prod(
                    [
                        np.asarray(w.weight(x + mod)) / nom
                        for x in names
                        if x + mod in w.variations
                    ]
                    or [np.ones(n)],
                    axis=0,
                )
                for L, names in c["sfs"].items():
                    res[L + v] = res[L] * ratio(names)
                res["trig" + v] = res["trig"] * ratio(
                    c["sfs"]["promptMVA"] + [c["trig_sf"]]
                )
                res["all" + v] = res["trig" + v]
        # tightIso = tightID x PF-iso SF; 2017 has no promptMVA SF (mvaonly = tightID SF only)
        iso = MU_ISO[self._year]
        eta = np.clip(np.asarray(lep.eta), -2.4, 2.399)
        pt = np.clip(np.asarray(lep.pt), 15.0 if self._year == "2017" else 10.0, None)
        for v, vt in [("", "nominal"), ("_up", "systup"), ("_down", "systdown")]:
            if v and not syst:
                continue
            res["tightIso" + v] = res["tightID" + v] * self.SF_map["MUO"][iso].evaluate(eta, pt, vt)
            if self._year == "2017":
                res["mvaonly" + v] = res["tightID" + v]
        return res

    def ele_level_sfs(self, lep, res, syst):
        """Electron SF products per level from ELE_SF (reco, ID, HLT; all up or all down)."""
        eta = np.clip(np.asarray(lep.eta + lep.deltaEtaSC), -2.499, 2.499)
        pt = np.asarray(lep.pt)
        sf = {}
        for name, specs in ELE_SF[self._year].items():
            sf[name] = {}
            for v, vt in [("", "sf"), ("_up", "sfup"), ("_down", "sfdown")]:
                val = np.ones(len(pt))
                for f, corr, yr, wp, lo, hi in specs:
                    m = (pt >= lo) & (pt < hi)
                    if m.any():
                        plo, phi = self.egm_pt[(f, corr, wp)]
                        p = np.clip(pt[m], plo + 1e-3, phi - 1e-3)
                        val[m] = self.egm[f][corr].evaluate(yr, vt, wp, eta[m], p)
                sf[name][v] = val
        lev = ELE_LEV[self._year]
        for L in self.c["levels"] + ["all"]:
            names = lev.get(L, lev["trig"] if L == "all" else ["reco"])
            for v in ["", "_up", "_down"] if syst else [""]:
                res[L + v] = np.prod([sf[x][v] for x in names], axis=0)
        return res

    def postprocess(self, accumulator):
        return accumulator
