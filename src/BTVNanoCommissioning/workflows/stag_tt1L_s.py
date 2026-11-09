"""
s-jet calibration: semileptonic ttbar region (tt1L-s), one lepton flavour per workflow.

Workflows (``workflows/__init__.py``):
  stag_mu_tt1L_s, stag_e_tt1L_s                  histograms with the SPANet assignment
  stag_mu_tt1L_s_spanet_data, stag_e_tt1L_s_spanet_data
                                                 SPANet training inputs (trees, nominal
                                                 only, no SPANet evaluation)

Selection (per JES/JER/lepton-scale shift), with the object definitions of the 2D SF
workflows:
  - golden json (data), MET filters, jet veto map (framework default)
  - muon channel: IsoMu24, exactly one muon with pT > 26 GeV and the prompt-MVA ID
    (NanoAODv15 campaigns; tight ID + PFIso04 otherwise), no electron of the electron
    channel definition
  - electron channel: Ele30_WPTight_Gsf, exactly one electron with pT > 32 GeV and the
    prompt-MVA ID (NanoAODv15 campaigns; mvaIso WP80 otherwise), no muon of the muon
    channel definition
  - >= 4 jets: jet ID, pT > 25 GeV, |eta| < 2.5, dR > 0.4 to the selected lepton and to
    the loose veto leptons (jet_veto_lepton_clean)
  - PuppiMET pT > met_pt_min and mT(lepton, PuppiMET) > mtw_min (STAG_CONFIG, 0 = off)

Production (histogram workflows):
  - s-tagging bins (tagID 0-7 = BT, BL, CT, CL, L, SL, SM, ST) from the "tagging_bin"
    correction of the WP json
  - SPANet jet-parton assignment (lt.b, htb.b, W.q1/q2) on the 6 leading jets, greedy
    best-overall assignment; sJet / cJet are the two W daughters (cJet is the c-tagged one)
  - categories: stag_helpers.TTSL_CATEGORIES
  - event weights from weight_manager (PU, lepton SFs, theory weights, ttbar reweights)

Output histograms (syst axis first):
  - ``sJet_pt_tagID_flav`` / ``cJet_pt_tagID_flav``: the fit inputs, all systematics
  - control histograms with an additional sJet flavour-class axis (nominal only unless
    STAG_CONTROL_SYST=1)
  - ``cutflow``, ``sumw`` and ``sumw_<variation>``

Trees (``--isArray``, always on for the spanet_data workflows): one row per event with the
SPANet inputs (``SpanetJet_*`` as fixed-size arrays of MAX_JETS, ``spanet_lepton_*``,
``spanet_met_*``), the truth targets (``spanet_target_*``, ``gen_is_ttsl``; MC), event
variables and, for the histogram workflows, the SPANet outputs. Written by array_writer
(arrays_<...>/nominal/<dataset>/), so the weight branches, xSecWeight and the
TotalEventCount/TotalEventWeight trees are those of the other workflows.

Environment variables (read when the processor is constructed):
  STAG_SPANET_MODEL        path to the SPANet ONNX model
  STAG_TAGGING_WPS         path to the "tagging_bin" correctionlib json
  STAG_SPANET_W_SCORE_CUT  pass_spanet_w threshold
  STAG_BTAG_BINS           tagID bins counting as b tagged, e.g. "0" or "BT,BL"
  STAG_CTAG_BINS           tagID bins counting as c tagged, e.g. "2" or "CT,CL"
  STAG_MET_PT_MIN          minimum PuppiMET pT [GeV]
  STAG_MTW_MIN             minimum mT(lepton, PuppiMET) [GeV]
  STAG_CONTROL_SYST        "1" to fill control histograms for all systematics
"""

import os

import numpy as np
import awkward as ak
import hist as Hist
from coffea import processor

from BTVNanoCommissioning.utils.correction import (
    load_lumi,
    load_SF,
    common_shifts,
    reweighting,
    weight_manager,
)
from BTVNanoCommissioning.helpers.func import accumulate_shifts, update, dump_lumi
from BTVNanoCommissioning.helpers.update_branch import missing_branch
from BTVNanoCommissioning.utils.array_writer import array_writer
from BTVNanoCommissioning.utils.selection import (
    HLT_helper,
    jet_id,
    MET_filters,
    mu_idiso,
    mu_promptmvaid,
    ele_mvatightid,
    ele_promptmvaid,
    jet_veto_lepton_clean,
)
from BTVNanoCommissioning.helpers.stag_helpers import (
    get_stag_config,
    TaggingBin,
    jet_tagging_bin,
    pad_np,
    custom_flav,
    flav_class,
    delta_phi,
    p4,
    p4_mass,
    spanet_inputs,
    onnx_feed,
    extract_best_assignment,
    SpanetEvaluator,
    gen_ttsl_partons,
    match_targets,
    reconstruct_ttsl,
    MAX_JETS,
    TTSL_CATEGORIES,
)

# NanoAODv15 campaigns: prompt-MVA lepton IDs (as in the 2D SF workflows)
NANOV15_CAMPAIGNS = [
    "Summer24",
    "CustomSummer24",
    "CustomSummer24UParTPaper",
    "CustomSummer25",
    "Winter25",
    "Prompt25",
]

CHANNELS = {"semittM_s": "mu", "semittE_s": "e"}

# jet pT binning of the calibration
PT_BINS = [
    25,
    30,
    35,
    40,
    45,
    50,
    60,
    70,
    80,
    90,
    100,
    120,
    140,
    160,
    180,
    200,
    300,
    400,
    500,
    1000,
]


def _np(x, dtype=None):
    out = ak.to_numpy(x)
    return out if dtype is None else np.asarray(out, dtype=dtype)


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
        selectionModifier="semittM_s",
        spanet_data=False,
    ):
        self._year = year
        self._campaign = campaign
        self.name = name
        self.isSyst = isSyst
        self.chunksize = chunksize
        self.ttbar_reweights = ttbar_reweights
        if selectionModifier not in CHANNELS:
            raise ValueError(
                f"stag_tt1L_s: unknown selectionModifier '{selectionModifier}', "
                f"expected one of {list(CHANNELS)}"
            )
        self.selMod = selectionModifier
        self.channel = CHANNELS[selectionModifier]

        # SPANet training-input production: no model needed, nominal only, trees on
        self.spanet_data = spanet_data
        self.evaluate_spanet = not spanet_data
        self.isArray = isArray or spanet_data
        self.noHist = noHist

        self.cfg = get_stag_config(campaign)
        self.w_score_cut = self.cfg["spanet_w_score_cut"]
        self.btag_bins = self.cfg["btag_bins"]
        self.ctag_bins = self.cfg["ctag_bins"]
        self.control_syst = os.environ.get("STAG_CONTROL_SYST", "0") == "1"

        self.lumiMask = load_lumi(self._campaign)
        self.SF_map = load_SF(
            self._year, self._campaign, self.cfg["lepton_sf_selmod"][self.channel]
        )

        # external inputs: fail early on the submission side if they are missing
        self.tagging_wps_file = self.cfg["tagging_wps"]
        self.tagging_scores_file = self.cfg["tagging_scores"]
        self._tag_corrector = None
        if self.evaluate_spanet:
            TaggingBin(self.tagging_wps_file, self.tagging_scores_file)
            self.spanet = SpanetEvaluator(self.cfg["spanet_model"])
            self.spanet.check()  # sessions are recreated lazily on the workers
        else:
            self.spanet = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_tag_corrector"] = None
        return state

    @property
    def tag_corrector(self):
        if self._tag_corrector is None:
            self._tag_corrector = TaggingBin(
                self.tagging_wps_file, self.tagging_scores_file
            )
        return self._tag_corrector

    @property
    def accumulator(self):
        return self._accumulator

    # ----------------------------------------------------------------------------------
    # histograms
    # ----------------------------------------------------------------------------------
    def define_histograms(self):
        h = {}
        syst = Hist.axis.StrCategory([], name="syst", growth=True)
        cat = Hist.axis.StrCategory(list(TTSL_CATEGORIES), name="category")
        jflav = Hist.axis.IntCategory(
            [0, 1, 3, 4, 5], name="jflav", label="sJet flavour class"
        )

        h["cutflow"] = Hist.Hist(
            Hist.axis.StrCategory([], name="step", growth=True),
            Hist.storage.Weight(),
        )
        if not self.evaluate_spanet:
            return h

        # fit inputs
        for jet in ("sJet", "cJet"):
            h[f"{jet}_pt_tagID_flav"] = Hist.Hist(
                syst,
                cat,
                Hist.axis.Variable(
                    PT_BINS, name=f"{jet}_pt", label=f"$p_T$({jet}) [GeV]"
                ),
                Hist.axis.Regular(8, 0, 8, name=f"{jet}_tagID", label=f"tagID({jet})"),
                Hist.axis.Regular(
                    11, -5, 6, name=f"{jet}_flav", label=f"flavour({jet})"
                ),
                Hist.storage.Weight(),
            )

        for hname, (_, axis) in self.control_variables().items():
            h[hname] = Hist.Hist(syst, cat, jflav, axis, Hist.storage.Weight())
        return h

    @staticmethod
    def control_variables():
        """hist name -> (variable key, axis)"""
        R, V = Hist.axis.Regular, Hist.axis.Variable
        out = {
            "n_jet": ("n_jet", R(4, 4, 8, name="n_jet", label="Number of jets")),
            "ht": ("ht", R(30, 100, 550, name="ht", label="$H_T$ [GeV]")),
            "met_pt": (
                "met_pt",
                R(30, 0, 150, name="met_pt", label="$p_T^{miss}$ [GeV]"),
            ),
            "mtw": ("mtw", R(30, 0, 150, name="mtw", label="$m_T(W)$ [GeV]")),
            "lep_pt": (
                "lep_pt",
                R(30, 0, 300, name="lep_pt", label="$p_T(\\ell)$ [GeV]"),
            ),
            "lep_eta": (
                "lep_eta",
                R(25, -2.5, 2.5, name="lep_eta", label="$\\eta(\\ell)$"),
            ),
            "sJet_pt": (
                "sJet_pt",
                V(PT_BINS, name="sJet_pt", label="$p_T(j_s)$ [GeV]"),
            ),
            "sJet_tagID": (
                "sJet_tagID",
                R(8, 0, 8, name="sJet_tagID", label="tagID($j_s$)"),
            ),
            "sJet_flav": (
                "sJet_flav",
                R(11, -5, 6, name="sJet_flav", label="flavour($j_s$)"),
            ),
            "cJet_pt": (
                "cJet_pt",
                V(PT_BINS, name="cJet_pt", label="$p_T(j_c)$ [GeV]"),
            ),
            "cJet_tagID": (
                "cJet_tagID",
                R(8, 0, 8, name="cJet_tagID", label="tagID($j_c$)"),
            ),
            "cJet_flav": (
                "cJet_flav",
                R(11, -5, 6, name="cJet_flav", label="flavour($j_c$)"),
            ),
            "whad_m": (
                "whad_m",
                R(30, 20, 150, name="whad_m", label="$m(W_{had})$ [GeV]"),
            ),
            "whad_dphi": (
                "dphi_whad",
                R(32, 0, 3.2, name="whad_dphi", label="$\\Delta\\phi(q_1,q_2)$"),
            ),
            "whad_ptRel": (
                "whad_ptRel",
                R(30, 0, 1.5, name="whad_ptRel", label="$p_T(q_1)/p_T(q_2)$"),
            ),
            "whadq1_pt": (
                "whadq1_pt",
                R(50, 20, 120, name="whadq1_pt", label="$p_T(q_1)$ [GeV]"),
            ),
            "whadq2_pt": (
                "whadq2_pt",
                R(60, 20, 80, name="whadq2_pt", label="$p_T(q_2)$ [GeV]"),
            ),
            "whadq1_tagID": (
                "whadq1_tagID",
                R(8, 0, 8, name="whadq1_tagID", label="tagID($q_1$)"),
            ),
            "whadq2_tagID": (
                "whadq2_tagID",
                R(8, 0, 8, name="whadq2_tagID", label="tagID($q_2$)"),
            ),
            "bhad_tagID": (
                "bhad_tagID",
                R(8, 0, 8, name="bhad_tagID", label="tagID($b_{had}$)"),
            ),
            "blep_tagID": (
                "blep_tagID",
                R(8, 0, 8, name="blep_tagID", label="tagID($b_{lep}$)"),
            ),
            "bl_m": (
                "bl_m",
                R(50, 0, 250, name="bl_m", label="$m(\\ell, b_{lep})$ [GeV]"),
            ),
            "bl_dR": (
                "bl_dR",
                R(40, 0, 4, name="bl_dR", label="$\\Delta R(b_{lep}, \\ell)$"),
            ),
            "bb_m": ("bb_m", R(50, 0, 250, name="bb_m", label="$m(bb)$ [GeV]")),
            "bb_dphi": (
                "dphi_bb",
                R(32, 0, 3.2, name="bb_dphi", label="$\\Delta\\phi(b,b)$"),
            ),
            "bb_ptRel": (
                "bb_ptRel",
                R(30, 0, 1.5, name="bb_ptRel", label="$p_T(b_{had})/p_T(b_{lep})$"),
            ),
        }
        for obj, label in (("w", "W"), ("lt", "b lep"), ("htb", "b had")):
            for kind in ("assignment", "detection"):
                key = f"spanet_{obj}_{kind}_prob"
                out[key] = (
                    key,
                    R(50, 0, 1, name=key, label=f"SPANet {label} {kind} score"),
                )
        return out

    # ----------------------------------------------------------------------------------
    # driver
    # ----------------------------------------------------------------------------------
    def process(self, events):
        events = missing_branch(events, f"{self._year}_{self._campaign}")
        sumws = reweighting(
            events,
            self.isSyst,
            ttbar_reweights=self.ttbar_reweights,
            campaign=self._campaign,
        )
        vetoed_events, shifts = common_shifts(self, events)
        if self.spanet_data:
            shifts = [s for s in shifts if s[1] is None]

        return accumulate_shifts(
            self.process_shift(update(vetoed_events, collections), sumws, name)
            for collections, name in shifts
        )

    def select_leptons(self, events):
        """Tight muons and electrons of the channel definitions."""
        if self._campaign in NANOV15_CAMPAIGNS:
            mu_id = mu_promptmvaid(events)
            el_id = ele_promptmvaid(events, self._campaign)
        else:
            mu_id = mu_idiso(events)
            el_id = ele_mvatightid(events, self._campaign)
        muons = events.Muon[(events.Muon.pt > 26) & mu_id]
        electrons = events.Electron[(events.Electron.pt > 32) & el_id]
        return muons, electrons

    def process_shift(self, events, sumws, shift_name):
        dataset = events.metadata["dataset"]
        isRealData = not hasattr(events, "genWeight")
        output = {} if self.noHist else self.define_histograms()

        if shift_name is None:
            output["sumw"] = sumws["sumw"]
            # one sum per generator-level variation, see correction.reweighting
            output.update({k: v for k, v in sumws.items() if k.startswith("sumw_")})

        # ------------------------------------------------------------------ selection
        req_lumi = np.ones(len(events), dtype=bool)
        if isRealData:
            req_lumi = self.lumiMask(events.run, events.luminosityBlock)
        if shift_name is None:
            output = dump_lumi(events[req_lumi], output)

        req_trig = _np(HLT_helper(events, self.cfg["triggers"][self.channel]), bool)
        req_metf = _np(MET_filters(events, self._campaign), bool)

        muons, electrons = self.select_leptons(events)
        n_mu = _np(ak.num(muons, axis=1))
        n_el = _np(ak.num(electrons, axis=1))
        if self.channel == "mu":
            req_lep = (n_mu == 1) & (n_el == 0)
            lepton = ak.firsts(muons)
        else:
            req_lep = (n_el == 1) & (n_mu == 0)
            lepton = ak.firsts(electrons)

        # jets: ID, pT, |eta| < 2.5, cleaned against the selected lepton and the loose
        # veto leptons
        jets = events.Jet
        req_clean_veto = jet_veto_lepton_clean(events, self._campaign)
        lep_list = muons if self.channel == "mu" else electrons
        jet_mask = ak.fill_none(
            jet_id(events, self._campaign, max_eta=2.5, min_pt=25)
            & ak.all(jets.metric_table(lep_list) > 0.4, axis=2, mask_identity=True)
            & req_clean_veto,
            False,
            axis=-1,
        )
        sel_jets = jets[jet_mask]
        sel_jets = sel_jets[ak.argsort(sel_jets.pt, axis=1, ascending=False)]
        n_jet = _np(ak.num(sel_jets, axis=1))

        # PuppiMET and W transverse mass
        met_pt = _np(events.PuppiMET.pt, np.float64)
        met_phi = _np(events.PuppiMET.phi, np.float64)
        lep_pt_all = _np(ak.fill_none(lepton.pt, 0.0), np.float64)
        lep_phi_all = _np(ak.fill_none(lepton.phi, 0.0), np.float64)
        mtw = np.sqrt(
            2.0 * lep_pt_all * met_pt * (1.0 - np.cos(delta_phi(lep_phi_all, met_phi)))
        )
        req_met = met_pt > self.cfg["met_pt_min"]
        req_mtw = mtw > self.cfg["mtw_min"]

        steps = [
            ("all", np.ones(len(events), dtype=bool)),
            ("lumi", req_lumi),
            ("metfilter", req_metf),
            ("trigger", req_trig),
            ("1lep", req_lep),
            (">=4jets", n_jet >= 4),
            ("met", req_met),
            ("mtw", req_mtw),
        ]
        event_level = np.ones(len(events), dtype=bool)
        for step, m in steps:
            event_level = event_level & np.asarray(m, dtype=bool)
            if shift_name is None and "cutflow" in output:
                output["cutflow"].fill(step=step, weight=float(event_level.sum()))

        if not event_level.any():
            if self.isArray and shift_name is None:
                array_writer(
                    self,
                    events[event_level],
                    events,
                    None,
                    ["nominal"],
                    dataset,
                    isRealData,
                    empty=True,
                )
            return {dataset: output}

        # ------------------------------------------------------------------ production
        ev = events[event_level]
        jets_ev = sel_jets[event_level]
        lep_ev = lep_list[event_level][:, 0]
        n = len(ev)
        njet = n_jet[event_level]

        v = {"n_jet": njet}
        v["ht"] = _np(ak.sum(jets_ev.pt, axis=1), np.float64)
        v["met_pt"] = met_pt[event_level]
        v["mtw"] = mtw[event_level]
        lep = {f: _np(lep_ev[f], np.float64) for f in ("pt", "eta", "phi", "mass")}
        v["lep_pt"], v["lep_eta"] = lep["pt"], lep["eta"]
        v["lep_charge"] = _np(lep_ev.charge, np.int32)

        # padded jets (pt sorted) -> SPANet inputs
        jet = {
            f: pad_np(jets_ev[f], MAX_JETS, 0.0, np.float64)
            for f in ("pt", "eta", "phi", "mass")
        }
        jet["mask"] = np.arange(MAX_JETS)[None, :] < np.minimum(njet, MAX_JETS)[:, None]
        met = {"pt": v["met_pt"], "phi": met_phi[event_level]}
        inputs = spanet_inputs(jet, lep, met)

        # truth targets (MC only, -1 for non-semileptonic ttbar)
        targets = None
        if not isRealData and "GenPart" in ev.fields:
            partons = gen_ttsl_partons(ev.GenPart)
            nmax = max(int(njet.max()), 1)
            targets = match_targets(
                pad_np(jets_ev.eta, nmax, 0.0, np.float64),
                pad_np(jets_ev.phi, nmax, 0.0, np.float64),
                np.arange(nmax)[None, :] < njet[:, None],
                partons,
            )
            targets["gen_is_ttsl"] = partons["is_sl"]

        recon = None
        if self.evaluate_spanet:
            jet["tagID"] = pad_np(
                jet_tagging_bin(jets_ev, self.tag_corrector), MAX_JETS, -1, np.int32
            )
            if not isRealData:
                jet["flav"] = custom_flav(
                    pad_np(jets_ev.partonFlavour, MAX_JETS, 0, np.int32)
                )

            outputs = self.spanet(onnx_feed(inputs), event=ak.to_numpy(ev.event))
            assignment = extract_best_assignment(
                outputs["lt_assignment_probability"],
                outputs["htb_assignment_probability"],
                outputs["W_assignment_probability"],
            )
            recon = reconstruct_ttsl(
                jet,
                assignment,
                outputs,
                self.w_score_cut,
                not isRealData,
                btag_bins=self.btag_bins,
                ctag_bins=self.ctag_bins,
            )
            v.update({k: val for k, val in recon.items() if not k.startswith("_")})

            # lepton - leptonic b variables
            lp = p4(lep["pt"], lep["eta"], lep["phi"], lep["mass"])
            lb = tuple(a + b for a, b in zip(lp, recon["_blep_p4"]))
            v["bl_m"] = p4_mass(*lb)
            bl_idx = np.maximum(recon["spanet_lt_jet_idx"], 0)
            bl_eta = jet["eta"][np.arange(n), bl_idx]
            bl_phi = jet["phi"][np.arange(n), bl_idx]
            v["bl_dR"] = np.sqrt(
                (bl_eta - lep["eta"]) ** 2 + delta_phi(bl_phi, lep["phi"]) ** 2
            )
            v["jflav"] = flav_class(v["sJet_flav"]).astype(np.int32)

        # ------------------------------------------------------------------ weights
        pruned_ev = ev
        if self.channel == "mu":
            pruned_ev["SelMuon"] = lep_ev
        else:
            pruned_ev["SelElectron"] = lep_ev
        weights = weight_manager(
            pruned_ev,
            self.SF_map,
            self.isSyst,
            ttbar_reweights=self.ttbar_reweights,
            campaign=self._campaign,
        )
        if shift_name is None:
            systematics = ["nominal"] + list(weights.variations)
        else:
            systematics = [shift_name]
        if self.isSyst is False:
            systematics = systematics[:1]

        # ------------------------------------------------------------------ output
        if not self.noHist and self.evaluate_spanet:
            self.fill_histograms(output, v, weights, systematics, shift_name)

        if self.isArray and shift_name is None:
            branches = self.add_tree_branches(pruned_ev, v, inputs, targets, recon)
            array_writer(
                self,
                pruned_ev,
                events,
                weights,
                ["nominal"],
                dataset,
                isRealData,
                doOnly=branches,
            )

        return {dataset: output}

    # ----------------------------------------------------------------------------------
    def fill_histograms(self, output, v, weights, systematics, shift_name):
        cat_masks = {
            c: np.asarray(f(v), dtype=bool) for c, f in TTSL_CATEGORIES.items()
        }
        controls = self.control_variables()

        # flatten (event, category) pairs so each histogram is filled once per syst
        pos, labels = [], []
        for cat, m in cat_masks.items():
            sel = np.nonzero(m)[0]
            pos.append(sel)
            labels.append(np.full(len(sel), cat))
        pos = np.concatenate(pos)
        labels = np.concatenate(labels)
        if len(pos) == 0:
            return
        vals = {k: np.asarray(val)[pos] for k, val in v.items() if np.ndim(val) == 1}

        for syst in systematics:
            if shift_name is None and syst != "nominal" and syst in weights.variations:
                w = weights.weight(modifier=syst)[pos]
            else:
                w = weights.weight()[pos]

            for jet in ("sJet", "cJet"):
                output[f"{jet}_pt_tagID_flav"].fill(
                    **{
                        "syst": syst,
                        "category": labels,
                        f"{jet}_pt": vals[f"{jet}_pt"],
                        f"{jet}_tagID": vals[f"{jet}_tagID"],
                        f"{jet}_flav": vals[f"{jet}_flav"],
                    },
                    weight=w,
                )

            # control histograms: nominal only unless STAG_CONTROL_SYST=1
            if syst != "nominal" and not self.control_syst:
                continue
            for hname, (key, axis) in controls.items():
                output[hname].fill(
                    **{
                        "syst": syst,
                        "category": labels,
                        "jflav": vals["jflav"],
                        axis.name: vals[key],
                    },
                    weight=w,
                )

    @staticmethod
    def add_tree_branches(pruned_ev, v, inputs, targets, recon):
        """Store the tree content on the pruned events; returns the branch list
        for array_writer (weight branches are added there)."""
        branches = ["run", "luminosityBlock", "event"]
        # SPANet inputs: SpanetJet_* as fixed-size (n_events, MAX_JETS) arrays
        for k, val in inputs.items():
            pruned_ev[k] = val
            branches.append(k)
        for k in ("n_jet", "lep_charge", "ht", "met_pt", "mtw"):
            pruned_ev[k] = v[k]
            branches.append(k)
        for k, val in (targets or {}).items():
            pruned_ev[k] = val
            branches.append(k)
        for k, val in (recon or {}).items():
            if not k.startswith("_") and np.ndim(val) == 1:
                pruned_ev[k] = np.asarray(val)
                branches.append(k)
        return branches

    def postprocess(self, accumulator):
        return accumulator
