import awkward as ak, numpy as np
from coffea import processor

from BTVNanoCommissioning.utils.correction import (
    load_lumi,
    load_SF,
    weight_manager,
    common_shifts,
    reweighting,
)
from BTVNanoCommissioning.helpers.func import (
    accumulate_shifts,
    update,
    dump_lumi,
    PFCand_link,
    SV_link_ragged,
)
from BTVNanoCommissioning.helpers.update_branch import missing_branch
from BTVNanoCommissioning.utils.histogramming.histogrammer import (
    histogrammer,
    histo_writer,
)
from BTVNanoCommissioning.utils.histogramming.histograms.sf2d import (
    SCHEMES,
    TAG_SEL_CODES,
)
from BTVNanoCommissioning.utils.array_writer import array_writer
from BTVNanoCommissioning.utils.selection import (
    HLT_helper,
    jet_id,
    mu_idiso,
    mu_promptmvaid,
    ele_mvatightid,
    ele_promptmvaid,
    MET_filters,
    btag_wp,
    fineflav,
    jet_veto_lepton_clean,
)


class NanoProcessor(processor.ProcessorABC):
    ## Define histograms
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
        selectionModifier="semittE_2D",
    ):
        self._year = year
        self._campaign = campaign
        self.name = name
        self.isSyst = isSyst
        self.isArray = isArray
        self.noHist = noHist
        self.lumiMask = load_lumi(self._campaign)
        self.chunksize = chunksize
        self.ttbar_reweights = ttbar_reweights
        self.selMod = selectionModifier
        ## Load corrections
        self.SF_map = load_SF(self._year, self._campaign, self.selMod)

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        events = missing_branch(
            events,
            f"{self._year}_{self._campaign}",
            extra_taggers=(
                ("PNet", "DeepFlav") if getattr(self, "taggerCompare", False) else ()
            ),
        )
        sumws = reweighting(
            events,
            self.isSyst,
            ttbar_reweights=self.ttbar_reweights,
            campaign=self._campaign,
        )
        vetoed_events, shifts = common_shifts(self, events)

        return accumulate_shifts(
            self.process_shift(update(vetoed_events, collections), sumws, name)
            for collections, name in shifts
        )

    def process_shift(self, events, sumws, shift_name):
        dataset = events.metadata["dataset"]
        isRealData = not hasattr(events, "genWeight")

        objs = ["MET", "jet0", "jet1", "jet2", "jet3"]
        if self.selMod == "semittE_2D":
            objs.append("ele")
            chn = "ele"
            triggers = ["Ele30_WPTight_Gsf"]
        elif self.selMod == "semittM_2D":
            objs.append("mu")
            chn = "mu"
            triggers = ["IsoMu24"]
        else:
            raise ValueError(self.selMod, "is not a valid selection modifier.")

        ## Create histograms
        output = {}
        if not self.noHist:
            output = histogrammer(
                events.Jet.fields,
                obj_list=objs,
                hist_collections=["common", "fourvec", "ttsemilep_2D"],
                channel=chn,
                njet=4,
                tagger_compare=getattr(self, "taggerCompare", False),
            )

        if shift_name is None:
            output["sumw"] = sumws["sumw"]
            # one sum per generator-level variation, see correction.reweighting
            output.update({k: v for k, v in sumws.items() if k.startswith("sumw_")})

        ####################
        #    Selections    #
        ####################

        ## Lumimask
        req_lumi = np.ones(len(events), dtype="bool")
        if isRealData:
            req_lumi = self.lumiMask(events.run, events.luminosityBlock)
        # only dump for nominal case
        if shift_name is None:
            output = dump_lumi(events[req_lumi], output)

        ## HLT
        req_trig = HLT_helper(events, triggers)

        ## Lepton cuts
        if self.selMod == "semittE_2D":
            if self._campaign in [
                "Summer24",
                "CustomSummer24",
                "CustomSummer24UParTPaper",
                "CustomSummer25",
                "Winter25",
                "Prompt25",
            ]:  # NanoAODv15
                event_iso_lep = events.Electron[
                    (events.Electron.pt > 32) & ele_promptmvaid(events, self._campaign)
                ]
            else:
                event_iso_lep = events.Electron[
                    (events.Electron.pt > 32) & ele_mvatightid(events, self._campaign)
                ]
        elif self.selMod == "semittM_2D":
            if self._campaign in [
                "Summer24",
                "CustomSummer24",
                "CustomSummer24UParTPaper",
                "CustomSummer25",
                "Winter25",
                "Prompt25",
            ]:  # NanoAODv15
                event_iso_lep = events.Muon[
                    (events.Muon.pt > 26) & mu_promptmvaid(events)
                ]
            else:
                event_iso_lep = events.Muon[(events.Muon.pt > 26) & mu_idiso(events)]
            event_soft_mu = events.Muon[
                (events.Muon.pt > 5)
                & (abs(events.Muon.eta) < 2.5)
                & (events.Muon.tightId > 0.5)
                & (events.Muon.pfRelIso04_all > 0.25)
            ]
        req_lep = ak.count(event_iso_lep.pt, axis=1) == 1

        # DY -> mumu veto
        event_iso_lep = ak.pad_none(event_iso_lep, 1, axis=1)
        if self.selMod == "semittE_2D":
            req_DYveto = np.ones(len(events), dtype="bool")
        elif self.selMod == "semittM_2D":
            event_di_mu = event_soft_mu + event_iso_lep[:, 0]
            req_DYveto = (event_di_mu.mass > 12) & (
                (event_di_mu.mass < 81) | (event_di_mu.mass > 101)
            )
            req_DYveto = ak.all(req_DYveto, axis=1)

        ## Jet cuts
        # additional cleaning against the loose veto-lepton collections
        # (Run 2 parity); None-handling documented in jet_veto_lepton_clean.
        req_clean_veto = jet_veto_lepton_clean(events, self._campaign)
        jet_sel = ak.fill_none(
            jet_id(events, self._campaign, min_pt=25)
            & (
                ak.all(
                    events.Jet.metric_table(event_iso_lep) > 0.4,
                    axis=2,
                    mask_identity=True,
                )
            )
            & req_clean_veto,
            False,
            axis=-1,
        )
        event_jet = events.Jet[jet_sel]
        # raw jet index for the up-to-4 selected jets stored in SelJet; kept
        # ragged (not padded) to match SelJet's own shape, as in
        # ctag_DY_valid_sf.py
        jetindx4 = ak.local_index(events.Jet.pt)[jet_sel]
        n_jet = ak.count(event_jet.pt, axis=1)
        req_jets = (n_jet >= 3) & (n_jet <= 4)
        # req_jets = n_jet >= 4

        # 2D tagger name (used below for the 2Db/2Dc category masks)
        tagger = (
            "UParTAK4"
            if self._campaign
            in [
                "Summer24",
                "CustomSummer24",
                "CustomSummer24UParTPaper",
                "CustomSummer25",
                "Winter25",
                "Prompt25",
            ]
            else "PNet"
        )

        ## Other cuts

        ## MET cuts
        event_MET = ak.zip(
            {
                "pt": events.PuppiMET.pt,
                "eta": ak.zeros_like(events.PuppiMET.pt),
                "phi": events.PuppiMET.phi,
                "mass": ak.zeros_like(events.PuppiMET.pt),
            },
            with_name="PtEtaPhiMLorentzVector",
        )
        req_MET = event_MET.pt > 50.0
        req_metfilter = MET_filters(events, self._campaign)

        # Cut on tranverse W mass
        # Calculation taken from https://github.com/cms-btv-pog/BTVNanoCommissioning/blob/9f56298727c9c05df49b5701a0e92f8bf43d610e/src/BTVNanoCommissioning/workflows/ctag_Wctt_valid_sf.py#L299-L305
        event_dphi = event_iso_lep[:, 0].phi - events.PuppiMET.phi
        event_dphi = np.where(event_dphi < np.pi, event_dphi + 2 * np.pi, event_dphi)
        event_dphi = np.where(event_dphi > np.pi, event_dphi - 2 * np.pi, event_dphi)
        event_Wmt = np.sqrt(
            2 * event_iso_lep[:, 0].pt * events.PuppiMET.pt * (1 - np.cos(event_dphi))
        )
        req_Wmt = event_Wmt > 40

        event_level = ak.fill_none(
            req_lumi & req_trig & req_lep
            # & req_DYveto
            & req_jets & req_MET & req_metfilter & req_Wmt,
            False,
        )
        if len(events[event_level]) == 0:
            if self.isArray:
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

        ####################
        # Selected objects #
        ####################

        pruned_ev = events[event_level]
        pruned_ev["SelJet"] = event_jet[event_level][:, :4]
        # Match secondary vertices to each of the (up to 4) selected jets.
        # SV_link_ragged falls back to a dR match when the Jet->SV index link
        # is unavailable (e.g. CustomSummer24, a private NanoAOD production
        # that keeps the SV collection but drops it).
        if "SV" in events.fields:
            svmass, svd3dsig = SV_link_ragged(
                events, event_level, jetindx4[event_level][:, :4]
            )
            pruned_ev["SelJet", "svmass"] = svmass
            pruned_ev["SelJet", "svd3dsig"] = svd3dsig
        # slot index within SelJet (0 = leading, 1 = subleading, ...), not the
        # original NanoAOD Jet index -- same convention as DY and ttdilep
        pruned_ev["SelJet", "jetIdx"] = ak.local_index(pruned_ev.SelJet, axis=1)
        if self.selMod == "semittE_2D":
            pruned_ev["SelElectron"] = event_iso_lep[event_level][:, 0]
        elif self.selMod == "semittM_2D":
            pruned_ev["SelMuon"] = event_iso_lep[event_level][:, 0]
        pruned_ev["njet"] = ak.count(event_jet[event_level].pt, axis=1)
        b_jet_mask = btag_wp(
            event_jet[event_level], self._year, self._campaign, tagger, "2Db", "M"
        )
        c_jet_mask = btag_wp(
            event_jet[event_level], self._year, self._campaign, tagger, "2Dc", "M"
        )
        pruned_ev["nbjet"] = ak.count(event_jet[event_level].pt[b_jet_mask], axis=1)
        pruned_ev["ncjet"] = ak.count(event_jet[event_level].pt[c_jet_mask], axis=1)
        if not isRealData and "nBHadrons" in pruned_ev.SelJet.fields:
            pruned_ev["SelJet", "fineFlav"] = fineflav(pruned_ev.SelJet)
        # Per-scheme 2D tag categories: event-level passTag (at least one of
        # the two LEADING SelJet jets in a B2+ category) and per-event B2+/C2+
        # jet counts, for every scheme whose category branch exists in this
        # sample. The code sets are explicit per scheme (TAG_SEL_CODES):
        # phase4's dense codes interleave, so positional slices of the
        # SCHEMES code lists would select the wrong region.
        for scheme, (branch, _codes) in SCHEMES.items():
            if branch not in events.Jet.fields:
                continue
            jetcat = pruned_ev.SelJet[branch]
            b2_mask = ak.zeros_like(jetcat, dtype=bool)
            for code in TAG_SEL_CODES[scheme]["b2plus"]:
                b2_mask = b2_mask | (jetcat == code)
            c2_mask = ak.zeros_like(jetcat, dtype=bool)
            for code in TAG_SEL_CODES[scheme]["c2plus"]:
                c2_mask = c2_mask | (jetcat == code)
            pruned_ev[f"passTag_{scheme}"] = ak.fill_none(
                ak.any(b2_mask[:, :2], axis=1), False
            )
            pruned_ev[f"nbjet_{scheme}"] = ak.sum(b2_mask, axis=1)
            pruned_ev[f"ncjet_{scheme}"] = ak.sum(c2_mask, axis=1)
        pruned_ev["MET"] = event_MET[event_level]
        pruned_ev["w_mt"] = event_Wmt[event_level]

        ####################
        #     Output       #
        ####################

        # Configure SFs
        weights = weight_manager(
            pruned_ev,
            self.SF_map,
            self.isSyst,
            ttbar_reweights=self.ttbar_reweights,
            campaign=self._campaign,
        )
        # Configure systematics
        if shift_name is None:
            systematics = ["nominal"] + list(weights.variations)
        else:
            systematics = [shift_name]

        # Configure histograms
        if not self.noHist:
            output = histo_writer(
                pruned_ev, output, weights, systematics, self.isSyst, self.SF_map
            )
        # Output arrays
        if self.isArray:
            array_writer(
                self,
                pruned_ev,
                events,
                weights,
                systematics,
                dataset,
                isRealData,
                schema="SLIM",
                perJet=True,
            )
        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
