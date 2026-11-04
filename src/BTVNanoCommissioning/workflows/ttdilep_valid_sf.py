import awkward as ak, numpy as np
from coffea import processor

# functions to load SFs, corrections
from BTVNanoCommissioning.utils.correction import (
    load_lumi,
    load_SF,
    weight_manager,
    common_shifts,
    reweighting,
)

# user helper function
from BTVNanoCommissioning.helpers.func import (
    update,
    dump_lumi,
    PFCand_link,
    SV_link,
    accumulate_shifts,
)
from BTVNanoCommissioning.helpers.update_branch import missing_branch

## load histograms & selctions for this workflow
from BTVNanoCommissioning.utils.histogramming.histogrammer import (
    histogrammer,
    histo_writer,
)
from BTVNanoCommissioning.utils.array_writer import array_writer
from BTVNanoCommissioning.utils.selection import (
    HLT_helper,
    jet_id,
    ele_promptmvaid,
    ele_mvatightid,
    ele_cuttightid,
    mu_promptmvaid,
    mu_idiso,
    MET_filters,
    fineflav,
    jet_veto_lepton_clean,
)


class NanoProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year="2022",
        campaign="Summer22Run3",
        name="",
        isSyst=False,
        isArray=False,
        noHist=False,
        chunksize=75000,
        ttbar_reweights="none",
        selectionModifier="tt_dilep",
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

    ## Apply corrections on momentum/mass on MET, Jet, Muon
    def process(self, events):
        events = missing_branch(events, f"{self._year}_{self._campaign}")
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

        ## Create histograms
        objs = ["mu", "ele", "jet0", "jet1"]
        hist_collections = ["common", "fourvec", "ttdilep"]
        if self.selMod == "ttdilep_sf_2D":
            objs.append("MET")
            hist_collections.append("SF2D")
        output = {}
        if not self.noHist:
            output = histogrammer(
                events.Jet.fields,
                obj_list=objs,
                hist_collections=hist_collections,
                njet=2,
                # SelJet.svmass is valid since the jetindx fix (111b17f);
                # see histograms/sf2d.py.
                sf2d_svmass=True,
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
        triggers = [
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
        ]
        req_trig = HLT_helper(events, triggers)

        if self.selMod == "ttdilep_sf_2D":
            # Veto-lepton jet cleaning (Run 2 parity), computed BEFORE
            # events.Muon/Electron are overwritten with the selected leptons
            # below; None-handling documented in jet_veto_lepton_clean.
            req_clean_veto = jet_veto_lepton_clean(events, self._campaign)

        ## Muon cuts
        # muon twiki: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2
        if self._campaign in [
            "Summer24",
            "CustomSummer24",
            "CustomSummer24UParTPaper",
            "CustomSummer25",
            "Winter25",
            "Prompt25",
        ]:  # NanoAODv15
            events["Muon"] = events.Muon[(events.Muon.pt > 25) & mu_promptmvaid(events)]
        else:
            events["Muon"] = events.Muon[(events.Muon.pt > 25) & mu_idiso(events)]
        events["Muon"] = ak.pad_none(events.Muon, 1, axis=1)
        req_muon = ak.count(events.Muon.pt, axis=1) == 1

        ## Electron cuts
        # electron twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2
        if self._campaign in [
            "Summer24",
            "CustomSummer24",
            "CustomSummer24UParTPaper",
            "CustomSummer25",
            "Winter25",
            "Prompt25",
        ]:  # NanoAODv15
            events["Electron"] = events.Electron[
                (events.Electron.pt > 25) & ele_promptmvaid(events, self._campaign)
            ]
        else:
            if self.selMod == "ttdilep_sf_2D":
                events["Electron"] = events.Electron[
                    (events.Electron.pt > 25) & ele_mvatightid(events, self._campaign)
                ]
            else:
                events["Electron"] = events.Electron[
                    (events.Electron.pt > 25) & ele_cuttightid(events, self._campaign)
                ]
        events["Electron"] = ak.pad_none(events.Electron, 1, axis=1)
        req_ele = ak.count(events.Electron.pt, axis=1) == 1

        ## Jet cuts
        jetsel = ak.fill_none(
            jet_id(events, self._campaign, min_pt=25)
            & (
                ak.all(
                    events.Jet.metric_table(events.Muon) > 0.4,
                    axis=2,
                    mask_identity=True,
                )
            )
            & (
                ak.all(
                    events.Jet.metric_table(events.Electron) > 0.4,
                    axis=2,
                    mask_identity=True,
                )
            ),
            False,
        )
        if self.selMod == "ttdilep_sf_2D":
            # additional cleaning against the loose veto-lepton collections
            jetsel = jetsel & req_clean_veto
        event_jet = events.Jet[jetsel]
        # req_jets = ak.num(event_jet.pt) >= 2
        req_jets = ak.num(event_jet.pt) == 2

        ## Other cuts
        req_opposite_charge = (
            events.Electron[:, 0:1].charge * events.Muon[:, 0:1].charge
        ) == -1
        req_opposite_charge = ak.fill_none(req_opposite_charge, False)
        req_opposite_charge = ak.flatten(req_opposite_charge)

        ## Dilepton mass cut (reject low-mass DY/quarkonia resonances)
        dilep_mass = (events.Electron[:, 0:1] + events.Muon[:, 0:1]).mass
        req_dilep_mass = ak.fill_none(dilep_mass > 20, False)
        req_dilep_mass = ak.flatten(req_dilep_mass)

        ## store jet index for PFCands, create mask on the jet index
        jetindx = ak.mask(ak.local_index(events.Jet.pt), jetsel)
        jetindx = ak.pad_none(jetindx, 2)
        jetindx = jetindx[:, :2]

        ## raw jet index of the two jets actually stored as SelJet. `jetindx`
        ## above cannot be used for this: ak.mask keeps positions, so it holds
        ## raw Jet slots 0/1 (None when they fail jetsel), not the first two
        ## *selected* jets.
        jetindx2 = ak.local_index(events.Jet.pt)[jetsel][:, :2]

        ## MET
        if self.selMod == "ttdilep_sf_2D":
            event_MET = ak.zip(
                {
                    "pt": events.PuppiMET.pt,
                    "eta": ak.zeros_like(events.PuppiMET.pt),
                    "phi": events.PuppiMET.phi,
                    "mass": ak.zeros_like(events.PuppiMET.pt),
                },
                with_name="PtEtaPhiMLorentzVector",
            )
            req_MET = events.PuppiMET.pt > 20
            # transverse mass of the (leading electron + leading muon) system
            # against PuppiMET; only 1-cos(dphi) enters, so no phi wrapping is
            # needed
            ll = events.Electron[:, 0] + events.Muon[:, 0]
            mtll = np.sqrt(
                2 * ll.pt * event_MET.pt * (1 - np.cos(ll.delta_phi(event_MET)))
            )
            # 50 GeV (was 100, Run 2): 73 % instead of 45 % of the events, purity 0.905 vs 0.908 (scan 2026-10-23)
            req_mtll = ak.fill_none(mtll > 50, False)
        req_metfilter = MET_filters(events, self._campaign)

        event_level = (
            req_trig
            & req_lumi
            & req_muon
            & req_ele
            & req_jets
            & req_opposite_charge
            & req_dilep_mass
            & req_metfilter
        )
        if self.selMod == "ttdilep_sf_2D":
            event_level = event_level & req_MET & req_mtll
        event_level = ak.fill_none(event_level, False)
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

        # Keep the structure of events and pruned the object size
        pruned_ev = events[event_level]
        pruned_ev["SelJet"] = event_jet[event_level][:, :2]
        # position of each entry within the stored SelJet list itself (0 =
        # leading, 1 = subleading), not the original NanoAOD Jet index
        pruned_ev["SelJet", "jetIdx"] = ak.local_index(pruned_ev.SelJet, axis=1)
        pruned_ev["SelMuon"] = events.Muon[event_level][:, 0]
        pruned_ev["SelElectron"] = events.Electron[event_level][:, 0]
        pruned_ev["dr_mujet0"] = pruned_ev.Muon.delta_r(pruned_ev.Jet[:, 0])
        pruned_ev["dr_mujet1"] = pruned_ev.Muon.delta_r(pruned_ev.Jet[:, 1])
        pruned_ev["njet"] = ak.count(event_jet[event_level].pt, axis=1)

        # Find the PFCands associate with selected jets. Search from jetindex->JetPFCands->PFCand
        if "PFCands" in events.fields:
            pruned_ev.PFCands = PFCand_link(events, event_level, jetindx)

        # Match secondary vertices to the two selected jets. SV_link falls
        # back to a dR match when the Jet->SV index link is unavailable
        # (e.g. CustomSummer24, a private NanoAOD production that keeps the
        # SV collection but drops it).
        if self.selMod == "ttdilep_sf_2D" and "SV" in events.fields:
            svmass, svd3dsig = SV_link(events, event_level, jetindx2)
            pruned_ev["SelJet", "svmass"] = ak.concatenate(
                [svmass[0][:, None], svmass[1][:, None]], axis=1
            )
            pruned_ev["SelJet", "svd3dsig"] = ak.concatenate(
                [svd3dsig[0][:, None], svd3dsig[1][:, None]], axis=1
            )

        if (
            self.selMod == "ttdilep_sf_2D"
            and not isRealData
            and "nBHadrons" in pruned_ev.SelJet.fields
        ):
            pruned_ev["SelJet", "fineFlav"] = fineflav(pruned_ev.SelJet)

        if self.selMod == "ttdilep_sf_2D":
            pruned_ev["MET"] = event_MET[event_level]

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
                perJet=(self.selMod == "ttdilep_sf_2D"),
            )
        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
