import itertools

import awkward as ak, numpy as np
from coffea import processor
from coffea.nanoevents.methods import nanoaod

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
    SV_link,
    flatten,
)
from BTVNanoCommissioning.helpers.update_branch import missing_branch
from BTVNanoCommissioning.utils.histogramming.histogrammer import (
    histogrammer,
    replay_fills,
)
from BTVNanoCommissioning.utils.histogramming.histograms.sf2d import (
    SCHEMES as SF2D_SCHEMES,
)
from BTVNanoCommissioning.utils.array_writer import array_writer
from BTVNanoCommissioning.utils.selection import (
    HLT_helper,
    jet_id,
    mu_promptmvaid,
    ele_promptmvaid,
    mu_idiso,
    ele_mvatightid,
    MET_filters,
    softmu_mask,
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
        selectionModifier="WcM",
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

        isMu = False
        isEle = False
        ### selections from Spandan
        if "WcM" in self.selMod or "semittM" in self.selMod:
            triggers = ["IsoMu24"]
            isMu = True
            dxySigcut = 1.0
            muNeEmSum = 0.7
            muonpTratioCut = 0.4
            ### remove muNeEmSum for cutbased
            if "cutbased" in self.selMod:
                muNeEmSum = 1.0
            ### remove muNeEmSum and loosen muon EF for noMuVeto
            if "noMuVeto" in self.selMod:
                muNeEmSum = 1.0
                muonpTratioCut = 0.6
            isolepdz, isolepdxy, isolepsip3d = 0.01, 0.002, 2.0
        elif "WcE" in self.selMod or "semittE" in self.selMod:
            # TODO: check other trigger?
            triggers = ["Ele30_WPTight_Gsf"]
            isEle = True
            dxySigcut = 0.0
            muNeEmSum = 1.0
            muonpTratioCut = 0.6
            isolepdz, isolepdxy, isolepsip3d = 0.02, 0.01, 2.5
        else:
            raise ValueError(self.selMod, "is not a valid selection modifier.")
        # Run 2 parity for the 2D SF selections: tighter soft-muon/jet pT
        # ratio in both channels
        if self.selMod in ["WcM_2D", "WcE_2D"]:
            muonpTratioCut = 0.5

        output = {}
        if not self.noHist:
            output = histogrammer(
                events.Jet.fields,
                obj_list=["hl", "soft_l", "MET", "dilep", "mujet"],
                hist_collections=["common", "fourvec", "Wc", "SF2D"],
                include_nmujet=True,
                include_nsoftmu=True,
                include_osss=True,
                cutbased=("cutbased" in self.selMod),
                year=self._year,
                campaign=self._campaign,
                # MuonJet.svmass is valid since the jetindx fix (111b17f);
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
        req_trig = HLT_helper(events, triggers)

        ## Lepton cuts
        if isMu:
            # muon twiki: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2
            if self._campaign in [
                "Summer24",
                "CustomSummer24",
                "CustomSummer24UParTPaper",
                "CustomSummer25",
                "Winter25",
                "Prompt25",
            ]:  # NanoAODv15
                iso_lep = events.Muon[(events.Muon.pt > 26) & mu_promptmvaid(events)]
            else:
                iso_lep = events.Muon[(events.Muon.pt > 26) & mu_idiso(events)]
        elif isEle:
            # intended (2D SF selection): isolated electron |eta| < 1.5
            if self._campaign in [
                "Summer24",
                "CustomSummer24",
                "CustomSummer24UParTPaper",
                "CustomSummer25",
                "Winter25",
                "Prompt25",
            ]:  # NanoAODv15
                iso_lep = events.Electron[
                    (events.Electron.pt > 32)
                    & (abs(events.Electron.eta) < 1.5)
                    & ele_promptmvaid(events, self._campaign)
                ]
            else:
                iso_lep = events.Electron[
                    (events.Electron.pt > 32)
                    & (abs(events.Electron.eta) < 1.5)
                    & ele_mvatightid(events, self._campaign)
                ]
        req_lep = ak.count(iso_lep.pt, axis=1) == 1

        iso_lep = ak.pad_none(iso_lep, 1, axis=1)
        iso_lep = iso_lep[:, 0]
        # sub_lep = events.Muon[
        #     (events.Muon.pt < 30) &  (abs(events.Muon.eta) < 2.4) & (events.Muon.tightId > 0.5)
        # ]
        # sub_lep = ak.pad_none(sub_lep, 1, axis=1)
        # sub_lep = sub_lep[:, 0]

        """
        if isMu:
            iso_lepindx = ak.mask(
                ak.local_index(events.Muon.pt),
                ((events.Muon.pt > 30) & mu_idiso(events)) == 1,
            )
        elif isEle:
            iso_lepindx = ak.mask(
                ak.local_index(events.Electron.pt),
                ((events.Electron.pt > 34) & ele_mvatightid(events, self._campaign))
                == 1,
            )
        iso_lepindx = ak.pad_none(iso_lepindx, 1)
        iso_lepindx = iso_lepindx[:, 0]
        """

        ## Jet cuts
        # additional cleaning against the loose veto-lepton collections
        # (Run 2 parity); None-handling documented in jet_veto_lepton_clean.
        req_clean_veto = jet_veto_lepton_clean(events, self._campaign)
        jet_sel = ak.fill_none(
            jet_id(events, self._campaign, min_pt=25)
            & (ak.all(events.Jet.metric_table(iso_lep) > 0.4, axis=2))
            & req_clean_veto,
            False,
            axis=-1,
        )
        event_jet = events.Jet[jet_sel]
        nseljet = ak.count(event_jet.pt, axis=1)
        # req_base_jets = (nseljet >= 1) & (nseljet <= 3)
        req_jets = nseljet == 1

        ## Soft Muon cuts
        soft_muon = events.Muon[
            (events.Muon.pt < 25)
            & (abs(events.Muon.eta) < 2.4)
            & (events.Muon.tightId > 0.5)
            & (events.Muon.pfRelIso04_all > 0.25)
        ]
        soft_muon_tight = soft_muon[
            (abs(soft_muon.dxy / soft_muon.dxyErr) > dxySigcut)
            & (soft_muon.jetIdx != -1)
        ]
        req_softmu = ak.count(soft_muon.pt, axis=1) >= 1
        req_softmu_tight = ak.count(soft_muon_tight.pt, axis=1) >= 1
        req_softmu_pt = ak.count(soft_muon[soft_muon.pt > 5.0].pt, axis=1) >= 1
        mujetsel = ak.fill_none(
            (
                (ak.all(event_jet.metric_table(soft_muon) <= 0.4, axis=2))
                & ((event_jet.muonIdx1 != -1) | (event_jet.muonIdx2 != -1))
                & ((event_jet.muEF + event_jet.neEmEF) < muNeEmSum)
                & (event_jet.pt > 20)
                & ((event_jet.pt / event_jet.E) > 0.03)
            ),
            False,
            axis=-1,
        )
        mujetsel2 = ak.fill_none(
            (
                ((events.Jet.muEF + events.Jet.neEmEF) < muNeEmSum)
                & (
                    ak.all(
                        events.Jet.metric_table(soft_muon) <= 0.4,
                        axis=2,
                        mask_identity=True,
                    )
                )
                & ((events.Jet.muonIdx1 != -1) | (events.Jet.muonIdx2 != -1))
            ),
            False,
            axis=-1,
        )
        soft_muon = ak.pad_none(soft_muon, 1, axis=1)
        soft_muon["dxySig"] = soft_muon.dxy / soft_muon.dxyErr

        ## Muon-jet cuts
        event_jet["isMuonJet"] = mujetsel
        mu_jet = event_jet[mujetsel]
        otherjets = event_jet[~mujetsel]
        req_mujet = ak.num(mu_jet.pt, axis=1) >= 1
        mu_jet = ak.pad_none(mu_jet, 1, axis=1)

        # jet energy fraction cuts
        muNeEmSum_sel = ak.fill_none(
            (event_jet.muEF + event_jet.neEmEF) < muNeEmSum,
            False,
            axis=-1,
        )
        req_muNeEmSum = ak.num(event_jet[muNeEmSum_sel].pt, axis=1) == 1
        muEF_sel = ak.fill_none(
            (event_jet.muEF < 0.5),
            False,
            axis=-1,
        )
        req_muEF = ak.num(event_jet[muEF_sel].pt, axis=1) == 1

        ## store jet index for PFCands, create mask on the jet index
        jet_selpf = (jet_sel) & (mujetsel2)
        jetindx = ak.mask(ak.local_index(events.Jet.pt), jet_selpf == True)
        jetindx = ak.pad_none(jetindx, 1)
        jetindx = jetindx[:, 0]

        ## raw jet index of the jet actually stored as MuonJet/SelJet
        ## (sjets[:, 0]). `jetindx` above cannot be used for this: ak.mask keeps
        ## positions, so it holds raw Jet slot 0 (None whenever that slot is not
        ## itself the muon-tagged jet), not the leading jet_sel jet.
        jetindx1 = ak.pad_none(ak.local_index(events.Jet.pt)[jet_sel], 1)[:, 0]

        # Other cuts
        # Ratio between soft muon pt and muon jet pt
        # req_pTratio = (soft_muon[:, 0].pt / mu_jet[:, 0].pt) < muonpTratioCut
        idx = np.where(iso_lep.jetIdx == -1, 0, iso_lep.jetIdx)
        ## Additional cut to reject QCD events, used in BTV-20-001
        # req_QCDveto = (
        #     (iso_lep.pfRelIso04_all < 0.05)
        # & (abs(iso_lep.dz) < isolepdz)
        # & (abs(iso_lep.dxy) < isolepdxy)
        # & (iso_lep.sip3d < isolepsip3d)
        # & (
        #     iso_lep.pt
        #     / ak.firsts(
        #         events.Jet[
        #             (events.Jet.muonIdx1 == iso_lepindx)
        #             | ((events.Jet.muonIdx2 == iso_lepindx))
        #         ].pt
        #     )
        #     > 0.75
        # )
        # )

        ## Dilepton veto
        if self._campaign in [
            "Summer24",
            "CustomSummer24",
            "CustomSummer24UParTPaper",
            "CustomSummer25",
            "Winter25",
            "Prompt25",
        ]:  # NanoAODv15
            dilep_mu = events.Muon[(events.Muon.pt > 12) & mu_promptmvaid(events)]
        else:
            dilep_mu = events.Muon[(events.Muon.pt > 12) & mu_idiso(events)]
        if self._campaign in [
            "Summer24",
            "CustomSummer24",
            "CustomSummer24UParTPaper",
            "CustomSummer25",
            "Winter25",
            "Prompt25",
        ]:  # NanoAODv15
            dilep_ele = events.Electron[
                (events.Electron.pt > 15) & ele_promptmvaid(events, self._campaign)
            ]
        else:
            dilep_ele = events.Electron[
                (events.Electron.pt > 15) & ele_mvatightid(events, self._campaign)
            ]
        req_dilepveto = (
            ak.count(dilep_mu.pt, axis=1) + ak.count(dilep_ele.pt, axis=1) != 2
        )

        dilep_mass = iso_lep + soft_muon[:, 0]
        if isMu and "noMuVeto" not in self.selMod:
            req_dilepmass = (dilep_mass.mass > 12.0) & (
                (dilep_mass.mass < 80) | (dilep_mass.mass > 100)
            )
        else:
            req_dilepmass = iso_lep.pt > 0

        ## MET and Transverse W mass
        iso_lep_trans = ak.zip(
            {
                "pt": iso_lep.pt,
                "eta": ak.zeros_like(iso_lep.pt),
                "phi": iso_lep.phi,
                "mass": iso_lep.mass,
            },
            with_name="PtEtaPhiMLorentzVector",
        )

        MET = ak.zip(
            {
                "pt": events.PuppiMET.pt,
                "eta": ak.zeros_like(events.PuppiMET.pt),
                "phi": events.PuppiMET.phi,
                "mass": ak.zeros_like(events.PuppiMET.pt),
            },
            with_name="PtEtaPhiMLorentzVector",
        )

        req_metfilter = MET_filters(events, self._campaign)
        event_jet_0 = ak.pad_none(event_jet, 1, axis=1)
        event_jet_0 = event_jet_0[:, 0]
        # leading soft muon inside the selected jet (c purity of OS-SS ~78 % -> ~92 %)
        req_softmu_injet = ak.fill_none(
            event_jet_0.delta_r(soft_muon[:, 0]) <= 0.4, False
        )
        req_met_pt = MET.pt > 30.0
        jetmet_dphi = abs(event_jet_0.delta_phi(MET))
        req_jetmet_dphi = jetmet_dphi > 1.0
        trkmet = (
            events.TrkMET if int(self._year) >= 2024 else events.TkMET
        )  # NanoAODv15 uses `TrkMET`
        # TrkMET isn't registered as a MissingET mixin by NanoAODSchema (unlike
        # TkMET/MET/PuppiMET), so it has no vector behavior attached; delta_phi
        # against it used to work anyway via awkward1's looser duck typing, but
        # the current vector package requires a real registered type. Retag it
        # explicitly -- same fields, same values, only the type changes. (For
        # TkMET, already tagged MissingET by the schema, this is a no-op.)
        trkmet = ak.zip(
            {"pt": trkmet.pt, "phi": trkmet.phi, "sumEt": trkmet.sumEt},
            with_name="MissingET",
            behavior=nanoaod.behavior,
        )
        metTrkmet_dphi = abs(MET.delta_phi(trkmet))
        req_metTrkmet_dphi = metTrkmet_dphi < 1.0  # & (metTrkmet_dphi >= 0.5)

        # W cuts
        wmasscut = 40  # before 55
        wmasscut_tight = 55
        wptcut = 30
        if "semitt" in self.selMod:
            wmasscut = 0
        Wcand = MET + iso_lep_trans  # transverse mass
        Wmass = Wcand.mass
        Wpt = Wcand.pt
        req_WpT = Wpt > wptcut
        req_mtw = Wmass > wmasscut
        req_mtw_max120 = Wmass < 120
        req_mtw_min55 = Wmass > wmasscut_tight
        # req for pt ratio of leading jet and W pt
        req_pTratio = (soft_muon[:, 0].pt / event_jet_0.pt) < muonpTratioCut
        jetw_ptratio = event_jet_0.pt / Wpt
        req_jWpTratio = (jetw_ptratio < 2) & (jetw_ptratio > 0.5)
        # mask on distance between W_phi and jet_phi abs(deltaPhi(ak4jets[0].phi, Vboson.phi) > 2 (if tight sel)
        jetw_dphi = abs(event_jet_0.delta_phi(Wcand))
        req_dphi_Wjet = jetw_dphi > 2.0
        # delta Phi between leading jet and lepton; intended (2D SF selection): < 2.8 (was 2.0)
        jetl_dphi = abs(event_jet_0.delta_phi(iso_lep))
        req_jetl_dphi = jetl_dphi < 2.8

        # ==This is the manual calculation for transverse mass==
        """
        dphi = iso_lep.phi-events.PuppiMET.phi
        dphi = np.where(dphi<np.pi,dphi+2*np.pi,dphi)
        dphi = np.where(dphi>np.pi,dphi-2*np.pi,dphi)
        trans = np.sqrt(2*iso_lep.pt*events.PuppiMET.pt*(1-np.cos(dphi)))
        """

        event_level = (
            req_trig
            & req_lumi
            & req_lep
            & req_jets
            & req_softmu
            & req_softmu_injet
            & req_dilepmass
            & req_mtw
            & req_metfilter
            & req_met_pt
            & req_jetmet_dphi
            & req_metTrkmet_dphi
            & req_WpT
            & req_jWpTratio
            & req_dphi_Wjet
            & req_jetl_dphi
            & req_muNeEmSum
            & req_muEF
            & req_softmu_pt
            & req_pTratio
            & req_mtw_max120
        )

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

        shmu = iso_lep[event_level]
        wm = Wmass[event_level]
        wp = Wpt[event_level]
        sjets = event_jet[event_level]
        ssmu = soft_muon[event_level]
        smet = MET[event_level]
        smuon_jet = mu_jet[event_level]
        sotherjets = otherjets[event_level]
        sdilep = dilep_mass[event_level]
        nsoftmu = ak.count(ssmu.pt, axis=1)
        # nmujet = ak.count(smuon_jet.pt, axis=1)
        # smuon_jet = smuon_jet[ak.num(smuon_jet, axis=1) > 0]
        # smuon_jet = ak.pad_none(smuon_jet, 1, axis=1)  # Ensure at least one entry
        # smuon_jet = smuon_jet[~ak.is_none(smuon_jet, axis=1)]  # Remove None entries muon_jet = smuon_jet[:, 0]
        smuon_jet = sjets[:, 0]
        ssmu = ssmu[:, 0]
        sz = shmu + ssmu
        # Electron/Muon (Candidate behavior) + MET (plain PtEtaPhiMLorentzVector)
        # has no registered numpy.add overload under the new vector package;
        # strip shmu to a plain LorentzVector (same pt/eta/phi/mass, no
        # candidate-specific fields) before summing with MET -- same pattern
        # already used for iso_lep_trans above, just without zeroing eta.
        shmu_lv = ak.zip(
            {"pt": shmu.pt, "eta": shmu.eta, "phi": shmu.phi, "mass": shmu.mass},
            with_name="PtEtaPhiMLorentzVector",
        )
        sw = shmu_lv + smet

        osss = shmu.charge * ssmu.charge * -1
        ossswrite = shmu.charge * ssmu.charge * -1

        njet = ak.count(sjets.pt, axis=1)
        # Find the PFCands associate with selected jets, search from jetindex->JetPFCands->PFCand
        if "PFCands" in events.fields:
            spfcands = PFCand_link(events, event_level, jetindx)

        # Keep the structure of events and pruned the object size
        pruned_ev = events[event_level]
        pruned_ev["SelJet"] = sjets
        # lead_jet = sjets[:, 0]
        # pruned_ev["LeadingJet"] = lead_jet
        if self.selMod.endswith("M") or self.selMod == "WcM_2D":
            pruned_ev["SelMuon"] = shmu
        else:
            pruned_ev["SelElectron"] = shmu
        pruned_ev["MuonJet"] = (
            smuon_jet  # leading selected jet fullfilling all event-level cuts
        )
        # Match secondary vertices to the muon-tagged jet. SV_link falls
        # back to a dR match when the Jet->SV index link is unavailable
        # (e.g. CustomSummer24, a private NanoAOD production that keeps the
        # SV collection but drops it).
        if "SV" in events.fields:
            svmass, svd3dsig = SV_link(events, event_level, jetindx1)
            pruned_ev["MuonJet", "svmass"] = svmass
            pruned_ev["MuonJet", "svd3dsig"] = svd3dsig
        if not isRealData and "nBHadrons" in pruned_ev.SelJet.fields:
            pruned_ev["SelJet", "fineFlav"] = fineflav(pruned_ev.SelJet)
            pruned_ev["MuonJet", "fineFlav"] = fineflav(pruned_ev.MuonJet)
        pruned_ev["SoftMuon"] = ssmu
        pruned_ev["OtherJets"] = sotherjets
        pruned_ev["MET_pt"] = smet.pt
        if "Wc" in self.selMod:
            pruned_ev["osss"] = osss
        else:
            pruned_ev["osss"] = ossswrite
        pruned_ev["njet"] = njet
        pruned_ev["w_mass"] = wm
        pruned_ev["w_pt"] = wp
        pruned_ev["dilep_mass"] = sdilep.mass
        pruned_ev["dilep_pt"] = sdilep.pt
        pruned_ev["jetw_dphi"] = jetw_dphi[event_level]
        pruned_ev["jetmet_dphi"] = jetmet_dphi[event_level]
        pruned_ev["metTrkmet_dphi"] = metTrkmet_dphi[event_level]
        pruned_ev["jetl_dphi"] = jetl_dphi[event_level]
        if "PFCands" in events.fields:
            pruned_ev.PFCands = spfcands

        # Add custom variables
        pruned_ev["dr_mujet_softmu"] = ssmu.delta_r(smuon_jet)
        pruned_ev["dr_mujet_lep1"] = shmu.delta_r(smuon_jet)
        pruned_ev["dr_lep1_softmu"] = shmu.delta_r(ssmu)
        pruned_ev["soft_l_ptratio"] = ssmu.pt / smuon_jet.pt
        pruned_ev["l1_ptratio"] = shmu.pt / smuon_jet.pt
        pruned_ev["MuonJet_beta"] = smuon_jet.pt / smuon_jet.E
        pruned_ev["MuonJet_muneuEF"] = smuon_jet.muEF + smuon_jet.neEmEF
        pruned_ev["jetw_ptratio"] = jetw_ptratio[event_level]

        if "hadronFlavour" in pruned_ev.SelJet.fields:
            isRealData = False
            genflavor = ak.values_astype(
                pruned_ev.SelJet.hadronFlavour
                + 1
                * (
                    (pruned_ev.SelJet.partonFlavour == 0)
                    & (pruned_ev.SelJet.hadronFlavour == 0)
                ),
                int,
            )
            if "MuonJet" in pruned_ev.fields:
                smflav = ak.values_astype(
                    1
                    * (
                        (pruned_ev.MuonJet.partonFlavour == 0)
                        & (pruned_ev.MuonJet.hadronFlavour == 0)
                    )
                    + pruned_ev.MuonJet.hadronFlavour,
                    int,
                )
        else:
            isRealData = True
            genflavor = ak.zeros_like(pruned_ev.SelJet.pt, dtype=int)
            if "MuonJet" in pruned_ev.fields:
                smflav = ak.zeros_like(pruned_ev.MuonJet.pt, dtype=int)

        # Store masks for additional cuts
        # pruned_ev["mask_njets"] = req_jets[event_level]
        pruned_ev["mask_mujet"] = req_mujet[event_level]
        pruned_ev["mask_muNeEmSum"] = req_muNeEmSum[event_level]
        pruned_ev["mask_muEF"] = req_muEF[event_level]
        pruned_ev["mask_jetl_dphi"] = req_jetl_dphi[event_level]
        pruned_ev["mask_dilepveto"] = req_dilepveto[event_level]
        pruned_ev["mask_smj_ptratio"] = req_pTratio[event_level]
        pruned_ev["mask_met_pt"] = req_met_pt[event_level]
        pruned_ev["mask_jetmet_dphi"] = req_jetmet_dphi[event_level]
        pruned_ev["mask_metTrkmet_dphi"] = req_metTrkmet_dphi[event_level]
        pruned_ev["req_mtw_max120"] = req_mtw_max120[event_level]
        pruned_ev["req_mtw_min55"] = req_mtw_min55[event_level]
        pruned_ev["mask_wpt"] = req_WpT[event_level]
        pruned_ev["mask_softmu_tight"] = req_softmu_tight[event_level]
        pruned_ev["mask_softmu_pt"] = req_softmu_pt[event_level]
        pruned_ev["mask_jWpTratio"] = req_jWpTratio[event_level]
        pruned_ev["mask_dphi_Wjet"] = req_dphi_Wjet[event_level]

        ####################
        # Weight & Geninfo #
        ####################

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

        ####################
        #  Fill histogram  #
        ####################

        def weight_for(syst):
            if syst == "nominal" or syst == shift_name:
                return weights.weight()
            return weights.weight(modifier=syst)

        def fill_all(output, weight, syst):
            syst = np.full(len(weight), syst)
            for histname, h in output.items():
                if (
                    "Deep" in histname
                    and "btag" not in histname
                    and histname in events.Jet.fields
                ):
                    h.fill(
                        syst,
                        flatten(genflavor),
                        flatten(ak.broadcast_arrays(osss, sjets["pt"])[0]),
                        flatten(sjets[histname]),
                        weight=flatten(ak.broadcast_arrays(weight, sjets["pt"])[0]),
                    )
                elif (
                    "PFCands" in events.fields
                    and "PFCands" in histname
                    and histname.split("_")[1] in events.PFCands.fields
                ):
                    h.fill(
                        syst,
                        flatten(ak.broadcast_arrays(smflav, spfcands["pt"])[0]),
                        flatten(ak.broadcast_arrays(osss, spfcands["pt"])[0]),
                        flatten(spfcands[histname.replace("PFCands_", "")]),
                        weight=flatten(ak.broadcast_arrays(weight, spfcands["pt"])[0]),
                    )
                elif "jet_" in histname and "mu" not in histname:
                    h.fill(
                        syst,
                        flatten(genflavor),
                        flatten(ak.broadcast_arrays(osss, sjets["pt"])[0]),
                        flatten(sjets[histname.replace("jet_", "")]),
                        weight=flatten(ak.broadcast_arrays(weight, sjets["pt"])[0]),
                    )
                elif "hl_" in histname and histname.replace("hl_", "") in shmu.fields:
                    h.fill(
                        syst,
                        osss,
                        flatten(shmu[histname.replace("hl_", "")]),
                        weight=weight,
                    )
                elif (
                    "soft_l" in histname
                    and histname.replace("soft_l_", "") in ssmu.fields
                ):
                    h.fill(
                        syst,
                        smflav,
                        osss,
                        flatten(ssmu[histname.replace("soft_l_", "")]),
                        weight=weight,
                    )
                elif (
                    "mujet_" in histname
                    and histname.replace("mujet_", "") in smuon_jet.fields
                ):
                    h.fill(
                        syst,
                        smflav,
                        osss,
                        flatten(smuon_jet[histname.replace("mujet_", "")]),
                        weight=weight,
                    )
                elif "btag" in histname and "Trans" not in histname:
                    for i in range(2):
                        if (
                            not histname.endswith(str(i))
                            or histname.replace(f"_{i}", "") not in smuon_jet.fields
                        ):
                            continue
                        h.fill(
                            syst,
                            flav=smflav,
                            osss=osss,
                            discr=np.where(
                                smuon_jet[histname.replace(f"_{i}", "")] < 0,
                                -2,
                                smuon_jet[histname.replace(f"_{i}", "")],
                            ),
                            weight=weight,
                        )
                        if not isRealData and "btag" in self.SF_map.keys():
                            h.fill(
                                syst=syst,
                                flav=smflav,
                                osss=osss,
                                discr=np.where(
                                    smuon_jet[histname.replace(f"_{i}", "")] < 0,
                                    -0.2,
                                    smuon_jet[histname.replace(f"_{i}", "")],
                                ),
                                weight=weight,
                            )
                # elif "btag" in histname and "Trans" in histname:
                #     if histname not in smuon_jet:
                #         continue
                #     for i in range(2):
                #         histname = histname.replace("Trans", "").replace(f"_{i}", "")
                #         h.fill(
                #             syst="noSF",
                #             flav=smflav,
                #             osss=osss,
                #             discr=1.0 / np.tanh(smuon_jet[histname]),
                #             weight=weight,
                #         )

            output["njet"].fill(syst, osss, njet, weight=weight)
            # output["nmujet"].fill(syst, osss, nmujet, weight=weight)
            output["nsoftmu"].fill(syst, osss, nsoftmu, weight=weight)
            output["npv"].fill(syst, osss, pruned_ev.PV.npvsGood, weight=weight)
            if "TT" in pruned_ev.metadata["dataset"]:
                top_mask = (pruned_ev.GenPart.pdgId == 6) & pruned_ev.GenPart.hasFlags(
                    ["isLastCopy"]
                )
                top_pt = pruned_ev.GenPart[top_mask][:, 0].pt
                output["top_pt"].fill(syst, osss, top_pt, weight=weight)
                antitop_mask = (
                    pruned_ev.GenPart.pdgId == -6
                ) & pruned_ev.GenPart.hasFlags(["isLastCopy"])
                antitop_pt = pruned_ev.GenPart[antitop_mask][:, 0].pt
                output["antitop_pt"].fill(syst, osss, antitop_pt, weight=weight)
            output["softlpt"].fill(syst, smflav, osss, ssmu.pt, weight=weight)
            output["hl_ptratio"].fill(
                syst,
                genflavor[:, 0],
                osss=osss,
                ratio=shmu.pt / sjets[:, 0].pt,
                weight=weight,
            )
            output["soft_l_ptratio"].fill(
                syst,
                flav=smflav,
                osss=osss,
                ratio=ssmu.pt / smuon_jet.pt,
                weight=weight,
            )
            output["dr_lmujetsmu"].fill(
                syst,
                flav=smflav,
                osss=osss,
                dr=smuon_jet.delta_r(ssmu),
                weight=weight,
            )
            output["dr_lmujethmu"].fill(
                syst,
                flav=smflav,
                osss=osss,
                dr=smuon_jet.delta_r(shmu),
                weight=weight,
            )
            output["dr_hmusmu"].fill(
                syst,
                osss=osss,
                dr=shmu.delta_r(ssmu),
                weight=weight,
            )
            output["mujet_muneuEF"].fill(
                syst,
                smflav,
                osss=osss,
                muneuEF=flatten(pruned_ev["MuonJet_muneuEF"]),
                weight=weight,
            )
            output["dilep_pt"].fill(syst, osss, flatten(sz.pt), weight=weight)
            output["dilep_eta"].fill(syst, osss, flatten(sz.eta), weight=weight)
            output["dilep_phi"].fill(syst, osss, flatten(sz.phi), weight=weight)
            output["dilep_mass"].fill(syst, osss, flatten(sz.mass), weight=weight)
            output["w_pt"].fill(syst, osss, flatten(sw.pt), weight=weight)
            output["w_phi"].fill(syst, osss, flatten(sw.phi), weight=weight)
            output["w_mass"].fill(syst, osss, flatten(sw.mass), weight=weight)
            output["jetw_dphi"].fill(
                syst, smflav, osss, flatten(jetw_dphi[event_level]), weight=weight
            )
            output["jetw_ptratio"].fill(
                syst, smflav, osss, flatten(jetw_ptratio[event_level]), weight=weight
            )
            output["MET_pt"].fill(syst, osss, flatten(smet.pt), weight=weight)
            output["MET_phi"].fill(syst, osss, flatten(smet.phi), weight=weight)
            output["jetmet_dphi"].fill(
                syst, smflav, osss, flatten(jetmet_dphi[event_level]), weight=weight
            )
            output["metTrkmet_dphi"].fill(
                syst, osss, flatten(metTrkmet_dphi[event_level]), weight=weight
            )
            output["jetl_dphi"].fill(
                syst, smflav, osss, flatten(jetl_dphi[event_level]), weight=weight
            )

            # SF-fit templates (histograms/sf2d.py): tag category x jet pT x
            # SV variable, filled with OS/SS kept as separate "osss" categories
            # (never subtracted -- SFbc-2D/OptimizedWP/CLAUDE.md settled
            # question 7). The muon-tagged jet is the fit object here, and is
            # the jet SV_link resolves the svmass for.
            muon_jet = pruned_ev.MuonJet
            for scheme, (branch, _codes) in SF2D_SCHEMES.items():
                if branch not in muon_jet.fields:
                    continue
                for var in ("nSVs", "svmass"):
                    name = f"sf2d_{scheme}_{var}"
                    if name not in output or var not in muon_jet.fields:
                        continue
                    output[name].fill(
                        syst=syst,
                        flav=smflav,
                        osss=osss,
                        tagcat=flatten(muon_jet[branch]),
                        pt=flatten(muon_jet.pt),
                        weight=weight,
                        **{var: flatten(muon_jet[var])},
                    )

        if not self.noHist:
            systs = list(
                itertools.takewhile(
                    lambda s: not (self.isSyst == False and s != "nominal"), systematics
                )
            )
            replay_fills(output, fill_all, systs, weight_for, len(weights.weight()))

        #######################
        #  Create root files  #
        #######################

        if self.isArray:
            # write out the muon-tagged jet (the fit variable) as "SelJet", matching
            # the per-jet branch naming used by the other 2D workflows -- one row
            # per event here since there's only ever one tagged jet, not a list
            pruned_ev["SelJet"] = pruned_ev["MuonJet"]
            # always 0: there's only ever one muon-tagged jet stored as "SelJet" here
            pruned_ev["SelJet", "jetIdx"] = ak.zeros_like(
                pruned_ev.SelJet.pt, dtype=np.int64
            )
            array_writer(
                self,
                pruned_ev,
                events,
                weights,
                systematics,
                dataset,
                isRealData,
                schema="SLIM",
            )

        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
