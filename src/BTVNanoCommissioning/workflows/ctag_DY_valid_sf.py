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
from BTVNanoCommissioning.utils.histogramming.histograms.qgtag import qg_writer
from BTVNanoCommissioning.utils.histogramming.histogrammer import (
    histogrammer,
    histo_writer,
)
from BTVNanoCommissioning.utils.array_writer import array_writer
from BTVNanoCommissioning.helpers.stag_helpers import (
    TaggingBin,
    get_stag_config,
    add_stag_fields,
    stag_array_schema,
    tagger_jet_fields,
)
from BTVNanoCommissioning.utils.selection import (
    HLT_helper,
    jet_id,
    mu_promptmvaid,
    ele_promptmvaid,
    mu_idiso,
    ele_mvatightid,
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
        selectionModifier="DYM_ctag",
        stag=False,
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
        # Load corrections
        self.SF_map = load_SF(self._year, self._campaign, self.selMod)
        # stag_* workflows: s-tag bins, scores and flavours of the sfs-3d WPs
        # (helpers/stag_helpers.py) instead of the 2D tag categories
        self.stag = stag
        if stag:
            stag_cfg = get_stag_config(campaign)
            self.tagging = TaggingBin(stag_cfg["tagging_wps"])
            self.tagging.check()
            # only the outputs of this tagger are histogrammed and written to the arrays
            self.tagger = stag_cfg["tagger"]

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
        if "DYM" in self.selMod or "QG" in self.selMod:
            triggers = ["Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8"]
            isMu = True
        elif "DYE" in self.selMod:
            triggers = ["Ele23_Ele12_CaloIdL_TrackIdL_IsoVL"]
            isEle = True
        else:
            raise ValueError(self.selMod, "is not a valid selection modifier.")

        hists = ["common", "fourvec", "DY"]
        obj_list = ["posl", "negl", "dilep", "jet0"]
        njet_hist = 1
        if "2D" in self.selMod:
            hists.append("DY_2D")
            hists.append("STAG" if self.stag else "SF2D")
            # also histogram jet1, jet2 (with tagging categories) so they
            # can be used as additional categories in the global SF fit
            obj_list = obj_list + ["jet1", "jet2"]
            njet_hist = 3
        output = {}
        if not self.noHist:
            if "QG" not in self.selMod:
                output = histogrammer(
                    jet_fields=(
                        tagger_jet_fields(events.Jet.fields, self.tagger)
                        if self.stag
                        else events.Jet.fields
                    ),
                    obj_list=obj_list,
                    hist_collections=hists,
                    include_m=isMu,
                    njet=njet_hist,
                    # DY is the only channel where SelJet_svmass is valid in the
                    # current trees; see histograms/sf2d.py.
                    sf2d_svmass=True,
                    stag_flavour=self.stag,
                )
            else:
                output = histogrammer(
                    jet_fields=(
                        tagger_jet_fields(events.Jet.fields, self.tagger)
                        if self.stag
                        else events.Jet.fields
                    ),
                    obj_list=[],
                    hist_collections=["qgtag"],
                    axes_collections=["qgtag"],
                    is_dijet=False,
                )

        if shift_name is None:
            output["sumw"] = sumws["sumw"]
            # one sum per generator-level variation, see correction.reweighting
            output.update({k: v for k, v in sumws.items() if k.startswith("sumw_")})

        ####################
        #    Selections    #
        ####################

        # Lumimask
        req_lumi = np.ones(len(events), dtype="bool")
        if isRealData:
            req_lumi = self.lumiMask(events.run, events.luminosityBlock)
        # only dump for nominal case
        if shift_name is None:
            output = dump_lumi(events[req_lumi], output)

        # HLT
        req_trig = HLT_helper(events, triggers)
        req_metfilter = MET_filters(events, self._campaign)

        # Muon cuts
        if "2D" in self.selMod and self._campaign in [
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

        # Electron cuts
        if "2D" in self.selMod and self._campaign in [
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

        if isMu:
            thisdilep = dilep_mu
            otherdilep = dilep_ele
            if "2D" in self.selMod:
                req_leppt = ak.max(dilep_mu.pt, axis=1, mask_identity=False) > 20
        else:
            thisdilep = dilep_ele
            otherdilep = dilep_mu
            if "2D" in self.selMod:
                req_leppt = ak.max(dilep_ele.pt, axis=1, mask_identity=False) > 25

        # dilepton
        pos_dilep = thisdilep[thisdilep.charge > 0]
        neg_dilep = thisdilep[thisdilep.charge < 0]
        req_pl = ak.count(pos_dilep.pt, axis=1) >= 1
        req_nl = ak.count(neg_dilep.pt, axis=1) >= 1
        req_dilep_chrg = ak.num(thisdilep.charge) >= 2
        req_otherdilep_chrg = ak.num(otherdilep.charge) == 0
        req_dilep = ak.fill_none(
            req_pl & req_nl & req_dilep_chrg & req_otherdilep_chrg,
            False,
            axis=-1,
        )
        if "2D" in self.selMod:
            req_dilep = req_dilep & req_leppt

        pos_dilep = ak.pad_none(pos_dilep, 1, axis=1)
        neg_dilep = ak.pad_none(neg_dilep, 1, axis=1)

        dilep_mass = pos_dilep[:, 0] + neg_dilep[:, 0]
        req_dilepmass = (
            (dilep_mass.mass > 61) & (dilep_mass.mass < 121) & (dilep_mass.pt > 15)
        )

        # Jet cuts
        pl_iso = ak.all(
            events.Jet.metric_table(pos_dilep[:, 0]) > 0.4, axis=2, mask_identity=True
        )
        nl_iso = ak.all(
            events.Jet.metric_table(neg_dilep[:, 0]) > 0.4, axis=2, mask_identity=True
        )

        if "QG" in self.selMod:
            jetmask = jet_id(events, self._campaign, max_eta=5.0, min_pt=20.0)
        else:
            # stag_* workflows: pT > 25 GeV as for the s-tagging WPs and templates
            jetmask = jet_id(events, self._campaign, min_pt=25 if self.stag else 20)

        if "2D" in self.selMod:
            # clean against the shared loose veto-lepton collections (Run 2
            # parity); the Z leptons are a subset of these by construction.
            # None-handling documented in jet_veto_lepton_clean.
            lep_iso = jet_veto_lepton_clean(events, self._campaign)
        else:
            lep_iso = pl_iso & nl_iso

        jet_sel = ak.fill_none(
            lep_iso & jetmask,
            False,
            axis=-1,
        )

        event_jet = events.Jet[jet_sel]

        if "2D" in self.selMod:
            req_jets = ak.count(event_jet.pt, axis=1) == 1
        else:
            req_jets = ak.count(event_jet.pt, axis=1) >= 1

        # store jet index for PFCands, create mask on the jet index
        jetindx = ak.mask(
            ak.local_index(events.Jet.pt),
            jet_sel == 1,
        )
        jetindx = ak.pad_none(jetindx, 1)
        jetindx = jetindx[:, 0]

        # raw jet index for the up-to-3 selected jets stored in SelJet (2D
        # selMod); kept ragged (not padded) to match SelJet's own shape
        jetindx3 = ak.local_index(events.Jet.pt)[jet_sel]

        selection = (
            req_lumi & req_trig & req_dilep & req_dilepmass & req_jets & req_metfilter
        )

        if "2D" in self.selMod:
            # Run 2 parity: balanced Z + 1-jet topology.
            #
            # The jet/Z pT balance cut (Run 2: 0.75 < pT(jet)/pT(Z) < 1.25) is
            # DELIBERATELY NOT APPLIED. Measured on CustomSummer24 MC: it costs
            # 49 % of the probe jets -- the single most expensive cut in this
            # channel -- while moving the light-jet composition by only ~1 pp
            # (gluon fraction 0.289 -> 0.279, `undef` 0.053 -> 0.040). The
            # light-jet statistics it removes is what makes SF_flavL in the
            # b-like categories (B0/B1/B2) unmeasurable: with the cut, the
            # Asimov sigma of SF_flavL_B1 degrades from 0.100 to 0.243 and
            # B2/B4 fall below the prefit Fisher gate entirely.
            #
            # What actually keeps this sample quark-rich is taking ONE probe per
            # event instead of every selected jet (gluon fraction 0.384 -> 0.306),
            # i.e. `njet == 1` plus a single stored probe -- not the balance cut.
            # So the purity argument is served by the jet count, and the balance
            # cut only costs resolution. Keep dphi(Z,jet) and pT(Z), which are
            # cheap (-18 % combined) and remove the genuinely unbalanced tail.
            jet0 = ak.pad_none(event_jet, 1, axis=1)[:, 0]
            req_zjet_dphi = ak.fill_none(abs(dilep_mass.delta_phi(jet0)) > 2, False)
            req_dilep_pt = ak.fill_none(dilep_mass.pt > 25, False)
            selection = selection & req_zjet_dphi & req_dilep_pt

        if "QG" in self.selMod:
            temp_jet = ak.pad_none(event_jet, 1, axis=1)

            req_lead_jet = ak.fill_none(
                (
                    np.abs(temp_jet[:, 0].delta_phi(pos_dilep[:, 0] + neg_dilep[:, 0]))
                    > 2.7
                ),
                False,
                axis=-1,
            )
            selection = selection & req_lead_jet

        event_level = ak.fill_none(
            selection,
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

        sposmu = pos_dilep[event_level][:, 0]
        snegmu = neg_dilep[event_level][:, 0]
        sz = sposmu + snegmu

        sel_jet = event_jet[event_level][:, 0]

        sel_mu = ak.concatenate([sposmu, snegmu])
        smu = ak.zip(
            {
                b: ak.Array(np.reshape(sel_mu[b].to_numpy(), (len(sposmu[b]), 2)))
                for b in sposmu.fields
            }
        )

        # Keep the structure of events and pruned the object size
        pruned_ev = events[event_level]
        if self.selMod == "QG":
            pruned_ev["SelJet"] = event_jet[event_level][:, 0]
            # a plain (behavior-less) record, not sz itself -- attaching pt/eta/
            # phi/mass fields onto sz's own vector-behaved record would leave it
            # with both cartesian (x,y from the sum) and polar (pt/eta/phi/mass)
            # fields at once, which awkward2's CandidateArray validation rejects
            # on any later access (e.g. histogramming)
            pruned_ev["Tag"] = ak.zip(
                {"pt": sz.pt, "eta": sz.eta, "phi": sz.phi, "mass": sz.mass}
            )
        else:
            # keep up to 3 jets/event so jet1, jet2 (and their SV info) are
            # stored alongside jet0, not just the leading jet
            pruned_ev["SelJet"] = event_jet[event_level][:, :3]
            if (
                "2D" in self.selMod
                and not isRealData
                and "nBHadrons" in pruned_ev.SelJet.fields
            ):
                pruned_ev["SelJet", "fineFlav"] = fineflav(pruned_ev.SelJet)

        if isMu:
            pruned_ev["MuonPlus"] = sposmu
            pruned_ev["MuonMinus"] = snegmu
            pruned_ev["posl"] = sposmu
            pruned_ev["negl"] = snegmu
            pruned_ev["SelMuon"] = smu
            # kinOnly = ["SelMuon", "MuonPlus", "MuonMinus"]
        else:
            pruned_ev["ElectronPlus"] = sposmu
            pruned_ev["ElectronMinus"] = snegmu
            pruned_ev["posl"] = sposmu
            pruned_ev["negl"] = snegmu
            pruned_ev["SelElectron"] = smu
            # kinOnly = ["SelElectron", "ElectronPlus", "ElectronMinus"]
        # a plain (behavior-less) record -- see comment on the "Tag" block above
        pruned_ev["dilep"] = ak.zip(
            {"pt": sz.pt, "eta": sz.eta, "phi": sz.phi, "mass": sz.mass}
        )
        pruned_ev["njet"] = ak.count(event_jet[event_level].pt, axis=1)

        pruned_ev["dr_mu1jet"] = sposmu.delta_r(sel_jet)
        pruned_ev["dr_mu2jet"] = snegmu.delta_r(sel_jet)
        # Find the PFCands associate with selected jets. Search from jetindex->JetPFCands->PFCand
        if "PFCands" in events.fields and "QG" not in self.selMod:
            pruned_ev["PFCands"] = PFCand_link(events, event_level, jetindx)

        # Match secondary vertices to each of the (up to 3) selected jets.
        # SV_link_ragged falls back to a dR match when the Jet->SV index link
        # is unavailable (e.g. CustomSummer24, a private NanoAOD production
        # that keeps the SV collection but drops it).
        if "SV" in events.fields and "QG" not in self.selMod:
            svmass, svd3dsig = SV_link_ragged(
                events, event_level, jetindx3[event_level][:, :3]
            )
            pruned_ev["SelJet", "svmass"] = svmass
            pruned_ev["SelJet", "svd3dsig"] = svd3dsig

        # position of each entry within the stored SelJet list itself (0 =
        # leading, 1 = subleading, ...), not the original NanoAOD Jet index
        if "QG" not in self.selMod:
            pruned_ev["SelJet", "jetIdx"] = ak.local_index(pruned_ev.SelJet, axis=1)
        if self.stag and "QG" not in self.selMod:
            add_stag_fields(pruned_ev, "SelJet", self.tagging, isRealData)

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
            if "QG" not in self.selMod:
                output = histo_writer(
                    pruned_ev, output, weights, systematics, self.isSyst, self.SF_map
                )
            else:
                output = qg_writer(
                    pruned_ev,
                    output,
                    weights,
                    systematics,
                    self.isSyst,
                    self.SF_map,
                )
        # Output arrays
        if self.isArray:
            if "QG" in self.selMod:
                othersData = [
                    "SV_*",
                    "PV_npvs",
                    "PV_npvsGood",
                    "Rho_*",
                    "SoftMuon_dxySig",
                    "Muon_sip3d",
                    "run",
                    "luminosityBlock",
                ]
                for trigger in triggers:
                    othersData.append(f"HLT_{trigger}")
            else:
                othersData = [
                    "PFCands_*",
                    "MuonJet_*",
                    "SV_*",
                    "PV_npvs",
                    "PV_npvsGood",
                    "Rho_*",
                    "SoftMuon_dxySig",
                    "Muon_sip3d",
                ]

            array_writer(
                self,
                pruned_ev,
                events,
                weights,
                systematics,
                dataset,
                isRealData,
                schema=(
                    stag_array_schema(self.tagger, self.tagging.branches)
                    if self.stag
                    else "SLIM"
                ),
                perJet=("QG" not in self.selMod),
            )

        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
