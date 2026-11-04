import awkward as ak, numpy as np
import correctionlib
import os
from coffea import processor

from BTVNanoCommissioning.utils.correction import (
    load_lumi,
    load_SF,
    weight_manager,
    common_shifts,
    reweighting,
)
from BTVNanoCommissioning.helpers.func import (
    update,
    dump_lumi,
    SV_link,
    accumulate_shifts,
)
from BTVNanoCommissioning.helpers.update_branch import missing_branch
from BTVNanoCommissioning.utils.histogramming.histogrammer import (
    histogrammer,
    histo_writer,
)
from BTVNanoCommissioning.utils.histogramming.histograms.dijet_2D import (
    TAG_TAGGERS,
    available_tag_taggers,
    book_tag_split,
    dijet_writer,
)
from BTVNanoCommissioning.utils.array_writer import array_writer
from BTVNanoCommissioning.utils.selection import (
    jet_id,
    MET_filters,
    btag_wp,
    fineflav,
    jet_veto_lepton_clean,
)

# Prescaled trigger ladders, keyed by the pT window each path is used in. The
# window edges are the ones already used in workflows/dijet.py (~1.15-1.5 x
# threshold, i.e. safely on plateau) and are shared by both ladders.
_WINDOWS = [
    [60, 86],
    [86, 110],
    [110, 170],
    [170, 236],
    [236, 305],
    [305, 375],
    [375, 460],
    [460, 575],
    [575, 1e7],
]
_THRESHOLDS = [40, 60, 80, 140, 200, 260, 320, 400, 500]

# Two ways to trigger the same tag-and-probe pair:
#
# "DiPFJetAve" -- the window variable is the average pT of the two leading
#   jets, so BOTH jets enter the trigger decision. Used with a balanced pair
#   (asymmetry cut), which ties the probe pT to pT_ave and therefore floors it
#   at ~60 GeV: the 25-35 and 35-50 GeV bins of the SF pT grid stay empty.
#
# "PFJet" -- single-jet ladder, window variable is the LEADING jet pT. The
#   leading jet is the tag, so the probe never enters the trigger decision --
#   strictly cleaner for tag-and-probe -- and dropping the balance cut lets the
#   probe go down to 20 GeV while the tag keeps the trigger on plateau. This is
#   what reaches the bottom of the SF pT grid.
TRIGGER_LADDERS = {
    "DiPFJetAve": {f"DiPFJetAve{thr}": win for thr, win in zip(_THRESHOLDS, _WINDOWS)},
    "PFJet": {f"PFJet{thr}": win for thr, win in zip(_THRESHOLDS, _WINDOWS)},
}

# Jet fields copied onto the stored TagJet record. Deliberately the same
# content as the probe (SelJet) carries in the SLIM_DIJET tree, so the swapped
# (tag <-> probe) pairing can be built offline without re-running.
_TAGJET_FIELDS = [
    "pt",
    "eta",
    "phi",
    "mass",
    "hadronFlavour",
    "partonFlavour",
    "nSVs",
    "btagPNetHFvLF",
    "btagPNetBvC",
    "btagUParTAK4HFvLF",
    "btagUParTAK4BvC",
    "btagNoPIDParTAK4HFvLF",
    "btagNoPIDParTAK4BvC",
]


class NanoProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year="2024",
        campaign="CustomSummer24",
        name="",
        isSyst=False,
        isArray=False,
        noHist=False,
        chunksize=75000,
        ttbar_reweights="none",
        selectionModifier="DiPFJetAve",
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
        if self.selMod not in TRIGGER_LADDERS:
            raise ValueError(self.selMod, "is not a valid selection modifier.")
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

        # The away jet is classified with every 2D tagger the tuple carries, so
        # the tag side of the fit can be defined by either of them. The first
        # available one is the primary: the tag-side event *selection* cuts on
        # that one alone, so it is the only one whose classification is a
        # partition of the selected sample by construction (see below).
        # NoPIDParTAK4 only exists in the private CustomSummer24 tuples; central
        # NanoAOD falls back to UParTAK4.
        tag_taggers = available_tag_taggers(events.Jet.fields)
        if not tag_taggers:
            raise ValueError(
                f"No 2D tagger category branch in {dataset}: the dijet 2D workflow "
                "needs btagNoPIDParTAK42Dbin or btagUParTAK42Dbin."
            )
        tagger = TAG_TAGGERS[tag_taggers[0]]
        bin2d = f"btag{tagger}2Dbin"

        ## Create histograms
        output = {}
        if not self.noHist:
            output = histogrammer(
                events.Jet.fields,
                obj_list=["jet0"],
                hist_collections=["common", "fourvec", "SF2D", "dijet_2D"],
                njet=1,
                sf2d_svmass=True,
            )
            output = book_tag_split(output, tag_taggers)

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

        req_metfilter = MET_filters(events, self._campaign)

        ## Jet cuts: tracker acceptance, so both jets are taggable. Cleaned
        ## against the shared veto-lepton collections so the probe is the same
        ## jet object the lepton channels calibrate (Run 2 parity); effect is
        ## per-mille in QCD but removes e.g. W+jets contamination jets that
        ## overlap an isolated lepton.
        jet_sel = jet_id(
            events, self._campaign, max_eta=2.5, min_pt=20
        ) & jet_veto_lepton_clean(events, self._campaign)
        event_jet = events.Jet[jet_sel]
        jetindx = ak.local_index(events.Jet.pt)[jet_sel]
        req_2jet = ak.num(event_jet.pt, axis=1) >= 2

        jets = ak.pad_none(event_jet, 3, axis=1)
        j0, j1, j2 = jets[:, 0], jets[:, 1], jets[:, 2]

        ## Dijet topology
        balanced = self.selMod == "DiPFJetAve"
        ptave = 0.5 * (j0.pt + j1.pt)
        dphi_jj = abs(j0.delta_phi(j1))
        req_dphi = ak.fill_none(dphi_jj > 2.7, False)
        asym = abs(j0.pt - j1.pt) / (j0.pt + j1.pt)
        # The balance cut is what ties the probe pT to pT_ave, so it is exactly
        # what has to go in the single-jet mode that reaches low probe pT.
        if balanced:
            req_bal = ak.fill_none(asym < 0.2, False)
        else:
            req_bal = np.ones(len(events), dtype="bool")

        ## Third-jet activity: a hard third jet spoils the pT balance, a close
        ## one leaks energy (and tracks) into the probe. Both are cut on, and
        ## both variables are written out so the cut can be varied downstream.
        ## Scaled to the leading jet in the unbalanced mode, where pT_ave is no
        ## longer representative of the hard scale.
        alpha_ref = ptave if balanced else j0.pt
        alpha = ak.fill_none(j2.pt / alpha_ref, 0.0)
        req_alpha = alpha < 0.15
        dr_j3 = ak.fill_none(
            ak.min(
                ak.concatenate(
                    [ak.singletons(j0.delta_r(j2)), ak.singletons(j1.delta_r(j2))],
                    axis=1,
                ),
                axis=1,
            ),
            99.0,
        )
        req_dr_j3 = dr_j3 > 0.8

        ## Trigger: each path only in its own window of the ladder's own
        ## variable -- pT_ave for DiPFJetAve, leading jet pT for PFJet
        triggers = TRIGGER_LADDERS[self.selMod]
        trig_var = ptave if balanced else j0.pt
        trig_masks = {}
        for trg, (lo, hi) in triggers.items():
            if not hasattr(events.HLT, trg):
                continue
            trig_masks[trg] = ak.fill_none(
                events.HLT[trg] & (trig_var >= lo) & (trig_var < hi), False
            )
        if not trig_masks:
            raise ValueError(f"None of the {self.selMod} paths exist in", dataset)
        req_trig = np.zeros(len(events), dtype="bool")
        for m in trig_masks.values():
            req_trig = req_trig | ak.to_numpy(m)

        ## Tag and probe. The away jet must sit in one of the three flavour-
        ## enriching categories of the tagger being calibrated: b-tight,
        ## c-tight, or the untagged (light) category. Which of the two leading
        ## jets is the probe is decided by the event number -- deterministic,
        ## so an event keeps the same assignment across JES/JER shifts, which a
        ## fresh RNG draw per shift would not.
        tag_b = btag_wp(jets, self._year, self._campaign, tagger, "2Db", "T")
        tag_c = btag_wp(jets, self._year, self._campaign, tagger, "2Dc", "T")
        tag_l = jets[bin2d] == 0
        tag_qual = ak.fill_none(tag_b | tag_c | tag_l, False)
        q0, q1 = tag_qual[:, 0], tag_qual[:, 1]
        if balanced:
            # probe = jet0 requires jet1 to be a valid tag, and vice versa; when
            # both work, split evenly on the event number
            probe_is_0 = ak.where(q0 & q1, (events.event % 2) == 0, q1)
            req_tag = q0 | q1
        else:
            # The leading jet is the trigger object, so it has to be the tag --
            # that is what keeps the probe out of the trigger decision.
            probe_is_0 = np.zeros(len(events), dtype="bool")
            req_tag = q0

        # probe must be taggable at all -- 2Dbin == -1 means the tagger scores
        # were unset for that jet -- and inside the calibrated pT grid, which
        # only bites in the unbalanced mode where the probe is free to be soft
        probe_bin = ak.where(probe_is_0, jets[bin2d][:, 0], jets[bin2d][:, 1])
        probe_pt = ak.where(probe_is_0, j0.pt, j1.pt)
        # the 20-25 GeV band is deliberate: it fills only the underflow of the
        # 25-based SF pT grid and serves as a below-grid control
        req_probe = ak.fill_none((probe_bin >= 0) & (probe_pt >= 20.0), False)

        event_level = ak.fill_none(
            req_lumi
            & req_metfilter
            & req_2jet
            & req_trig
            & req_dphi
            & req_bal
            & req_alpha
            & req_dr_j3
            & req_tag
            & req_probe,
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

        sjets = event_jet[event_level]
        spick0 = probe_is_0[event_level]
        sprobe = ak.where(spick0, sjets[:, 0], sjets[:, 1])
        stag = ak.where(spick0, sjets[:, 1], sjets[:, 0])

        # raw Jet-collection indices of the probe/tag, for the SV match
        idx = ak.pad_none(jetindx, 2, axis=1)
        probe_rawidx = ak.where(probe_is_0, idx[:, 0], idx[:, 1])
        tag_rawidx = ak.where(probe_is_0, idx[:, 1], idx[:, 0])

        pruned_ev = events[event_level]
        pruned_ev["SelJet"] = sprobe
        # Match secondary vertices to probe and tag. SV_link falls back to a dR
        # match when the Jet->SV index link is unavailable (e.g. the private
        # CustomSummer24 tuples, which keep SV but drop Jet_svIdx1).
        tag_sv = None
        if "SV" in events.fields:
            svmass, svd3dsig = SV_link(events, event_level, probe_rawidx)
            pruned_ev["SelJet", "svmass"] = svmass
            pruned_ev["SelJet", "svd3dsig"] = svd3dsig
            tag_sv = SV_link(events, event_level, tag_rawidx)

        tag_zip = {f: stag[f] for f in _TAGJET_FIELDS if f in stag.fields}
        if tag_sv is not None:
            tag_zip["svmass"], tag_zip["svd3dsig"] = tag_sv
        # Fine MC flavour on BOTH sides (selection.fineflav): the probe for the
        # uds/g and b/bb composition of the calibrated sample, the tag for the
        # true origin purity of the tagb/tagc/tagl classes. MC only.
        if not isRealData and "nBHadrons" in stag.fields:
            pruned_ev["SelJet", "fineFlav"] = fineflav(pruned_ev.SelJet)
            tag_zip["fineFlav"] = fineflav(stag)
        pruned_ev["TagJet"] = ak.zip(tag_zip)
        # Which of the three flavour-enriching tag categories the away jet fell
        # in, per tagger: 2 = b-tight, 1 = c-tight, 0 = everything else. The
        # three are disjoint and cover the sample, so the tag-class-split
        # templates always add back up to the inclusive one.
        #
        # For the primary tagger, class 0 is exactly the untagged category
        # (2Dbin == 0): req_tag already restricted the away jet to b-tight,
        # c-tight or untagged in that tagger. For any *other* tagger no such
        # requirement was made, so its class 0 is "neither b- nor c-tight" and
        # also picks up the intermediate categories (2Dbin 40/41/50/51) --
        # a coarser tag-side category, not a light-jet one. tag_2Dbin_<tagger>
        # is written out per tagger so that split stays recoverable offline.
        for short in tag_taggers:
            full = TAG_TAGGERS[short]
            pruned_ev[f"tag_2Dbin_{short}"] = stag[f"btag{full}2Dbin"]
            pruned_ev[f"tag_cls_{short}"] = ak.values_astype(
                ak.where(
                    btag_wp(stag, self._year, self._campaign, full, "2Db", "T"),
                    2,
                    ak.where(
                        btag_wp(stag, self._year, self._campaign, full, "2Dc", "T"),
                        1,
                        0,
                    ),
                ),
                np.int32,
            )

        pruned_ev["njet"] = ak.num(sjets.pt, axis=1)
        pruned_ev["ptave"] = ptave[event_level]
        pruned_ev["alpha"] = alpha[event_level]
        pruned_ev["asym"] = asym[event_level]
        pruned_ev["dphi_jj"] = dphi_jj[event_level]
        pruned_ev["dr_j3"] = dr_j3[event_level]

        ####################
        #     Output       #
        ####################

        weights = weight_manager(
            pruned_ev,
            self.SF_map,
            self.isSyst,
            ttbar_reweights=self.ttbar_reweights,
            campaign=self._campaign,
        )

        # Prescale weight: the DiPFJetAve ladder is prescaled, so data has to
        # be reweighted per (run, lumisection, path) before it can be compared
        # to MC. Same tables as workflows/dijet.py.
        if isRealData:
            run_num = {
                "2022": "355374_362760",
                "2023": "366727_370790",
                "2024": "378985_386951",
                "2025": "391658_398860",
            }.get(self._year)
            if run_num is None:
                raise NotImplementedError(
                    f"Prescale weights not available for data in {self._year}."
                )
            psweight = np.zeros(len(pruned_ev))
            for trg in trig_masks:
                psfile = f"src/BTVNanoCommissioning/data/Prescales/ps_weight_{trg}_year{self._year}.json"
                if not os.path.isfile(psfile):
                    psfile = f"src/BTVNanoCommissioning/data/Prescales/ps_weight_{trg}_run{run_num}.json"
                if not os.path.isfile(psfile):
                    raise NotImplementedError(
                        f"Prescale weights not available for {trg} in {self._year}. "
                        "Please run `scripts/dump_prescale.py`."
                    )
                pseval = correctionlib.CorrectionSet.from_file(psfile)
                thisps = pseval["prescaleWeight"].evaluate(
                    pruned_ev.run,
                    f"HLT_{trg}",
                    ak.values_astype(pruned_ev.luminosityBlock, np.float32),
                )
                fired = ak.to_numpy(trig_masks[trg][event_level])
                psweight = np.where(fired & (psweight == 0), thisps, psweight)
            pruned_ev["psweight"] = psweight
            weights.add("psweight", psweight)

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
            output = dijet_writer(
                pruned_ev, output, weights, systematics, self.isSyst, tag_taggers
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
                schema="SLIM_DIJET",
            )

        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
