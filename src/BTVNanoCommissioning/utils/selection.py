import awkward as ak
import numpy as np
from BTVNanoCommissioning.helpers.func import campaign_map


def HLT_helper(events, triggers):

    checkHLT = ak.Array([hasattr(events.HLT, _trig) for _trig in triggers])
    if ak.all(checkHLT == False):
        raise ValueError(
            "HLT paths:", triggers, " are all invalid in", events.metadata["dataset"]
        )
    elif ak.any(checkHLT == False):
        print(
            np.array(triggers)[~checkHLT], " not exist in", events.metadata["dataset"]
        )
    trig_arrs = [events.HLT[_trig] for _trig in triggers if hasattr(events.HLT, _trig)]
    req_trig = np.zeros(len(events), dtype="bool")
    for t in trig_arrs:
        req_trig = req_trig | t
    return req_trig


_SEL_CACHE = {}


def _coll_fingerprint(coll, fields=("pt", "eta", "phi")):
    """Identity of an object collection for the selection caches: counts and the
    kinematics. The JES/JER and lepton-scale shifts change only pt (and mass), so
    equal fingerprints mean the same objects with the same ID inputs."""
    parts = [hash(ak.to_numpy(ak.num(coll)).tobytes())]
    for f in fields:
        parts.append(hash(ak.to_numpy(ak.flatten(coll[f], axis=None)).tobytes()))
    return tuple(parts)


def _cached(key, compute):
    """Small cache (last 16 results) for selection masks repeated in every shift."""
    if key not in _SEL_CACHE:
        if len(_SEL_CACHE) >= 16:
            _SEL_CACHE.pop(next(iter(_SEL_CACHE)))
        _SEL_CACHE[key] = compute()
    return _SEL_CACHE[key]


def _chunk(events):
    m = getattr(events, "metadata", None) or {}
    return (m.get("filename"), m.get("entrystart"))


_JETID_CACHE = {}


def _jet_fingerprint(events, campaign):
    # same chunk and the same jets (JES/JER shifts change only pt and mass)
    m = getattr(events, "metadata", None) or {}
    eta = ak.to_numpy(ak.flatten(events.Jet.eta))
    phi = ak.to_numpy(ak.flatten(events.Jet.phi))
    return (
        m.get("filename"),
        m.get("entrystart"),
        campaign,
        hash(ak.to_numpy(ak.num(events.Jet)).tobytes()),
        hash(eta.tobytes()),
        hash(phi.tobytes()),
    )


def jet_id(events, campaign, max_eta=2.5, min_pt=20):
    key = _jet_fingerprint(events, campaign)
    if key not in _JETID_CACHE:
        _JETID_CACHE.clear()
        _JETID_CACHE[key] = _jet_id_flags(events, campaign)
    jetid = _JETID_CACHE[key]
    return (events.Jet.pt > min_pt) & (abs(events.Jet.eta) <= max_eta) & (jetid)


def _jet_id_flags(events, campaign):
    # Run 3 NanoAODs have a bug in jetId
    # Implement fix from:
    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetID13p6TeV#nanoAOD_Flags
    # Note: this is only the jetId==6, ie. passJetIdTightLepVeto. Looser selection is not implemented.
    has_jetId = hasattr(events.Jet, "jetId")
    if campaign in ["Summer22", "Summer22EE", "Summer23", "Summer23BPix"] and has_jetId:
        # NanoV12 (has jetId branch)
        jetid = ak.where(
            abs(events.Jet.eta) <= 2.7,
            (events.Jet.jetId >= 2)
            & (events.Jet.muEF < 0.8)
            & (events.Jet.chEmEF < 0.8),
            ak.where(
                (abs(events.Jet.eta) > 2.7) & (abs(events.Jet.eta) <= 3.0),
                (events.Jet.jetId >= 2) & (events.Jet.neHEF < 0.99),
                ak.where(
                    (abs(events.Jet.eta) > 3.0),
                    (events.Jet.jetId >= 2) & (events.Jet.neEmEF < 0.4),
                    ak.zeros_like(events.Jet.pt, dtype=bool),
                ),
            ),
        )
    elif campaign in [
        "Summer22",
        "Summer22EE",
        "Summer23",
        "Summer23BPix",
        "Winter24",
        "Summer24",
        "CustomSummer24",
        "CustomSummer24UParTPaper",
        "CustomSummer25",
        "Winter25",
        "Prompt25",
    ]:
        # NanoV13+ / NanoV15 reprocessing (no jetId branch, compute from components)
        barrel = (
            (events.Jet.neHEF < 0.99)
            & (events.Jet.neEmEF < 0.9)
            & (events.Jet.chMultiplicity + events.Jet.neMultiplicity > 1)
            & (events.Jet.chHEF > 0.01)
            & (events.Jet.chMultiplicity > 0)
        )
        t1 = (events.Jet.neHEF < 0.9) & (events.Jet.neEmEF < 0.99)
        t2 = events.Jet.neHEF < 0.99
        endcap = (events.Jet.neMultiplicity >= 2) & (events.Jet.neEmEF < 0.4)

        jetid = ak.where(
            abs(events.Jet.eta) <= 2.6,
            barrel,
            ak.where(
                (abs(events.Jet.eta) > 2.6) & (abs(events.Jet.eta) <= 2.7),
                t1,
                ak.where(
                    (abs(events.Jet.eta) > 2.7) & (abs(events.Jet.eta) <= 3.0),
                    t2,
                    ak.where(
                        (abs(events.Jet.eta) > 3.0),
                        endcap,
                        ak.zeros_like(events.Jet.pt, dtype=bool),
                    ),
                ),
            ),
        )
        jetid = ak.where(
            np.abs(events.Jet.eta) <= 2.7,
            jetid & (events.Jet.muEF < 0.8) & (events.Jet.chEmEF < 0.8),
            jetid,
        )
    elif campaign in ["2016preVFP-UL", "2016postVFP-UL"]:
        # Run 2 NanoAODv15 jet ID for 2016 (TightLepVeto)
        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetID13TeV
        barrel_2016 = (
            (events.Jet.neHEF < 0.9)
            & (events.Jet.neEmEF < 0.9)
            & (events.Jet.chMultiplicity + events.Jet.neMultiplicity > 1)
            & (events.Jet.chHEF > 0.0)
            & (events.Jet.chMultiplicity > 0)
        )
        t1_2016 = (events.Jet.neHEF < 0.98) & (events.Jet.neEmEF < 0.99)
        t2_2016 = events.Jet.neMultiplicity >= 1
        endcap_2016 = (events.Jet.neMultiplicity > 2) & (events.Jet.neEmEF < 0.9)

        jetid = ak.where(
            abs(events.Jet.eta) <= 2.4,
            barrel_2016,
            ak.where(
                (abs(events.Jet.eta) > 2.4) & (abs(events.Jet.eta) <= 2.7),
                t1_2016,
                ak.where(
                    (abs(events.Jet.eta) > 2.7) & (abs(events.Jet.eta) <= 3.0),
                    t2_2016,
                    ak.where(
                        (abs(events.Jet.eta) > 3.0),
                        endcap_2016,
                        ak.zeros_like(events.Jet.pt, dtype=bool),
                    ),
                ),
            ),
        )
        # TightLepVeto: only in barrel (|eta| <= 2.4) for 2016
        jetid = ak.where(
            np.abs(events.Jet.eta) <= 2.4,
            jetid & (events.Jet.muEF < 0.8) & (events.Jet.chEmEF < 0.8),
            jetid,
        )
    elif campaign in ["2017-UL", "2018-UL"]:
        # Run 2 NanoAODv15 jet ID for 2017 & 2018 (TightLepVeto)
        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetID13TeV
        barrel_1718 = (
            (events.Jet.neHEF < 0.9)
            & (events.Jet.neEmEF < 0.9)
            & (events.Jet.chMultiplicity + events.Jet.neMultiplicity > 1)
            & (events.Jet.chHEF > 0.0)
            & (events.Jet.chMultiplicity > 0)
        )
        t1_1718 = (events.Jet.neHEF < 0.90) & (events.Jet.neEmEF < 0.99)
        t2_1718 = events.Jet.neHEF < 0.9999
        endcap_1718 = (events.Jet.neMultiplicity > 2) & (events.Jet.neEmEF < 0.9)

        jetid = ak.where(
            abs(events.Jet.eta) <= 2.6,
            barrel_1718,
            ak.where(
                (abs(events.Jet.eta) > 2.6) & (abs(events.Jet.eta) <= 2.7),
                t1_1718,
                ak.where(
                    (abs(events.Jet.eta) > 2.7) & (abs(events.Jet.eta) <= 3.0),
                    t2_1718,
                    ak.where(
                        (abs(events.Jet.eta) > 3.0),
                        endcap_1718,
                        ak.zeros_like(events.Jet.pt, dtype=bool),
                    ),
                ),
            ),
        )
        # TightLepVeto: |eta| <= 2.7 for 2017 & 2018
        jetid = ak.where(
            np.abs(events.Jet.eta) <= 2.7,
            jetid & (events.Jet.muEF < 0.8) & (events.Jet.chEmEF < 0.8),
            jetid,
        )
    else:
        jetid = events.Jet.jetId >= 5

    return ak.values_astype(jetid, bool)


def _ele_etaSC(events, campaign):
    return (
        events.Electron.eta + events.Electron.deltaEtaSC
        if campaign
        not in [
            "Summer24",
            "CustomSummer24",
            "CustomSummer24UParTPaper",
            "CustomSummer25",
            "Winter25",
            "Prompt25",
        ]
        else events.Electron.superclusterEta
    )


def ele_cuttightid(events, campaign):
    ele_etaSC = _ele_etaSC(events, campaign)
    elemask = (
        (abs(ele_etaSC) < 1.4442) | ((abs(ele_etaSC) > 1.566) & (abs(ele_etaSC) < 2.5))
    ) & (events.Electron.cutBased > 3)
    return elemask


def ele_mvatightid(events, campaign):
    ele_etaSC = _ele_etaSC(events, campaign)
    elemask = (
        (abs(ele_etaSC) < 1.4442) | ((abs(ele_etaSC) > 1.566) & (abs(ele_etaSC) < 2.5))
    ) & (events.Electron.mvaIso_WP80 > 0.5)
    return elemask


def ele_promptmvaid(events, campaign):
    try:
        key = (
            "ele_promptmva",
            campaign,
            _chunk(events),
            _coll_fingerprint(events.Electron),
        )
    except Exception:
        return _ele_promptmvaid(events, campaign)
    return _cached(key, lambda: _ele_promptmvaid(events, campaign))


def _ele_promptmvaid(events, campaign):
    # https://indico.cern.ch/event/1575017/contributions/6635248/attachments/3115862/5524310/EGammaAug08.pdf
    if campaign not in [
        "Summer24",
        "CustomSummer24",
        "CustomSummer24UParTPaper",
        "CustomSummer25",
        "Winter25",
        "Prompt25",
    ]:
        # No promptMVA WP is defined/validated for this campaign yet. All
        # current call sites already gate on the campaign list above before
        # calling this function, so this branch should be unreachable; it's
        # an explicit guard rather than a real cut.
        raise NotImplementedError(
            f"ele_promptmvaid has no promptMVA working point defined for campaign '{campaign}'"
        )
    ele_etaSC = _ele_etaSC(events, campaign)
    elemask = (
        (abs(ele_etaSC) < 1.4442)
        | ((abs(ele_etaSC) > 1.566) & (abs(ele_etaSC) < 2.4))
        # ) & (events.Electron.promptMVA >= 0.8) #Medium
    ) & (
        events.Electron.promptMVA >= 0.9
    )  # Tight

    # we also need to add the following cuts as preselection:
    # miniPFRelIso_all < 0.4
    # sip3d < 8
    # |dxy| < 0.05
    # |dz| < 0.1
    # pT > 10 GeV
    # Missing inner hits < 2
    # |eta| < 2.4 (analysis choice, user 2026-10-23; EGM preselection is 2.5)

    elemask = elemask & (events.Electron.miniPFRelIso_all < 0.4)
    elemask = elemask & (events.Electron.sip3d < 8)
    elemask = elemask & (abs(events.Electron.dxy) < 0.05)
    elemask = elemask & (abs(events.Electron.dz) < 0.1)
    elemask = elemask & (events.Electron.pt > 10)
    # elemask =  elemask & (events.Electron.MissingInnerHits < 2)
    elemask = elemask & (abs(events.Electron.eta) < 2.4)

    return elemask


def softmu_mask(events, dxySigCut=0):
    softmumask = (
        (events.Muon.pt < 25)
        & (abs(events.Muon.eta) < 2.4)
        & (events.Muon.tightId > 0.5)
        & (events.Muon.pfRelIso04_all > 0.2)
        & (abs(events.Muon.dxy / events.Muon.dxyErr) > dxySigCut)
        & (events.Muon.jetIdx != -1)
    )
    return softmumask


def mu_idiso(events):
    mumask = (
        (abs(events.Muon.eta) < 2.4)
        & (events.Muon.tightId > 0.5)
        & (events.Muon.pfRelIso04_all <= 0.15)
    )
    return mumask


def mu_promptmvaid(events):
    try:
        key = ("mu_promptmva", _chunk(events), _coll_fingerprint(events.Muon))
    except Exception:
        return _mu_promptmvaid(events)
    return _cached(key, lambda: _mu_promptmvaid(events))


def _mu_promptmvaid(events):
    # https://muon-wiki.docs.cern.ch/guidelines/recommendations/#prompt-mva-formerly-tth-mva
    # https://muon-wiki.docs.cern.ch/guidelines/recommendations/#muon-isolation
    # https://cms-talk.web.cern.ch/t/prompt-mva-sfs-definition/132578
    # https://indico.cern.ch/event/1351304/contributions/5688794/attachments/2765665/4817340/CarlosVico_Muon_mvaTTH_24nov2023.pdf (slide 5 for WP)
    mumask = (
        (abs(events.Muon.eta) < 2.4)
        & (events.Muon.tightId > 0.5)
        & (events.Muon.promptMVA > 0.64)
    )

    # additionally the following cuts as preselection:
    # miniPFRelIso_all < 0.4
    # sip3d < 8
    # |dxy| < 0.05 cm
    # |dz| < 0.1 cm
    # pT > 10 GeV
    # abs eta < 2.4
    # isGlobal || isTracker
    # isPFCand == 1
    # mediumID == 1

    mumask = mumask & (events.Muon.miniPFRelIso_all < 0.4)
    mumask = mumask & (events.Muon.sip3d < 8)
    mumask = mumask & (abs(events.Muon.dxy) < 0.05)
    mumask = mumask & (abs(events.Muon.dz) < 0.1)
    mumask = mumask & (events.Muon.pt > 10)
    mumask = mumask & (events.Muon.isGlobal | events.Muon.isTracker)
    # Muon_isPFCand is not written by NanoAODv15, and this function is only
    # ever reached for v15 campaigns (Summer24, CustomSummer24, Winter25,
    # Prompt25 -- older campaigns route to mu_idiso instead), so the branch is
    # absent every time. Verified by direct inspection of the Summer24,
    # CustomSummer24 and Prompt25 tuples: 54 Muon_* branches, no isPFCand.
    # Guarded rather than deleted so the cut applies again if a production
    # restores the branch. Nothing is lost while it is absent: the POG Tight
    # and Medium muon IDs both require isPFMuon, and tightId (above) and
    # mediumId (below) are both already applied here.
    if hasattr(events.Muon, "isPFCand"):
        mumask = mumask & (events.Muon.isPFCand == 1)
    mumask = mumask & (events.Muon.mediumId == 1)

    return mumask


def btag_mu_idiso(events):
    mumask = (
        (abs(events.Muon.eta) < 2.4)
        & (events.Muon.tightId > 0.5)
        & (events.Muon.pfRelIso04_all < 0.12)
    )
    return mumask


def jet_cut(events, campaign, ptmin=180, ptmax=1e5, absetamin=0, absetamax=2.5):
    multijetmask = (
        (abs(events.Jet.eta) > absetamin)
        & (abs(events.Jet.eta) < absetamax)
        & (events.Jet.pt > ptmin)
        & (events.Jet.pt < ptmax)
        & (jet_id(events, campaign))
    )
    return multijetmask


def MET_filters(events, campaign):
    # apply MET filter
    isRealData = not hasattr(events, "genWeight")

    def flags():
        metfilter = ak.ones_like(events.run, dtype=bool)
        for flag in met_filters[campaign]["data" if isRealData else "mc"]:
            metfilter = events.Flag[flag] & metfilter
        return metfilter

    # event-level flags: the same in every JES/JER shift of a chunk
    events_id = (
        hash(ak.to_numpy(events.luminosityBlock).tobytes()),
        hash(ak.to_numpy(events.event).tobytes()),
    )
    metfilter = _cached(
        ("met_flags", campaign, isRealData, _chunk(events), events_id), flags
    )
    if not ak.any((events.run >= 362433) & (events.run <= 367144)):
        return metfilter  # ecalBadCalibFilter below is False outside these runs
    ## Flag_ecalBadCalibFilter
    badjet = (
        (events.Jet.pt > 50)
        & (events.Jet.eta >= -0.5)
        & (events.Jet.eta <= -0.1)
        & (events.Jet.phi >= -2.1)
        & (events.Jet.phi <= -1.8)
        & ((events.Jet.neEmEF > 0.9) | (events.Jet.chEmEF > 0.9))
        & (events.Jet.delta_phi(events.PuppiMET) > 2.9)
    )
    ecalBadCalibFilter = (
        (ak.sum(badjet, axis=-1) >= 1)
        & (events.PuppiMET.pt > 100)
        & (events.run >= 362433)
        & (events.run <= 367144)
    )
    metfilter = metfilter & ~ecalBadCalibFilter
    return metfilter


def btag_wp(jets, year, campaign, tagger, borc, wp):
    WP = wp_dict(year, campaign)
    if borc == "b":
        jet_mask = jets[f"btag{tagger}B"] > WP[tagger]["b"][wp]
    elif borc == "c":
        jet_mask = (jets[f"btag{tagger}CvB"] > WP[tagger]["c"][wp][1]) & (
            jets[f"btag{tagger}CvL"] > WP[tagger]["c"][wp][0]
        )
    elif borc == "2Db":
        jet_mask = jets[f"btag{tagger}2Dbin"] >= WP[tagger]["2D"]["b"][wp]
    elif borc == "2Dc":
        jet_mask = (jets[f"btag{tagger}2Dbin"] >= WP[tagger]["2D"]["c"][wp][0]) & (
            jets[f"btag{tagger}2Dbin"] <= WP[tagger]["2D"]["c"][wp][1]
        )
    else:
        raise ValueError("Invalid flavour!")
    return jet_mask


def fineflav(jets):
    """Fine-grained MC jet flavour (int8 per jet), Run 2 SFbc convention:

    0 = b  (hadronFlavour==5, nBHadrons<=1)
    1 = bb (hadronFlavour==5, nBHadrons>1)
    2 = c  (hadronFlavour==4, nCHadrons<=1)
    3 = cc (hadronFlavour==4, nCHadrons>1)
    4 = uds (light, abs(partonFlavour) in {1,2,3})
    5 = g   (light, partonFlavour==21)
    6 = undefined (everything else: no gen match / PU (genJetIdx<0),
        partonFlavour==0, light jets with stray b/c hadrons)

    The genJetIdx<0 test takes precedence over all flavour tests (applied
    as the final override below, so it always wins).
    """
    hflav = jets.hadronFlavour
    nb = jets.nBHadrons
    nc = jets.nCHadrons
    pflav = jets.partonFlavour
    light = (hflav == 0) & (nb == 0) & (nc == 0)
    flav = ak.full_like(hflav, 6, dtype=np.int8)
    flav = ak.where((hflav == 5) & (nb <= 1), 0, flav)
    flav = ak.where((hflav == 5) & (nb > 1), 1, flav)
    flav = ak.where((hflav == 4) & (nc <= 1), 2, flav)
    flav = ak.where((hflav == 4) & (nc > 1), 3, flav)
    flav = ak.where(light & (abs(pflav) >= 1) & (abs(pflav) <= 3), 4, flav)
    flav = ak.where(light & (pflav == 21), 5, flav)
    # no gen match (pileup) -> undefined, regardless of stored flavours
    flav = ak.where(jets.genJetIdx < 0, 6, flav)
    return ak.values_astype(flav, np.int8)


def veto_lepton_collections(events, campaign):
    """Loose lepton collections used to clean jets in the 2D SF selections
    (Run 2 parity: mirror the DY/Wc dilepton-veto lepton definitions).

    Returns (muons, electrons): muons pt>12, electrons pt>15, with the
    prompt-MVA IDs for the NanoAODv15 campaigns and the older IDs otherwise.
    """
    if campaign in [
        "Summer24",
        "CustomSummer24",
        "CustomSummer24UParTPaper",
        "CustomSummer25",
        "Winter25",
        "Prompt25",
    ]:  # NanoAODv15
        muons = events.Muon[(events.Muon.pt > 12) & mu_promptmvaid(events)]
        electrons = events.Electron[
            (events.Electron.pt > 15) & ele_promptmvaid(events, campaign)
        ]
    else:
        muons = events.Muon[(events.Muon.pt > 12) & mu_idiso(events)]
        electrons = events.Electron[
            (events.Electron.pt > 15) & ele_mvatightid(events, campaign)
        ]
    return muons, electrons


def jet_veto_lepton_clean(events, campaign, cone=0.4):
    """Per-jet mask: True when the jet is farther than `cone` from every
    lepton of `veto_lepton_collections`. A None from ak.all means "no veto
    lepton in this event" and is filled with True (the jet is clean).
    Shared by the four 2D SF workflows so the cone and the None semantics
    can only change in one place. Call it BEFORE reassigning events.Muon /
    events.Electron to selected subsets.
    """
    try:
        key = (
            "jet_lepton_clean",
            campaign,
            cone,
            _chunk(events),
            _coll_fingerprint(events.Jet, ("eta", "phi")),
            _coll_fingerprint(events.Muon),
            _coll_fingerprint(events.Electron),
        )
    except Exception:
        return _jet_veto_lepton_clean(events, campaign, cone)
    return _cached(key, lambda: _jet_veto_lepton_clean(events, campaign, cone))


def _jet_veto_lepton_clean(events, campaign, cone=0.4):
    muons, electrons = veto_lepton_collections(events, campaign)
    clean = None
    for coll in (muons, electrons):
        mask = ak.fill_none(
            ak.all(
                events.Jet.metric_table(coll) > cone,
                axis=2,
                mask_identity=True,
            ),
            True,
            axis=-1,
        )
        clean = mask if clean is None else clean & mask
    return clean


_mapping = {
    1: {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
    },
    2: {
        1: 40,
        2: 40,
        3: 40,
        4: 40,
        5: 40,
        6: 40,
        7: 40,
        8: 40,
    },
    3: {
        1: 41,
        2: 41,
        3: 41,
        4: 41,
        5: 41,
        6: 41,
        7: 41,
        8: 41,
    },
    4: {
        1: 44,
        2: 43,
        3: 42,
        4: 50,
        5: 51,
        6: 52,
        7: 53,
        8: 54,
    },
}
_b2D = {
    "L": 50,
    "M": 51,
    "T": 52,
}
_c2D = {
    "L": [40, 44],
    "M": [41, 44],
    "T": [42, 44],
}

# Phase4 category scheme (SFbc-2D/OptimizedWP/results/phase4_scheme_NoPIDParT.json,
# tagger NoPIDParT). Kept alongside _mapping/_b2D/_c2D (untouched) rather than
# replacing them -- legacy behaviour must stay bit-identical. `mapping` there
# is taken verbatim (x = HFvLF bin 1..4, y = BvC bin 1..8 -> category code
# 0..10). Unlike the legacy codes (40-44/50-54), phase4's codes interleave
# columns (e.g. column x=4 uses 1,3,5,6,7,8,9,10 while x=2/x=3 use 2/4), so
# they don't admit the same simple ">=threshold"/contiguous-range WP cuts in
# general.
_mapping_phase4 = {
    1: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0},
    2: {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2},
    3: {1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 7: 4, 8: 4},
    4: {1: 1, 2: 3, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9, 8: 10},
}
_jet_pt_bins = [
    (25, 35),
    (35, 50),
    (50, 70),
    (70, 90),
    (90, 120),
    (120, 10000),
]
btag_wp_dict = {
    "2016preVFP_2016preVFP-UL": {
        "UParTAK4": {
            "b": {
                "No": 0.0,
                "L": 0.0387,
                "M": 0.1847,
                "T": 0.5467,
                "XT": 0.6777,
                "XXT": 0.9219,
            },
            "c": {  # placeholder
                "No": [0.0, 0.0],
                "L": [0.1, 0.1],  # CvL, then CvB
                "M": [0.5, 0.5],
                "T": [0.8, 0.8],
                "XT": [0.9, 0.9],
            },
        },
    },
    "2016postVFP_2016postVFP-UL": {
        "UParTAK4": {
            "b": {
                "No": 0.0,
                "L": 0.0400,
                "M": 0.1898,
                "T": 0.5538,
                "XT": 0.6872,
                "XXT": 0.9353,
            },
            "c": {  # placeholder
                "No": [0.0, 0.0],
                "L": [0.1, 0.1],  # CvL, then CvB
                "M": [0.5, 0.5],
                "T": [0.8, 0.8],
                "XT": [0.9, 0.9],
            },
        },
    },
    "2017_2017-UL": {
        "UParTAK4": {
            "b": {
                "No": 0.0,
                "L": 0.0331,
                "M": 0.1776,
                "T": 0.5755,
                "XT": 0.7274,
                "XXT": 0.9666,
            },
            "c": {  # placeholder
                "No": [0.0, 0.0],
                "L": [0.1, 0.1],  # CvL, then CvB
                "M": [0.5, 0.5],
                "T": [0.8, 0.8],
                "XT": [0.9, 0.9],
            },
        },
    },
    "2018_2018-UL": {  # correct, the format is year_campaign
        "UParTAK4": {
            "b": {
                "No": 0.0,
                "L": 0.0308,
                "M": 0.1610,
                "T": 0.5405,
                "XT": 0.6992,
                "XXT": 0.9655,
            },
            "c": {  # placeholder
                "No": [0.0, 0.0],
                "L": [0.1, 0.1],  # CvL, then CvB
                "M": [0.5, 0.5],
                "T": [0.8, 0.8],
                "XT": [0.9, 0.9],
            },
        },
    },
    "2022_Summer22": {
        "DeepFlav": {
            "b": {
                "No": 0.0,
                "L": 0.0583,
                "M": 0.3086,
                "T": 0.7183,
                "XT": 0.8111,
                "XXT": 0.9512,
            },
            "c": {
                "No": [0.0, 0.0],
                "L": [0.042, 0.208],  # CvL, then CvB
                "M": [0.108, 0.299],
                "T": [0.303, 0.243],
            },
        },
        "RobustParTAK4": {
            "b": {
                "No": 0.0,
                "L": 0.0849,
                "M": 0.4319,
                "T": 0.8482,
                "XT": 0.9151,
                "XXT": 0.9874,
            },
            "c": {
                "No": [0.0, 0.0],
                "L": [0.039, 0.068],
                "M": [0.117, 0.130],
                "T": [0.360, 0.095],
            },
        },
        "PNet": {
            "b": {
                "No": 0.0,
                "L": 0.047,
                "M": 0.245,
                "T": 0.6734,
                "XT": 0.7862,
                "XXT": 0.961,
            },
            "c": {
                "No": [0.0, 0.0],
                "L": [0.054, 0.181],
                "M": [0.160, 0.306],
                "T": [0.492, 0.259],
            },
            "2D": {
                "HFvLF": np.array([0.0, 0.160, 0.332, 0.706, 1.00001]),
                "BvC": np.array(
                    [0.0, 0.036, 0.090, 0.260, 0.799, 0.932, 0.978, 0.993, 1.00001]
                ),
                "mapping": _mapping,
                "b": _b2D,
                "c": _c2D,
                "jet_pt_bins": _jet_pt_bins,
            },
        },
    },
    "2022_Summer22EE": {
        "DeepFlav": {
            "b": {
                "No": 0.0,
                "L": 0.0614,
                "M": 0.3196,
                "T": 0.73,
                "XT": 0.8184,
                "XXT": 0.9542,
            },
            "c": {
                "No": [0.0, 0.0],
                "L": [0.042, 0.206],  # CvL, then CvB
                "M": [0.108, 0.298],
                "T": [0.305, 0.241],
            },
        },
        "RobustParTAK4": {
            "b": {
                "No": 0.0,
                "L": 0.0897,
                "M": 0.451,
                "T": 0.8604,
                "XT": 0.9234,
                "XXT": 0.9893,
            },
            "c": {
                "No": [0.0, 0.0],
                "L": [0.039, 0.067],
                "M": [0.117, 0.128],
                "T": [0.358, 0.095],
            },
        },
        "PNet": {
            "b": {
                "No": 0.0,
                "L": 0.0499,
                "M": 0.2605,
                "T": 0.6915,
                "XT": 0.8033,
                "XXT": 0.9664,
            },
            "c": {
                "No": [0.0, 0.0],
                "L": [0.054, 0.182],
                "M": [0.160, 0.304],
                "T": [0.491, 0.258],
            },
            "2D": {
                "HFvLF": np.array([0.0, 0.160, 0.332, 0.706, 1.00001]),
                "BvC": np.array(
                    [0.0, 0.036, 0.090, 0.260, 0.799, 0.932, 0.978, 0.993, 1.00001]
                ),
                "mapping": _mapping,
                "b": _b2D,
                "c": _c2D,
                "jet_pt_bins": _jet_pt_bins,
            },
        },
    },
    "2023_Summer23": {
        "DeepFlav": {
            "b": {
                "No": 0.0,
                "L": 0.0479,
                "M": 0.2431,
                "T": 0.6553,
                "XT": 0.7667,
                "XXT": 0.9459,
            },
            "c": {
                "No": [0.0, 0.0],
                "L": [0.042, 0.234],  # CvL, then CvB
                "M": [0.102, 0.322],
                "T": [0.250, 0.262],
                "XT": [0.371, 0.440],
            },
        },
        "RobustParTAK4": {
            "b": {
                "No": 0.0,
                "L": 0.0681,
                "M": 0.3487,
                "T": 0.7969,
                "XT": 0.8882,
                "XXT": 0.9883,
            },
            "c": {
                "No": [0.0, 0.0],
                "L": [0.038, 0.086],
                "M": [0.109, 0.153],
                "T": [0.308, 0.113],
                "XT": [0.469, 0.275],
            },
        },
        "PNet": {
            "b": {
                "No": 0.0,
                "L": 0.0358,
                "M": 0.1917,
                "T": 0.6172,
                "XT": 0.7515,
                "XXT": 0.9659,
            },
            "c": {
                "No": [0.0, 0.0],
                "L": [0.052, 0.220],
                "M": [0.148, 0.353],
                "T": [0.434, 0.300],
                "XT": [0.634, 0.549],
            },
            "2D": {
                "HFvLF": np.array([0.0, 0.144, 0.292, 0.646, 1.00001]),
                "BvC": np.array(
                    [0.0, 0.036, 0.078, 0.240, 0.752, 0.915, 0.972, 0.992, 1.00001]
                ),
                "mapping": _mapping,
                "b": _b2D,
                "c": _c2D,
                "jet_pt_bins": _jet_pt_bins,
            },
        },
    },
    "2023_Summer23BPix": {
        "DeepFlav": {
            "b": {
                "No": 0.0,
                "L": 0.048,
                "M": 0.2435,
                "T": 0.6563,
                "XT": 0.7671,
                "XXT": 0.9483,
            },
            "c": {
                "No": [0.0, 0.0],
                "L": [0.042, 0.242],  # CvL, then CvB
                "M": [0.102, 0.328],
                "T": [0.250, 0.267],
                "XT": [0.371, 0.444],
            },
        },
        "RobustParTAK4": {
            "b": {
                "No": 0.0,
                "L": 0.0683,
                "M": 0.3494,
                "T": 0.7994,
                "XT": 0.8877,
                "XXT": 0.9883,
            },
            "c": {
                "No": [0.0, 0.0],
                "L": [0.038, 0.091],
                "M": [0.109, 0.157],
                "T": [0.308, 0.116],
                "XT": [0.469, 0.281],
            },
        },
        "PNet": {
            "b": {
                "No": 0.0,
                "L": 0.0359,
                "M": 0.1919,
                "T": 0.6133,
                "XT": 0.7544,
                "XXT": 0.9688,
            },
            "c": {
                "No": [0.0, 0.0],
                "L": [0.052, 0.228],
                "M": [0.149, 0.358],
                "T": [0.436, 0.303],
                "XT": [0.634, 0.5552],
            },
            "2D": {
                "HFvLF": np.array([0.0, 0.144, 0.288, 0.640, 1.00001]),
                "BvC": np.array(
                    [0.0, 0.036, 0.082, 0.243, 0.742, 0.909, 0.969, 0.992, 1.00001]
                ),
                "mapping": _mapping,
                "b": _b2D,
                "c": _c2D,
                "jet_pt_bins": _jet_pt_bins,
            },
        },
    },
    "2024_Summer24": {
        "UParTAK4": {
            "b": {
                "No": 0.0,
                "L": 0.0246,
                "M": 0.1272,
                "T": 0.4648,
                "XT": 0.6298,
                "XXT": 0.9739,
            },
            "c": {
                "No": [0.0, 0.0],
                "L": [0.086, 0.233],  # CvL, then CvB
                "M": [0.291, 0.457],
                "T": [0.650, 0.421],
                "XT": [0.810, 0.736],
            },
            "2D": {
                "HFvLF": np.array([0.000, 0.250, 0.452, 0.808, 1.000]),
                "BvC": np.array(
                    [0.000, 0.006, 0.017, 0.055, 0.761, 0.944, 0.985, 0.995, 1.000]
                ),
                "mapping": _mapping,
                "b": _b2D,
                "c": _c2D,
                "jet_pt_bins": _jet_pt_bins,
            },
        },
        # Trial WPs from SFbc-2D/scripts/make2DWPIntersect.py (is_predefined,
        # --tagger NoPIDParT); no dedicated single-tag b/c WPs exist for this
        # tagger, only the 2D scheme.
        "NoPIDParTAK4": {
            "2D": {
                "HFvLF": np.array([0.000, 0.134, 0.297, 0.729, 1.000]),
                "BvC": np.array(
                    [0.000, 0.035, 0.095, 0.280, 0.956, 0.992, 0.998, 0.999, 1.000]
                ),
                "mapping": _mapping,
                "b": _b2D,
                "c": _c2D,
                "jet_pt_bins": _jet_pt_bins,
            },
        },
        # Phase4 category scheme (SFbc-2D/OptimizedWP study, results/
        # phase4_scheme_NoPIDParT.json), added alongside the entry above --
        # same underlying HFvLF/BvC scores, different category boundaries and
        # mapping. HFvLF cuts are unchanged from legacy; 3 of 7 BvC cuts move.
        "NoPIDParTAK4_phase4": {
            "2D": {
                "HFvLF": np.array([0.000, 0.134, 0.297, 0.729, 1.000]),
                "BvC": np.array(
                    [
                        0.000,
                        0.035,
                        0.095,
                        0.22515727366719923,
                        0.29954950677023995,
                        0.9006733695665995,
                        0.992,
                        0.999,
                        1.000,
                    ]
                ),
                "mapping": _mapping_phase4,
                # No "b"/"c" single-WP entries: the phase4 study fits all 11
                # categories directly (SFbc-2D writeFlavCards.py --sf-structure)
                # and never derived L/M/T cuts on this scheme. Inventing them
                # here would hand callers of btag_wp(..., "2Db"/"2Dc", ...) an
                # uncalibrated working point that looks official.
                "jet_pt_bins": _jet_pt_bins,
            },
        },
    },
    "2025_Prompt25": {
        "UParTAK4": {
            "b": {
                "No": 0.0,
                "L": 0.0246,
                "M": 0.1272,
                "T": 0.4648,
                "XT": 0.6298,
                "XXT": 0.9739,
            },
            "c": {
                "No": [0.0, 0.0],
                "L": [0.086, 0.233],  # CvL, then CvB
                "M": [0.291, 0.457],
                "T": [0.650, 0.421],
                "XT": [0.810, 0.736],
            },
            "2D": {
                "HFvLF": np.array([0.000, 0.250, 0.452, 0.808, 1.000]),
                "BvC": np.array(
                    [0.000, 0.006, 0.017, 0.055, 0.761, 0.944, 0.985, 0.995, 1.000]
                ),
                "mapping": _mapping,
                "b": _b2D,
                "c": _c2D,
                "jet_pt_bins": _jet_pt_bins,
            },
        },
    },
}
# CustomSummer24 is a private NanoAODv15 production on the same MiniAOD
# conditions as Summer24 -- reuse its precomputed WP thresholds.
btag_wp_dict["2024_CustomSummer24"] = btag_wp_dict["2024_Summer24"]
btag_wp_dict["2024_CustomSummer24UParTPaper"] = btag_wp_dict["2024_Summer24"]
# CustomSummer25 uses the CustomSummer24 MC, so the WP boundaries stay the same.
btag_wp_dict["2025_CustomSummer25"] = btag_wp_dict["2024_CustomSummer24"]


import os, correctionlib


def wp_dict(year, campaign):
    """
    year :
    """
    # btag_wp_dicts={}
    cache_key = f"{year}_{campaign}"

    if cache_key in btag_wp_dict:
        return btag_wp_dict[cache_key]

    name_map = {
        "deepJet": "DeepFlav",
        "robustParticleTransformer": "RobustParTAK4",
        "particleNet": "PNet",
        "unifiedParticleTransformer": "UParTAK4",
    }

    wps_dict = {}
    if os.path.exists(
        f"/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/{campaign_map()[campaign]}/latest/"
    ):
        btag = correctionlib.CorrectionSet.from_file(
            f"/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/{campaign_map()[campaign]}/latest/btagging.json.gz"
        )
        ctag = correctionlib.CorrectionSet.from_file(
            f"/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/{campaign_map()[campaign]}/latest/ctagging.json.gz"
        )
        tagger_list = [i for i in list(btag.keys()) if "wp_values" in i]

        if len(tagger_list) == 0:
            btag_wp_dict[cache_key] = wps_dict
            return wps_dict

        for tagger in tagger_list:
            wps_dict[name_map[tagger.replace("_wp_values", "")]] = {"b": {}, "c": {}}
            # Get b WPs
            bwp = btag[tagger].inputs[0].description.split("/")
            wps_dict[name_map[tagger.replace("_wp_values", "")]]["b"] = {
                wp: btag[tagger].evaluate(wp) for wp in bwp
            }
            # Get c WPs in [CvL, CvB]
            cwp = ctag[tagger].inputs[0].description.split("/")
            wps_dict[name_map[tagger.replace("_wp_values", "")]]["c"] = {
                wp: [ctag[tagger].evaluate(wp, "CvL"), ctag[tagger].evaluate(wp, "CvB")]
                for wp in cwp
            }
        btag_wp_dict[cache_key] = wps_dict
        return wps_dict

    else:
        btag_wp_dict[cache_key] = wps_dict
        return wps_dict


met_filters = {
    "2016preVFP-UL": {
        "data": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "HBHENoiseFilter",
            "HBHENoiseIsoFilter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "eeBadScFilter",
        ],
        "mc": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "HBHENoiseFilter",
            "HBHENoiseIsoFilter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "eeBadScFilter",
        ],
    },
    "2016postVFP-UL": {
        "data": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "HBHENoiseFilter",
            "HBHENoiseIsoFilter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "eeBadScFilter",
        ],
        "mc": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "HBHENoiseFilter",
            "HBHENoiseIsoFilter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "eeBadScFilter",
        ],
    },
    "2017-UL": {
        "data": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "HBHENoiseFilter",
            "HBHENoiseIsoFilter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "hfNoisyHitsFilter",
            "eeBadScFilter",
            "ecalBadCalibFilter",
        ],
        "mc": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "HBHENoiseFilter",
            "HBHENoiseIsoFilter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "hfNoisyHitsFilter",
            "eeBadScFilter",
            "ecalBadCalibFilter",
        ],
    },
    "2018-UL": {
        "data": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "HBHENoiseFilter",
            "HBHENoiseIsoFilter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "hfNoisyHitsFilter",
            "eeBadScFilter",
            "ecalBadCalibFilter",
        ],
        "mc": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "HBHENoiseFilter",
            "HBHENoiseIsoFilter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "hfNoisyHitsFilter",
            "eeBadScFilter",
            "ecalBadCalibFilter",
        ],
    },
    "Summer22": {
        "data": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "hfNoisyHitsFilter",
            "eeBadScFilter",
        ],
        "mc": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "hfNoisyHitsFilter",
            "eeBadScFilter",
        ],
    },
    "Summer22EE": {
        "data": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "hfNoisyHitsFilter",
            "eeBadScFilter",
        ],
        "mc": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "hfNoisyHitsFilter",
            "eeBadScFilter",
        ],
    },
    "Summer23": {
        "data": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "hfNoisyHitsFilter",
            "eeBadScFilter",
        ],
        "mc": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "hfNoisyHitsFilter",
            "eeBadScFilter",
        ],
    },
    "Summer23BPix": {
        "data": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "hfNoisyHitsFilter",
            "eeBadScFilter",
        ],
        "mc": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "hfNoisyHitsFilter",
            "eeBadScFilter",
        ],
    },
    "Summer24": {
        "data": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "hfNoisyHitsFilter",
            "eeBadScFilter",
            "ecalBadCalibFilter",
        ],
        "mc": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "hfNoisyHitsFilter",
            "eeBadScFilter",
            "ecalBadCalibFilter",
        ],
    },
    "Prompt25": {
        "data": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "hfNoisyHitsFilter",
            "eeBadScFilter",
            "ecalBadCalibFilter",
        ],
        "mc": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "hfNoisyHitsFilter",
            "eeBadScFilter",
            "ecalBadCalibFilter",
        ],
    },
    "prompt_dataMC": {
        "data": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "hfNoisyHitsFilter",
            "eeBadScFilter",
        ],
        "mc": [
            "goodVertices",
            "globalSuperTightHalo2016Filter",
            "EcalDeadCellTriggerPrimitiveFilter",
            "BadPFMuonFilter",
            "BadPFMuonDzFilter",
            "hfNoisyHitsFilter",
            "eeBadScFilter",
        ],
    },
}
# CustomSummer24 is a private NanoAODv15 production on the same MiniAOD
# conditions as Summer24 -- reuse its MET filter list.
met_filters["CustomSummer24"] = met_filters["Summer24"]
met_filters["CustomSummer24UParTPaper"] = met_filters["Summer24"]
met_filters["CustomSummer25"] = met_filters["Prompt25"]
