import itertools

import awkward as ak
import numpy as np
from BTVNanoCommissioning.helpers.func import flatten
from .hist_helpers import get_hist_collections, get_axes_collections
from .histograms.sf2d import SCHEMES as SF2D_SCHEMES
from .histograms.sf2d import TAG_LADDER_THRESHOLDS, TAG_SEL_CODES


def _logit(x):
    # scores outside (0, 1) (defaults, jets outside a pT bin) go to underflow
    ok = (x > 0) & (x < 1)
    xs = ak.where(ok, x, 0.5)
    return ak.where(ok, np.log(xs / (1.0 - xs)), -999.0)


SYST_TOKEN = "__syst__"


class _FillRecorder:
    def __init__(self, name, hist, calls):
        self.name, self.hist, self.calls = name, hist, calls

    def fill(self, *args, **kwargs):
        self.calls.append((self.name, self.hist, args, kwargs))


def _to_numpy(x):
    if isinstance(x, ak.Array):
        try:
            return ak.to_numpy(x)
        except Exception:
            return x
    return x


def _is_token(x):
    if isinstance(x, str):
        return x == SYST_TOKEN
    return (
        isinstance(x, np.ndarray)
        and x.dtype.kind in "USO"
        and (len(x) == 0 or x[0] == SYST_TOKEN)
    )


def replay_fills(output, fill_all, systematics, weight_for, n_events, skip=None):
    """Fill all histograms for all weight variations at the cost of one pass.

    fill_all(output, weight, syst) runs the fill code once with weight = event
    index and syst = SYST_TOKEN; every fill call is recorded. The calls are then
    repeated per variation with weight = weight_for(syst)[index] and the real
    syst label, in the original order, so the histograms are identical. Valid
    only if fill_all uses the weight just for indexing/masking/broadcasting."""
    calls = []
    fill_all(
        {k: _FillRecorder(k, h, calls) for k, h in output.items()},
        np.arange(n_events, dtype=np.float64),
        SYST_TOKEN,
    )
    rec = []
    for name, h, args, kwargs in calls:
        kw = {k: _to_numpy(v) for k, v in kwargs.items()}
        idx = kw.pop("weight", None)
        if idx is not None:
            idx = np.asarray(idx).astype(np.int64)
        args = [_to_numpy(a) for a in args]
        # a scalar label is ~10x faster to fill than an array of labels; it needs
        # one non-empty array argument to broadcast against (else keep the array)
        scalar = any(
            isinstance(x, np.ndarray) and not _is_token(x) and x.ndim and len(x)
            for x in (*args, *kw.values())
        )
        rec.append((name, h, args, kw, idx, scalar))
    for syst in systematics:
        w = weight_for(syst)
        for name, h, args, kw, idx, scalar in rec:
            if skip is not None and skip(name, syst):
                continue
            sub = lambda x: (
                (
                    np.full(len(x), syst)
                    if isinstance(x, np.ndarray) and not scalar
                    else syst
                )
                if _is_token(x)
                else x
            )
            a = [sub(x) for x in args]
            k = {key: sub(v) for key, v in kw.items()}
            if idx is None:
                h.fill(*a, **k)
            else:
                h.fill(*a, weight=w[idx], **k)


def _fill_sf2d_histograms(output, seljets, genflavor, weight, syst):
    """Fill the SF-fit template histograms (histograms/sf2d.py): tag category
    x jet pT x SV variable, once per (2D scheme, SV variant) combination
    present in `output`. Both `seljets` and `genflavor` are already jagged
    per-jet (one entry per SelJet), so no broadcasting is needed for those --
    only `weight` (one entry per event) needs broadcasting up to per-jet.

    An svmass template is filled only where one was booked, which is the
    channel's own declaration that SelJet_svmass is trustworthy there (see
    histograms/sf2d.py). `svmass` has no cone cut -- it is the mass of the
    closest SV *in the event*, not necessarily this jet's -- so a jet with
    nSVs == 0 can still carry a borrowed non-negative mass. The joint gate is
    applied here: svmass is forced to -1 (bin 0 on the variable axis) wherever
    nSVs == 0, regardless of what SelJet.svmass itself holds.
    """
    flat_pt = flatten(seljets["pt"])
    if len(flat_pt) == 0:
        return
    flat_flav = flatten(genflavor)
    flat_weight = flatten(ak.broadcast_arrays(weight, seljets["pt"])[0])
    flat_syst = np.full(len(flat_pt), syst[0])

    for scheme, (branch, _codes) in SF2D_SCHEMES.items():
        if branch not in seljets.fields:
            continue
        flat_tagcat = flatten(seljets[branch])

        nsv_name = f"sf2d_{scheme}_nSVs"
        if nsv_name in output and "nSVs" in seljets.fields:
            output[nsv_name].fill(
                syst=flat_syst,
                flav=flat_flav,
                tagcat=flat_tagcat,
                pt=flat_pt,
                nSVs=flatten(seljets.nSVs),
                weight=flat_weight,
            )

        svmass_name = f"sf2d_{scheme}_svmass"
        if (
            svmass_name in output
            and "svmass" in seljets.fields
            and "nSVs" in seljets.fields
        ):
            flat_nsv = flatten(seljets.nSVs)
            flat_svmass = np.where(flat_nsv == 0, -1.0, flatten(seljets.svmass))
            output[svmass_name].fill(
                syst=flat_syst,
                flav=flat_flav,
                tagcat=flat_tagcat,
                pt=flat_pt,
                svmass=flat_svmass,
                weight=flat_weight,
            )


STAG_PREFIXES = ("stagtpl_", "stagscore_", "stagdisc_")


def _fill_stag_histograms(output, jets, weight, syst, osss=None):
    """Fill the s-tagging histograms (histograms/stag.py) present in `output` for a
    jet collection that carries the stag fields (helpers/stag_helpers.add_stag_fields),
    jagged (several jets per event) or one jet per event. `weight`, `syst` and `osss`
    are per event (or scalar); data jets (no stagFlav) are filled with flav 0.
    svmass is set to -1 where nSVs == 0, as for the 2D templates."""
    names = [k for k in output if k.startswith(STAG_PREFIXES)]
    if not names or "stagBin" not in jets.fields:
        return
    jagged = jets.ndim > 1

    def flat(x):
        # per-jet field -> flat numpy array
        return np.asarray(ak.to_numpy(ak.flatten(x) if jagged else x))

    pt = flat(jets.pt)
    n = len(pt)
    if n == 0:
        return
    if jagged:
        w = np.asarray(ak.to_numpy(ak.flatten(ak.broadcast_arrays(weight, jets.pt)[0])))
    else:
        w = np.asarray(weight)
    label = syst if isinstance(syst, str) else syst[0]
    common = {
        "syst": np.full(n, label),
        "flav": flat(jets.stagFlav) if "stagFlav" in jets.fields else np.zeros(n, int),
    }
    if osss is not None:
        if jagged:
            common["osss"] = np.asarray(
                ak.to_numpy(ak.flatten(ak.broadcast_arrays(osss, jets.pt)[0]))
            )
        else:
            common["osss"] = np.asarray(ak.to_numpy(osss))
    tagcat = flat(jets.stagBin)
    for name in names:
        h = output[name]
        axis_names = (h.hist if isinstance(h, _FillRecorder) else h).axes.name
        kw = {k: v for k, v in common.items() if k in axis_names}
        if name.startswith("stagtpl_"):
            var = name[len("stagtpl_") :]
            if var not in jets.fields:
                continue
            values = flat(jets[var])
            if var == "svmass" and "nSVs" in jets.fields:
                values = np.where(flat(jets.nSVs) == 0, -1.0, values)
            h.fill(tagcat=tagcat, pt=pt, weight=w, **kw, **{var: values})
            continue
        if name.startswith("stagscore_"):
            field = "stagScore_" + name[len("stagscore_") :]
        else:
            disc = name[len("stagdisc_") :]
            field = (
                f"btagNoPIDParTAK4{disc}" if disc.startswith("prob") else f"stag{disc}"
            )
        if field in jets.fields:
            h.fill(discr=flat(jets[field]), weight=w, **kw)


def histogrammer(
    jet_fields=None,
    obj_list=[],
    hist_collections=["fourvector"],
    axes_collections=["common"],
    custom_axes=None,
    **kwargs,
):
    """
    Get a dictionary of histograms from the given histogram collections with the specified axes.

    Parameters:
    jet_fields (list): List of jet fields to be included in the histograms, e.g. events.Jet.fields
    obj_list (list): List of objects to be included in the histograms, e.g. "jet", "mu", "jet0". Accessed by the collections.
    hist_collections (list): List of histogram collections to be included in the histograms. Defined in the utils/histogramming/histograms directory.
    axes_collections (list): List of axes collections to be included in the histograms. Defined in the utils/histogramming/axes directory.
    custom_axes (dict): Dictionary of custom axes to be included in the histograms. Overwrites the axes defined in the axes_collections.
    **kwargs: Additional keyword arguments to be passed to the histogram collection functions.

    Example:
    ```python
    output = histogrammer(
        jet_fields=events.Jet.fields,
        obj_list=["jet", "mu"],
        hist_collections=["example", "common"],
        axes_collections=["common"],
    )
    ```

    Returns:
    dict: A dictionary containing the defined histograms.
    """

    ## Common axes
    if custom_axes:
        axes = custom_axes
    else:
        axes = get_axes_collections(axes_collections)

    ## Histograms
    _hist_dict = get_hist_collections(
        axes,
        hist_collections,
        obj_list=obj_list,
        jet_fields=jet_fields,
        **kwargs,
    )

    return _hist_dict


def histo_writer(pruned_ev, output, weights, systematics, isSyst, SF_map):
    return histo_writter(pruned_ev, output, weights, systematics, isSyst, SF_map)


# Filled common histogram
def histo_writter(pruned_ev, output, weights, systematics, isSyst, SF_map):
    """
    Write histograms to the output dictionary based on pruned events and other parameters.

    This function processes the pruned events and writes the histograms to the `output` dictionary. It takes into account the weights, systematics, and scale factors.

    Parameters:
    pruned_ev (coffea.nanoaodevents): The pruned events data to be histogrammed.
    output (dict): The output dictionary where histograms will be stored.
    weights (coffea.analysis_tools.Weights): The weights object for the events.
    systematics (list): A list of systematic variations to be considered.
    isSyst (str,bool): Indicating whether systematic variations are to be applied.
    SF_map (dict): A dictionary containing scale factors for different variables.

    Example:
    ```python
    histo_writter(pruned_ev, output, weights, systematics, isSyst, SF_map)
    ```

    Returns:
    None
    """

    # Reduce the jet to the correct dimension in the plot
    found4jets = False
    found3jets = False
    found2jets = False
    for key in output.keys():
        if "jet3" in key:  # Because 0-indexing
            found4jets = True
            break
        if "jet2" in key:  # Because 0-indexing
            found3jets = True
        if "jet1" in key:  # Because 0-indexing
            found2jets = True
    nj = 4 if found4jets else 3 if found3jets else 2 if found2jets else 1
    if nj != 1:
        pruned_ev["SelJet"] = pruned_ev.SelJet[:, :nj]
    if "var" in str(ak.type(pruned_ev.SelJet.pt)) and nj == 1:
        pruned_ev.SelJet = pruned_ev.SelJet[:, 0]

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

    # Fill code for one systematic variation
    def _fill_all(output, weight, syst_perjet):
        nonlocal nj
        syst = np.full(len(weight), syst_perjet)
        jet_i = {}

        def _jet_i(i):
            # jet i of the events with >= i+1 SelJets, with its flavour, weight and syst label (once per pass)
            if i not in jet_i:
                m = ak.count(pruned_ev.SelJet["pt"], axis=1) >= (i + 1)
                jet_i[i] = (
                    pruned_ev.SelJet[m][:, i],
                    genflavor[m][:, i],
                    weight[m],
                    syst[m],
                )
            return jet_i[i]

        # Loop over the histograms
        for histname, h in output.items():
            if "_logit_" in histname and syst_perjet not in ("nominal", SYST_TOKEN):
                continue
            # Tagger score histograms
            if (
                "Deep" in histname
                and "btag" not in histname
                and histname in pruned_ev.SelJet.fields
            ):
                temp_weights = flatten(
                    ak.broadcast_arrays(weight, pruned_ev.SelJet["pt"])[0]
                )
                temp_syst = np.full(len(temp_weights), syst[0])
                h.fill(
                    temp_syst,
                    flatten(genflavor),
                    flatten(pruned_ev.SelJet[histname]),
                    weight=flatten(
                        ak.broadcast_arrays(weight, pruned_ev.SelJet["pt"])[0]
                    ),
                )
            # PFcands histograms
            elif (
                "PFCands" in pruned_ev.fields
                and "PFCands" in histname
                and histname.split("_")[1] in pruned_ev.PFCands.fields
            ):
                if "MuonJet" in pruned_ev.fields:
                    h.fill(
                        syst,
                        flatten(
                            ak.broadcast_arrays(smflav, pruned_ev.PFCands["pt"])[0]
                        ),
                        flatten(pruned_ev.PFCands[histname.replace("PFCands_", "")]),
                        weight=flatten(
                            ak.broadcast_arrays(weight, pruned_ev.PFCands["pt"])[0]
                        ),
                    )
                else:
                    h.fill(
                        syst,
                        flatten(
                            ak.broadcast_arrays(
                                genflavor[:, 0], pruned_ev.PFCands["pt"]
                            )[0]
                        ),
                        flatten(pruned_ev.PFCands[histname.replace("PFCands_", "")]),
                        weight=flatten(
                            ak.broadcast_arrays(weight, pruned_ev.PFCands["pt"])[0]
                        ),
                    )
            # Leading lepton histograms
            elif (
                "hl_" in histname
                and "hl" in pruned_ev.fields
                and histname.replace("hl_", "") in pruned_ev.hl.fields
            ):
                h.fill(
                    syst,
                    flatten(pruned_ev.hl[histname.replace("hl_", "")]),
                    weight=weight,
                )
            # Subleading lepton histograms
            elif (
                "sl_" in histname
                and "sl" in pruned_ev.fields
                and histname.replace("sl_", "") in pruned_ev.sl.fields
            ):
                h.fill(
                    syst,
                    flatten(pruned_ev.sl[histname.replace("sl_", "")]),
                    weight=weight,
                )
            # Selected electron histograms
            elif (
                "ele_" in histname
                and histname.replace("ele_", "") in pruned_ev.SelElectron.fields
                and "Plus" not in pruned_ev.fields
            ):
                h.fill(
                    syst,
                    flatten(pruned_ev.SelElectron[histname.replace("ele_", "")]),
                    weight=weight,
                )
            # Selected muon histograms
            elif (
                "mu_" in histname
                and histname.replace("mu_", "") in pruned_ev.SelMuon.fields
                and "Plus" not in pruned_ev.fields
            ):
                h.fill(
                    syst,
                    flatten(pruned_ev.SelMuon[histname.replace("mu_", "")]),
                    weight=weight,
                )
            # Negatively charged lepton histograms in DY workflow
            elif (
                "negl_" in histname
                and histname.replace("negl_", "") in pruned_ev.negl.fields
            ):
                h.fill(
                    syst,
                    flatten(pruned_ev.negl[histname.replace("negl_", "")]),
                    weight=weight,
                )
            # Posively charged lepton histograms in DY workflow
            elif (
                "posl_" in histname
                and histname.replace("posl_", "") in pruned_ev.posl.fields
            ):
                h.fill(
                    syst,
                    flatten(pruned_ev.posl[histname.replace("posl_", "")]),
                    weight=weight,
                )
            # Soft muon histograms
            elif "soft_l" in histname and "ptratio" not in histname:
                h.fill(
                    syst,
                    smflav,
                    flatten(pruned_ev.SoftMuon[histname.replace("soft_l_", "")]),
                    weight=weight,
                )
            elif "norm" == histname:
                output["norm"].fill(syst, weight=weight)
            elif "njet" == histname:
                output["njet"].fill(syst, pruned_ev.njet, weight=weight)
            elif histname.startswith("tag_ladder_"):
                scheme = histname[len("tag_ladder_") :]
                branch, _codes = SF2D_SCHEMES.get(scheme, (None, None))
                if branch is not None and branch in pruned_ev.SelJet.fields:
                    jetcat = pruned_ev.SelJet[branch]
                    for thr in TAG_LADDER_THRESHOLDS:
                        codes = TAG_SEL_CODES[scheme].get(thr)
                        if codes is None:
                            continue
                        mask = ak.zeros_like(jetcat, dtype=bool)
                        for code in codes:
                            mask = mask | (jetcat == code)
                        count = ak.to_numpy(ak.sum(mask, axis=1))
                        output[histname].fill(
                            syst=syst,
                            threshold=np.full(len(count), thr),
                            count=count,
                            weight=weight,
                        )
            elif "nbjet" == histname:
                output["nbjet"].fill(syst, pruned_ev.nbjet, weight=weight)
            elif "ncjet" == histname:
                output["ncjet"].fill(syst, pruned_ev.ncjet, weight=weight)
            elif "top_pt" == histname and "TT" in pruned_ev.metadata["dataset"]:
                top_mask = (pruned_ev.GenPart.pdgId == 6) & pruned_ev.GenPart.hasFlags(
                    ["isLastCopy"]
                )
                top_pt = pruned_ev.GenPart[top_mask][:, 0].pt
                output["top_pt"].fill(syst, top_pt, weight=weight)
            elif "antitop_pt" == histname and "TT" in pruned_ev.metadata["dataset"]:
                antitop_mask = (
                    pruned_ev.GenPart.pdgId == -6
                ) & pruned_ev.GenPart.hasFlags(["isLastCopy"])
                antitop_pt = pruned_ev.GenPart[antitop_mask][:, 0].pt
                output["antitop_pt"].fill(syst, antitop_pt, weight=weight)
            elif "npv" == histname:
                output["npv"].fill(syst, pruned_ev.PV.npvsGood, weight=weight)
            # Jet kinematics & deltaR between jet and lepton
            elif (
                "jet" in histname and "posl" not in histname and "negl" not in histname
            ):
                for i in range(nj):
                    if f"jet{i}" not in histname:
                        continue
                    if nj == 1:
                        sel_jet, flav, wgt, temp_syst = (
                            pruned_ev.SelJet,
                            genflavor,
                            weight,
                            syst,
                        )
                    else:
                        sel_jet, flav, wgt, temp_syst = _jet_i(i)
                    if str(i) in histname:
                        if "dr_mujet" in histname:
                            h.fill(
                                temp_syst,
                                flatten(flav),
                                flatten(sel_jet.delta_r(pruned_ev.SelMuon)),
                                weight=wgt,
                            )
                        else:
                            h.fill(
                                temp_syst,
                                flatten(flav),
                                flatten(sel_jet[histname.replace(f"jet{i}_", "")]),
                                weight=wgt,
                            )
                # fill negatively tagged jets, and inclusive jets, binned in pt
                if histname.endswith("_negtag_jet_pt"):
                    col_name = histname.replace("_pt", "")
                    if col_name not in pruned_ev.fields:
                        continue
                    jet_col = pruned_ev[col_name]
                    jet_syst = np.full(len(flatten(jet_col.pt)), syst[0])
                    h.fill(
                        syst_perjet,
                        flatten(jet_col.flavor),
                        flatten(jet_col.pt),
                        weight=flatten(ak.broadcast_arrays(weight, jet_col.pt)[0]),
                    )
                elif histname.endswith("_negtag_jet_cWP_pt"):
                    col_name = histname.replace("_pt", "")
                    if col_name not in pruned_ev.fields:
                        continue
                    jet_col = pruned_ev[col_name]
                    jet_syst = np.full(len(flatten(jet_col.pt)), syst[0])
                    h.fill(
                        syst_perjet,
                        flatten(jet_col.flavor),
                        flatten(jet_col.pt),
                        weight=flatten(ak.broadcast_arrays(weight, jet_col.pt)[0]),
                    )
                elif histname.endswith("jet_pt") and "AllSelJet" in pruned_ev.fields:
                    jet_syst = np.full(len(flatten(pruned_ev["AllSelJet"].pt)), syst[0])
                    h.fill(
                        syst_perjet,
                        flatten(pruned_ev["AllSelJet"].flavor),
                        flatten(pruned_ev["AllSelJet"].pt),
                        weight=flatten(
                            ak.broadcast_arrays(weight, pruned_ev["AllSelJet"].pt)[0]
                        ),
                    )
            # Mu-jets distribution
            elif "lmujet_" in histname:
                h.fill(
                    syst,
                    smflav,
                    flatten(pruned_ev.MuonJet[histname.replace("lmujet_", "")]),
                    weight=weight,
                )
            # Filled discriminants
            elif "btag" in histname or "PNet" in histname:
                # Events with muon jet
                if "MuonJet" in pruned_ev.fields:
                    flavs, seljets = smflav, pruned_ev.MuonJet
                    nj = 1
                else:
                    flavs, seljets = genflavor, pruned_ev.SelJet
                # Check if histogram name has a trailing jet index suffix
                has_jet_index = any(histname.endswith(f"_{i}") for i in range(nj))
                if has_jet_index:
                    for i in range(nj):
                        if not histname.endswith(str(i)):
                            continue
                        if nj > 1 and "MuonJet" not in pruned_ev.fields:
                            seljet, flav, wgt, fsyst = _jet_i(i)
                        elif nj > 1:
                            # not every event has >=(i+1) jets (e.g. DY with
                            # up to 3 jets stored, but not guaranteed 3/event)
                            mask_njets = ak.count(seljets.pt, axis=1) >= (i + 1)
                            temp_flavs = flavs[mask_njets]
                            temp_seljets = seljets[mask_njets]
                            temp_weight = weight[mask_njets]
                            temp_syst = syst[mask_njets]
                            flav, seljet, wgt, fsyst = (
                                temp_flavs[:, i],
                                temp_seljets[:, i],
                                temp_weight,
                                temp_syst,
                            )
                        else:
                            flav, seljet, wgt, fsyst = flavs, seljets, weight, syst
                        field = histname.replace(f"_{i}", "")
                        if field.startswith("btagVs_"):
                            _, t1, t2, d, _ = field.split("_")
                            h.fill(
                                syst=fsyst,
                                flav=flav,
                                t1=_logit(seljet[f"btag{t1}{d}"]),
                                t2=_logit(seljet[f"btag{t2}{d}"]),
                                weight=wgt,
                            )
                        elif field.endswith("_logit"):
                            h.fill(
                                syst=fsyst,
                                flav=flav,
                                discr=_logit(seljet[field[: -len("_logit")]]),
                                weight=wgt,
                            )
                        else:
                            h.fill(
                                syst=fsyst,
                                flav=flav,
                                discr=seljet[field],
                                weight=wgt,
                            )
                elif histname in seljets.fields:
                    # No jet index suffix (nj=1) - fill directly
                    discr = seljets[histname]
                    flat_discr = flatten(discr)
                    temp_syst = np.full(len(flat_discr), syst[0])
                    h.fill(
                        syst=temp_syst,
                        flav=flatten(flavs),
                        discr=flat_discr,
                        weight=flatten(ak.broadcast_arrays(weight, discr)[0]),
                    )

        if "dr_poslnegl" in output.keys():
            # DY histograms
            output["dr_poslnegl"].fill(
                syst, pruned_ev.posl.delta_r(pruned_ev.negl), weight=weight
            )
            # SelJet may hold a variable number of jets/event (e.g. DY
            # storing up to 3 jets), so the per-jet dr/flavor arrays are
            # ragged and must be flattened before filling
            dr_posl = pruned_ev.posl.delta_r(pruned_ev.SelJet)
            flat_dr_posl = flatten(dr_posl)
            output["dr_posljet"].fill(
                np.full(len(flat_dr_posl), syst[0]),
                flatten(genflavor),
                flat_dr_posl,
                weight=flatten(ak.broadcast_arrays(weight, dr_posl)[0]),
            )
            dr_negl = pruned_ev.negl.delta_r(pruned_ev.SelJet)
            flat_dr_negl = flatten(dr_negl)
            output["dr_negljet"].fill(
                np.full(len(flat_dr_negl), syst[0]),
                flatten(genflavor),
                flat_dr_negl,
                weight=flatten(ak.broadcast_arrays(weight, dr_negl)[0]),
            )

        # Muon enriched jet histograms
        if "MuonJet" in pruned_ev.fields:
            if (
                "hl" not in pruned_ev.fields
                and "SelElectron" in pruned_ev.fields
                and "SelMuon" in pruned_ev.fields
            ):
                pruned_ev["hl"] = ak.zip(
                    {
                        "pt": (
                            pruned_ev.SelMuon.pt
                            if pruned_ev.SelMuon.pt > pruned_ev.SelElectron.pt
                            else pruned_ev.SelElectron.pt
                        ),
                        "eta": (
                            pruned_ev.SelMuon.eta
                            if pruned_ev.SelMuon.pt > pruned_ev.SelElectron.pt
                            else pruned_ev.SelElectron.eta
                        ),
                        "phi": (
                            pruned_ev.SelMuon.phi
                            if pruned_ev.SelMuon.pt > pruned_ev.SelElectron.pt
                            else pruned_ev.SelElectron.phi
                        ),
                        "energy": (
                            pruned_ev.SelMuon.energy
                            if pruned_ev.SelMuon.pt > pruned_ev.SelElectron.pt
                            else pruned_ev.SelElectron.energy
                        ),
                    },
                    with_name="PtEtaPhiECandidate",
                )
            output["soft_l_ptratio"].fill(
                syst,
                flav=smflav,
                ratio=pruned_ev.SoftMuon.pt / pruned_ev.MuonJet.pt,
                weight=weight,
            )
            output["dr_lmujetsmu"].fill(
                syst,
                flav=smflav,
                dr=pruned_ev.MuonJet.delta_r(pruned_ev.SoftMuon),
                weight=weight,
            )
            if "hl" in pruned_ev.fields:
                output["hl_ptratio"].fill(
                    syst,
                    smflav,
                    ratio=pruned_ev.hl.pt / pruned_ev.MuonJet.pt,
                    weight=weight,
                )
            if "sl" in pruned_ev.fields:
                output["sl_ptratio"].fill(
                    syst,
                    smflav,
                    ratio=pruned_ev.sl.pt / pruned_ev.MuonJet.pt,
                    weight=weight,
                )
            if "SelMuon" in pruned_ev.fields and "hl" not in pruned_ev.fields:
                output["dr_lmujethmu"].fill(
                    syst,
                    flav=smflav,
                    dr=pruned_ev.MuonJet.delta_r(pruned_ev.SelMuon),
                    weight=weight,
                )
                output["dr_hmusmu"].fill(
                    syst, pruned_ev.SelMuon.delta_r(pruned_ev.SoftMuon), weight=weight
                )

        # dilepton system histograms: DY workflow
        if "dilep" in pruned_ev.fields:
            output["dilep_pt"].fill(syst, flatten(pruned_ev.dilep.pt), weight=weight)
            output["dilep_eta"].fill(syst, flatten(pruned_ev.dilep.eta), weight=weight)
            output["dilep_phi"].fill(syst, flatten(pruned_ev.dilep.phi), weight=weight)
            output["dilep_mass"].fill(
                syst, flatten(pruned_ev.dilep.mass), weight=weight
            )
            if "dilep_ptratio" in output.keys():
                # SelJet may hold a variable number of jets/event, so the
                # per-jet ratio/flavor arrays are ragged and must be
                # flattened (and syst/weight broadcast per-jet) before filling
                ratio = pruned_ev.dilep.pt / pruned_ev.SelJet.pt
                flat_ratio = flatten(ratio)
                output["dilep_ptratio"].fill(
                    np.full(len(flat_ratio), syst[0]),
                    flatten(genflavor),
                    flat_ratio,
                    weight=flatten(ak.broadcast_arrays(weight, ratio)[0]),
                )

        if "MET_pt" in output.keys():
            output["MET_pt"].fill(syst, flatten(pruned_ev.MET.pt), weight=weight)
            output["MET_phi"].fill(syst, flatten(pruned_ev.MET.phi), weight=weight)

        if "w_mt" in pruned_ev.fields:
            output["w_mt"].fill(syst, flatten(pruned_ev.w_mt), weight=weight)

        # ttbar dilepton kin workflow
        if "kindisc" in output.keys():
            flat_flav = flatten(pruned_ev.flavour)
            jet_syst = np.full(len(flat_flav), syst[0])
            output["kindisc"].fill(
                jet_syst,
                flat_flav,
                flatten(pruned_ev.kindisc),
                weight=flatten(ak.broadcast_arrays(weight, pruned_ev.kindisc)[0]),
            )
            output["close_mlj"].fill(
                jet_syst,
                flat_flav,
                flatten(pruned_ev.close_mlj),
                weight=flatten(ak.broadcast_arrays(weight, pruned_ev.close_mlj)[0]),
            )
            output["close_deta"].fill(
                jet_syst,
                flat_flav,
                flatten(pruned_ev.close_deta),
                weight=flatten(ak.broadcast_arrays(weight, pruned_ev.close_deta)[0]),
            )
            output["close_dphi"].fill(
                jet_syst,
                flat_flav,
                flatten(pruned_ev.close_dphi),
                weight=flatten(ak.broadcast_arrays(weight, pruned_ev.close_dphi)[0]),
            )
            output["close_ptrel"].fill(
                jet_syst,
                flat_flav,
                flatten(pruned_ev.close_ptrel),
                weight=flatten(ak.broadcast_arrays(weight, pruned_ev.close_ptrel)[0]),
            )
            output["close_lj2ll_deta"].fill(
                jet_syst,
                flat_flav,
                flatten(pruned_ev.close_lj2ll_deta),
                weight=flatten(
                    ak.broadcast_arrays(weight, pruned_ev.close_lj2ll_deta)[0]
                ),
            )
            output["close_lj2ll_dphi"].fill(
                jet_syst,
                flat_flav,
                flatten(pruned_ev.close_lj2ll_dphi),
                weight=flatten(
                    ak.broadcast_arrays(weight, pruned_ev.close_lj2ll_dphi)[0]
                ),
            )
            output["far_mlj"].fill(
                jet_syst,
                flat_flav,
                flatten(pruned_ev.far_mlj),
                weight=flatten(ak.broadcast_arrays(weight, pruned_ev.far_mlj)[0]),
            )
            output["far_deta"].fill(
                jet_syst,
                flat_flav,
                flatten(pruned_ev.far_deta),
                weight=flatten(ak.broadcast_arrays(weight, pruned_ev.far_deta)[0]),
            )
            output["far_dphi"].fill(
                jet_syst,
                flat_flav,
                flatten(pruned_ev.far_dphi),
                weight=flatten(ak.broadcast_arrays(weight, pruned_ev.far_dphi)[0]),
            )
            output["far_ptrel"].fill(
                jet_syst,
                flat_flav,
                flatten(pruned_ev.far_ptrel),
                weight=flatten(ak.broadcast_arrays(weight, pruned_ev.far_ptrel)[0]),
            )
            output["far_lj2ll_deta"].fill(
                jet_syst,
                flat_flav,
                flatten(pruned_ev.far_lj2ll_deta),
                weight=flatten(
                    ak.broadcast_arrays(weight, pruned_ev.far_lj2ll_deta)[0]
                ),
            )
            output["far_lj2ll_dphi"].fill(
                jet_syst,
                flat_flav,
                flatten(pruned_ev.far_lj2ll_dphi),
                weight=flatten(
                    ak.broadcast_arrays(weight, pruned_ev.far_lj2ll_dphi)[0]
                ),
            )
            output["j2ll_deta"].fill(
                jet_syst,
                flat_flav,
                flatten(pruned_ev.j2ll_deta),
                weight=flatten(ak.broadcast_arrays(weight, pruned_ev.j2ll_deta)[0]),
            )
            output["j2ll_dphi"].fill(
                jet_syst,
                flat_flav,
                flatten(pruned_ev.j2ll_dphi),
                weight=flatten(ak.broadcast_arrays(weight, pruned_ev.j2ll_dphi)[0]),
            )

        # SF-fit templates (Task A4/histograms/sf2d.py), shared by every
        # workflow that calls histo_writer (DY, ttdilep, ...). W+c fills
        # these itself alongside its other inline histograms instead, since
        # it doesn't go through histo_writer.
        if any(k.startswith("sf2d_") for k in output.keys()):
            _fill_sf2d_histograms(output, pruned_ev.SelJet, genflavor, weight, syst)
        # s-tagging templates and discriminants (histograms/stag.py), same jets
        if any(k.startswith(STAG_PREFIXES) for k in output.keys()):
            _fill_stag_histograms(output, pruned_ev.SelJet, weight, syst)

    variations = set(weights.variations)

    def weight_for(syst):
        if syst == "nominal" or syst not in variations:
            return weights.weight()
        return weights.weight(modifier=syst)

    systs = list(
        itertools.takewhile(
            lambda s: not (isSyst == False and s != "nominal"), systematics
        )
    )
    if "MuonJet" in pruned_ev.fields:
        # the btag branch sets nj = 1 for all later histograms and variations: keep the plain loop
        for syst in systs:
            _fill_all(output, weight_for(syst), syst)
    else:
        replay_fills(
            output,
            _fill_all,
            systs,
            weight_for,
            len(weights.weight()),
            skip=lambda name, syst: "_logit_" in name and syst != "nominal",
        )
    return output
