from BTVNanoCommissioning.helpers.func import update
from BTVNanoCommissioning.utils.selection import btag_wp_dict
import numpy as np
import awkward as ak

_BTAG_2D_MAPPING_CACHE = {}


def _get_2d_mapping_array(campaign, tagger):
    """Convert btag_wp_dict[campaign][tagger]["2D"]["mapping"] (a static dict-of-dicts,
    keyed by the 1-based np.digitize bin indices) into a padded numpy 2D array once, and
    cache it. Row/column 0 and the last row/column (beyond the mapping's max key) are
    filled with -1 so any in-range np.digitize output -- including the out-of-bin values
    produced by the -1 sentinel or an overflow value -- can be used directly as an index
    without raising.
    """
    cache_key = (campaign, tagger)
    if cache_key in _BTAG_2D_MAPPING_CACHE:
        return _BTAG_2D_MAPPING_CACHE[cache_key]

    mapping_dict = btag_wp_dict[campaign][tagger]["2D"]["mapping"]
    max_i = max(mapping_dict)
    max_j = max(max(row) for row in mapping_dict.values())
    mapping_arr = np.full((max_i + 2, max_j + 2), -1, dtype=np.int64)
    for i, row in mapping_dict.items():
        for j, val in row.items():
            mapping_arr[i, j] = val

    _BTAG_2D_MAPPING_CACHE[cache_key] = mapping_arr
    return mapping_arr


def missing_branch(events, campaign=None, extra_taggers=()):
    """
    Add missing branches or rename branches in the `events` object.

    This function adds missing branches or renames existing branches in the `events` object using the `missing_branch` parameter.

    Usage:
    Use the `hasattr` function to check for missing branches.

    Deprecated:
    The `add_jec` function is deprecated. Please use the `JME_shifts` function in the `correction` module instead.

    Example:
    ```python
    events.fixedGridRhoFastjetAll = (
        events.fixedGridRhoFastjetAll
        if hasattr(events, "fixedGridRhoFastjetAll")
        else events.Rho.fixedGridRhoFastjetAll
    )
    ```

    Parameters:
    events (coffea.nanoaodevents): The events object to update.
    missing_branch (str): The name of the missing branch to add or rename.

    Returns:
    events (coffea.nanoaodevents): Events with updated branches.

    """
    # Function implementation here
    if hasattr(events, "fixedGridRhoFastjetAll"):
        pass  # already a top-level branch, nothing to do
    elif hasattr(events, "Rho") and hasattr(events.Rho, "fixedGridRhoFastjetAll"):
        events["fixedGridRhoFastjetAll"] = events.Rho.fixedGridRhoFastjetAll
    else:
        raise AttributeError(
            "fixedGridRhoFastjetAll not found as top-level branch or under Rho collection. "
            "This branch is required for JEC corrections. "
            "Check that the input NanoAOD file contains Rho_fixedGridRhoFastjetAll."
        )

    ## calculate missing nodes
    if not hasattr(events.Jet, "btagDeepFlavB"):
        jets = events.Jet
        jets["btagDeepFlavB"] = (
            events.Jet.btagDeepFlavB_b
            + events.Jet.btagDeepFlavB_bb
            + events.Jet.btagDeepFlavB_lepb
        )
        events["Jet"] = update(
            events.Jet,
            {"btagDeepFlavB": jets.btagDeepFlavB},
        )
    if (
        hasattr(events.Jet, "btagDeepFlavCvL")
        and hasattr(events.Jet, "btagDeepFlavUDS")
        and not hasattr(events.Jet, "btagDeepFlavC")
    ):
        jets = events.Jet
        jets["btagDeepFlavC"] = (
            events.Jet.btagDeepFlavCvL / (1.0 - events.Jet.btagDeepFlavCvL)
        ) * (events.Jet.btagDeepFlavG + events.Jet.btagDeepFlavUDS)
        events["Jet"] = update(
            events.Jet,
            {"btagDeepFlavC": jets.btagDeepFlavC},
        )
    if hasattr(events.Jet, "btagDeepFlavCvB") and not hasattr(
        events.Jet, "btagDeepFlavC"
    ):
        jets = events.Jet
        jets["btagDeepFlavC"] = (
            events.Jet.btagDeepFlavCvB / (1.0 - events.Jet.btagDeepFlavCvB)
        ) * (events.Jet.btagDeepFlavB)
        events["Jet"] = update(
            events.Jet,
            {"btagDeepFlavC": jets.btagDeepFlavC},
        )
    if hasattr(events.Jet, "btagDeepFlavC") and not hasattr(
        events.Jet, "btagDeepFlavCvL"
    ):
        jets = events.Jet
        jets["btagDeepFlavCvL"] = np.maximum(
            np.minimum(
                np.where(
                    ((events.Jet.btagDeepFlavC / (1.0 - events.Jet.btagDeepFlavB)) > 0)
                    & (events.Jet.pt > 15),
                    (events.Jet.btagDeepFlavC / (1.0 - events.Jet.btagDeepFlavB)),
                    -1,
                ),
                0.999999,
            ),
            -1,
        )
        jets["btagDeepFlavCvB"] = np.maximum(
            np.minimum(
                np.where(
                    (
                        (
                            events.Jet.btagDeepFlavC
                            / (events.Jet.btagDeepFlavC + events.Jet.btagDeepFlavB)
                        )
                        > 0
                    )
                    & (events.Jet.pt > 15),
                    (
                        events.Jet.btagDeepFlavC
                        / (events.Jet.btagDeepFlavC + events.Jet.btagDeepFlavB)
                    ),
                    -1,
                ),
                0.999999,
            ),
            -1,
        )
        events["Jet"] = update(
            events.Jet,
            {
                "btagDeepFlavCvL": jets.btagDeepFlavCvL,
                "btagDeepFlavCvB": jets.btagDeepFlavCvB,
            },
        )
    if not hasattr(events.Jet, "btagPNetCvNotB") and hasattr(events.Jet, "btagPNetB"):
        jets = events.Jet
        jets["btagPNetCvNotB"] = (
            jets.btagPNetCvB * jets.btagPNetB / (1.0 - jets.btagPNetB) ** 2
        )
        events["Jet"] = update(
            events.Jet,
            {"btagPNetCvNotB": jets.btagPNetCvNotB},
        )
    if not hasattr(events.Jet, "btagRobustParTAK4CvNotB") and hasattr(
        events.Jet, "btagRobustParTAK4B"
    ):
        jets = events.Jet
        jets["btagRobustParTAK4CvNotB"] = (
            jets.btagRobustParTAK4CvB
            * jets.btagRobustParTAK4B
            / (1.0 - jets.btagRobustParTAK4B) ** 2
        )
        events["Jet"] = update(
            events.Jet,
            {"btagRobustParTAK4CvNotB": jets.btagRobustParTAK4CvNotB},
        )
    # NoPIDParTAK4 exposes raw per-flavour class probabilities instead of a
    # CvB discriminant; the base BvC score is derived from those probabilities.
    _nopid_probs = (
        "btagNoPIDParTAK4probb",
        "btagNoPIDParTAK4probbb",
        "btagNoPIDParTAK4probc",
        "btagNoPIDParTAK4probcc",
        "btagNoPIDParTAK4probg",
        "btagNoPIDParTAK4probud",
        "btagNoPIDParTAK4probs",
    )

    # extra_taggers (e.g. PNet, DeepFlav) only for the tagger comparison: adding
    # them by default would change the booked histograms of central campaigns.
    for tagger in ("UParTAK4", "UParTAK4NoAdv", "NoPIDParTAK4") + tuple(extra_taggers):
        bvc_name = f"btag{tagger}BvC"
        cvb_name = f"btag{tagger}CvB"
        base_bvc = None
        if not hasattr(events.Jet, bvc_name):
            if hasattr(events.Jet, cvb_name):
                cvb = events.Jet[cvb_name]
                base_bvc = ak.where(cvb < 0.0, -1, 1.0 - cvb)
            elif tagger == "NoPIDParTAK4" and all(
                hasattr(events.Jet, p) for p in _nopid_probs
            ):
                b = events.Jet.btagNoPIDParTAK4probb + events.Jet.btagNoPIDParTAK4probbb
                c = events.Jet.btagNoPIDParTAK4probc + events.Jet.btagNoPIDParTAK4probcc
                base_bvc = ak.where((b + c) <= 0.0, -1, b / (b + c))
        if base_bvc is not None:
            jets = events.Jet
            jets[bvc_name] = base_bvc
            jets[f"{bvc_name}t"] = ak.where(
                jets[bvc_name] < 0.0,
                -1,
                1.0 - np.sqrt(1.0 - jets[bvc_name]),
            )
            for lo, hi in (
                (25.0, 35.0),
                (35.0, 50.0),
                (50.0, 70.0),
                (70.0, 90.0),
                (90.0, 120.0),
            ):
                jets[f"{bvc_name}_pt{int(lo)}to{int(hi)}"] = ak.where(
                    (jets.pt < lo) | (jets.pt >= hi),
                    -2,
                    jets[bvc_name],
                )
            jets[f"{bvc_name}_pt120toinf"] = ak.where(
                jets.pt < 120.0,
                -2,
                jets[bvc_name],
            )

            update_map = {
                bvc_name: jets[bvc_name],
                f"{bvc_name}t": jets[f"{bvc_name}t"],
            }
            for lo, hi in ((25, 35), (35, 50), (50, 70), (70, 90), (90, 120)):
                update_map[f"{bvc_name}_pt{lo}to{hi}"] = jets[
                    f"{bvc_name}_pt{lo}to{hi}"
                ]
            update_map[f"{bvc_name}_pt120toinf"] = jets[f"{bvc_name}_pt120toinf"]
            events["Jet"] = update(events.Jet, update_map)

        hfvlf_name = f"btag{tagger}HFvLF"
        if "UParTAK4" in tagger:
            if (
                not hasattr(events.Jet, hfvlf_name)
                and hasattr(events.Jet, f"btag{tagger}UDG")
                and hasattr(events.Jet, f"btag{tagger}SvUDG")
                and hasattr(events.Jet, f"btag{tagger}CvL")
                and hasattr(events.Jet, f"btag{tagger}CvB")
            ):
                jets = events.Jet
                udg = jets[f"btag{tagger}UDG"]
                svudg = jets[f"btag{tagger}SvUDG"]
                cvl = jets[f"btag{tagger}CvL"]
                cvb = jets[f"btag{tagger}CvB"]

                jets[f"btag{tagger}S"] = ak.where(
                    svudg < 0.0,
                    -1,
                    svudg * udg / (1.0 - svudg),
                )
                jets[f"btag{tagger}C"] = ak.where(
                    svudg < 0.0,
                    -1,
                    cvl * (jets[f"btag{tagger}S"] + udg) / (1.0 - cvl),
                )
                jets[f"btag{tagger}B"] = ak.where(
                    svudg < 0.0,
                    -1,
                    (1.0 - cvb) * jets[f"btag{tagger}C"] / cvb,
                )
                jets[hfvlf_name] = ak.where(
                    svudg < 0.0,
                    -1,
                    (jets[f"btag{tagger}B"] + jets[f"btag{tagger}C"])
                    / (
                        jets[f"btag{tagger}B"]
                        + jets[f"btag{tagger}C"]
                        + jets[f"btag{tagger}S"]
                        + udg
                    ),
                )
                jets[f"{hfvlf_name}t"] = ak.where(
                    jets[hfvlf_name] < 0.0,
                    -1,
                    1.0 - np.sqrt(1.0 - jets[hfvlf_name]),
                )

                for lo, hi in (
                    (25.0, 35.0),
                    (35.0, 50.0),
                    (50.0, 70.0),
                    (70.0, 90.0),
                    (90.0, 120.0),
                ):
                    jets[f"{hfvlf_name}_pt{int(lo)}to{int(hi)}"] = ak.where(
                        (jets.pt < lo) | (jets.pt >= hi),
                        -2,
                        jets[hfvlf_name],
                    )
                jets[f"{hfvlf_name}_pt120toinf"] = ak.where(
                    jets.pt < 120.0,
                    -2,
                    jets[hfvlf_name],
                )

                update_map = {
                    hfvlf_name: jets[hfvlf_name],
                    f"{hfvlf_name}t": jets[f"{hfvlf_name}t"],
                }
                for lo, hi in ((25, 35), (35, 50), (50, 70), (70, 90), (90, 120)):
                    update_map[f"{hfvlf_name}_pt{lo}to{hi}"] = jets[
                        f"{hfvlf_name}_pt{lo}to{hi}"
                    ]
                update_map[f"{hfvlf_name}_pt120toinf"] = jets[
                    f"{hfvlf_name}_pt120toinf"
                ]
                # Keep the pure c and s class scores (not stored in NanoAOD).
                if tagger == "UParTAK4NoAdv":
                    update_map[f"btag{tagger}C"] = jets[f"btag{tagger}C"]
                    update_map[f"btag{tagger}S"] = jets[f"btag{tagger}S"]
                events["Jet"] = update(events.Jet, update_map)

        elif tagger == "NoPIDParTAK4":
            if not hasattr(events.Jet, hfvlf_name) and all(
                hasattr(events.Jet, p) for p in _nopid_probs
            ):
                jets = events.Jet
                b = jets.btagNoPIDParTAK4probb + jets.btagNoPIDParTAK4probbb
                c = jets.btagNoPIDParTAK4probc + jets.btagNoPIDParTAK4probcc
                lf = (
                    jets.btagNoPIDParTAK4probg
                    + jets.btagNoPIDParTAK4probud
                    + jets.btagNoPIDParTAK4probs
                )
                jets[hfvlf_name] = ak.where(
                    (b + c + lf) <= 0.0, -1, (b + c) / (b + c + lf)
                )
                jets[f"{hfvlf_name}t"] = ak.where(
                    jets[hfvlf_name] < 0.0,
                    -1,
                    1.0 - np.sqrt(1.0 - jets[hfvlf_name]),
                )

                for lo, hi in (
                    (25.0, 35.0),
                    (35.0, 50.0),
                    (50.0, 70.0),
                    (70.0, 90.0),
                    (90.0, 120.0),
                ):
                    jets[f"{hfvlf_name}_pt{int(lo)}to{int(hi)}"] = ak.where(
                        (jets.pt < lo) | (jets.pt >= hi),
                        -2,
                        jets[hfvlf_name],
                    )
                jets[f"{hfvlf_name}_pt120toinf"] = ak.where(
                    jets.pt < 120.0,
                    -2,
                    jets[hfvlf_name],
                )

                update_map = {
                    hfvlf_name: jets[hfvlf_name],
                    f"{hfvlf_name}t": jets[f"{hfvlf_name}t"],
                }
                for lo, hi in ((25, 35), (35, 50), (50, 70), (70, 90), (90, 120)):
                    update_map[f"{hfvlf_name}_pt{lo}to{hi}"] = jets[
                        f"{hfvlf_name}_pt{lo}to{hi}"
                    ]
                update_map[f"{hfvlf_name}_pt120toinf"] = jets[
                    f"{hfvlf_name}_pt120toinf"
                ]
                events["Jet"] = update(events.Jet, update_map)

        elif tagger in ("DeepFlav", "PNet"):
            if (
                not hasattr(events.Jet, hfvlf_name)
                and hasattr(events.Jet, f"btag{tagger}B")
                and hasattr(events.Jet, f"btag{tagger}CvL")
            ):
                jets = events.Jet
                b = jets[f"btag{tagger}B"]
                cvl = jets[f"btag{tagger}CvL"]
                jets[f"btag{tagger}C"] = ak.where(
                    b < 0.0,
                    -1,
                    cvl * (1 - b),
                )
                jets[hfvlf_name] = ak.where(
                    b < 0.0,
                    -1,
                    (b + jets[f"btag{tagger}C"]),
                )
                jets[f"{hfvlf_name}t"] = ak.where(
                    jets[hfvlf_name] < 0.0,
                    -1,
                    1.0 - np.sqrt(1.0 - jets[hfvlf_name]),
                )

                for lo, hi in (
                    (25.0, 35.0),
                    (35.0, 50.0),
                    (50.0, 70.0),
                    (70.0, 90.0),
                    (90.0, 120.0),
                ):
                    jets[f"{hfvlf_name}_pt{int(lo)}to{int(hi)}"] = ak.where(
                        (jets.pt < lo) | (jets.pt >= hi),
                        -2,
                        jets[hfvlf_name],
                    )
                jets[f"{hfvlf_name}_pt120toinf"] = ak.where(
                    jets.pt < 120.0,
                    -2,
                    jets[hfvlf_name],
                )

                update_map = {
                    hfvlf_name: jets[hfvlf_name],
                    f"{hfvlf_name}t": jets[f"{hfvlf_name}t"],
                }
                for lo, hi in ((25, 35), (35, 50), (50, 70), (70, 90), (90, 120)):
                    update_map[f"{hfvlf_name}_pt{lo}to{hi}"] = jets[
                        f"{hfvlf_name}_pt{lo}to{hi}"
                    ]
                update_map[f"{hfvlf_name}_pt120toinf"] = jets[
                    f"{hfvlf_name}_pt120toinf"
                ]
                events["Jet"] = update(events.Jet, update_map)

        # 2Dbin variants sharing this tagger's raw HFvLF/BvC scores: the
        # tagger's own (legacy) WP boundaries, plus the phase4 scheme for
        # NoPIDParTAK4 (SFbc-2D/OptimizedWP). Same hfvlf/bvc branches, just a
        # different btag_wp_dict entry (cuts + mapping) and output branch name.
        wp_keys = [tagger]
        if tagger == "NoPIDParTAK4":
            wp_keys.append("NoPIDParTAK4_phase4")

        for wp_key in wp_keys:
            bin2d_name = f"btag{wp_key}2Dbin"
            if (
                not hasattr(events.Jet, bin2d_name)
                and hasattr(events.Jet, hfvlf_name)
                and hasattr(events.Jet, bvc_name)
                # 2D WP boundaries aren't calibrated for every (campaign,
                # tagger) pair -- skip rather than KeyError.
                and "2D" in btag_wp_dict.get(campaign, {}).get(wp_key, {})
            ):
                jets = events.Jet
                hfvlf = ak.flatten(jets[hfvlf_name])
                bvc = ak.flatten(jets[bvc_name])
                nj = ak.num(jets)

                ihfvlf = np.digitize(
                    hfvlf, btag_wp_dict[campaign][wp_key]["2D"]["HFvLF"]
                )
                ibvc = np.digitize(bvc, btag_wp_dict[campaign][wp_key]["2D"]["BvC"])

                mapping_arr = _get_2d_mapping_array(campaign, wp_key)
                ihfvlf_idx = np.clip(ihfvlf, 0, mapping_arr.shape[0] - 1)
                ibvc_idx = np.clip(ibvc, 0, mapping_arr.shape[1] - 1)
                bin2d_vals = np.where(
                    hfvlf == -1, -1, mapping_arr[ihfvlf_idx, ibvc_idx]
                )

                jets[bin2d_name] = ak.unflatten(bin2d_vals, nj)

                for lo, hi in (
                    (25.0, 35.0),
                    (35.0, 50.0),
                    (50.0, 70.0),
                    (70.0, 90.0),
                    (90.0, 120.0),
                ):
                    jets[f"{bin2d_name}_pt{int(lo)}to{int(hi)}"] = ak.where(
                        (jets.pt < lo) | (jets.pt >= hi),
                        -2,
                        jets[bin2d_name],
                    )
                jets[f"{bin2d_name}_pt120toinf"] = ak.where(
                    jets.pt < 120.0,
                    -2,
                    jets[bin2d_name],
                )

                update_map = {bin2d_name: jets[bin2d_name]}
                for lo, hi in ((25, 35), (35, 50), (50, 70), (70, 90), (90, 120)):
                    update_map[f"{bin2d_name}_pt{lo}to{hi}"] = jets[
                        f"{bin2d_name}_pt{lo}to{hi}"
                    ]
                update_map[f"{bin2d_name}_pt120toinf"] = jets[
                    f"{bin2d_name}_pt120toinf"
                ]
                events["Jet"] = update(events.Jet, update_map)

    if hasattr(events, "METFixEE2017"):
        events["MET"] = events.METFixEE2017
    if hasattr(events.PuppiMET, "ptUnclusteredUp") and not hasattr(
        events.PuppiMET, "MetUnclustEnUpDeltaX"
    ):
        met = events.PuppiMET
        met["MetUnclustEnUpDeltaX"] = (met.ptUnclusteredUp - met.pt) * np.cos(
            met.phiUnclusteredUp
        )
        met["MetUnclustEnUpDeltaY"] = (met.ptUnclusteredUp - met.pt) * np.sin(
            met.phiUnclusteredUp
        )
        events["PuppiMET"] = update(
            events.PuppiMET,
            {
                "MetUnclustEnUpDeltaX": met.MetUnclustEnUpDeltaX,
                "MetUnclustEnUpDeltaY": met.MetUnclustEnUpDeltaY,
            },
        )
    return events
