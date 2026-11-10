import numpy as np
import hist as Hist

from BTVNanoCommissioning.helpers.stag_helpers import (
    TAG_BIN_NAMES,
    STAG_FLAVOURS,
    NOPID_PROBS,
    NOPID_DISCRIMINANTS,
)

# s-tagging histograms (3D b/c/s cascade, sfs-3d WPs), filled in the stag_* W+c, DY and
# tt e-mu workflows for the probe jets:
#   stagtpl_<var>:      SF-fit templates, s-tag bin x jet pT x <var>, <var> = nSVs,
#                       svmass, nConstituents
#   stagscore_<axis>:   BvAll (bvall), CvNotB (cvnotb), SvUDG (svudg) exactly as used for
#                       the WPs and the calibration: score_<axis> corrections of the WP
#                       json, reweighted where the WPs are, clamped into the axis range
#   stagdisc_<name>:    NoPIDParTAK4 class probabilities and the unreweighted
#                       discriminants BvAll, CvNotB, SvUDG, SvUD
# All with the s-tagging flavour axis (stagflav: b, bb, c, cc, s, ud, g, PU; data: 0).
# Histogram names must not contain "jet", "btag", "PNet", "Deep", "soft_l" or "posl":
# the generic fill loop in histogrammer.histo_writter dispatches on those substrings.

# Jet pT grid of the s-tagging templates; the same edges are the pT bins of the WPs
# (sfs-3d wp/configs: wp.pt_bins, last WP bin 200-250 GeV = all jets above 200 GeV)
PT_EDGES = [25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
PT_EDGES += [120.0, 140.0, 160.0, 180.0, 200.0, np.inf]
STAG_SCORES = ("bvall", "cvnotb", "svudg")
SCORE_LABELS = {"bvall": "BvAll", "cvnotb": "CvNotB", "svudg": "SvUDG"}
NCONSTITUENTS_EDGES = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30]  # > 30: overflow
SV_MASS_EDGES = [-1.5, 0, 0.5, 0.75, 1.0, 1.3, 1.6, 2.0, 2.5, 3.5, np.inf]


def stag_flavour_axis():
    return Hist.axis.IntCategory(
        list(range(len(STAG_FLAVOURS))),
        name="flav",
        label="Jet flavour (" + ", ".join(STAG_FLAVOURS) + ")",
    )


def get_histograms(axes, **kwargs):
    include_osss = kwargs.get("include_osss", False)
    scores = kwargs.get("stag_scores", STAG_SCORES)

    common_axes = [axes["syst"], stag_flavour_axis()]
    if include_osss:
        common_axes.append(axes["osss"])

    tagcat = Hist.axis.IntCategory(
        list(range(len(TAG_BIN_NAMES))),
        name="tagcat",
        label="s-tag bin (" + ", ".join(TAG_BIN_NAMES) + ")",
    )
    pt = Hist.axis.Variable(PT_EDGES, name="pt", label="Jet $p_{T}$ [GeV]")
    tpl_axes = {
        "nSVs": Hist.axis.Integer(
            0, 5, name="nSVs", label="N secondary vertices", overflow=True
        ),
        "svmass": Hist.axis.Variable(
            SV_MASS_EDGES, name="svmass", label="SV mass [GeV]"
        ),
        "nConstituents": Hist.axis.Variable(
            NCONSTITUENTS_EDGES, name="nConstituents", label="N jet constituents"
        ),
    }

    hists = {}
    for var, ax in tpl_axes.items():
        hists[f"stagtpl_{var}"] = Hist.Hist(
            *common_axes, tagcat, pt, ax, Hist.storage.Weight()
        )
    for axis in scores:
        hists[f"stagscore_{axis}"] = Hist.Hist(
            *common_axes,
            Hist.axis.Regular(
                50,
                0,
                1,
                name="discr",
                label=f"{SCORE_LABELS.get(axis, axis)} (WP score, as calibrated)",
            ),
            Hist.storage.Weight(),
        )
    for name in NOPID_PROBS + NOPID_DISCRIMINANTS:
        hists[f"stagdisc_{name}"] = Hist.Hist(
            *common_axes,
            Hist.axis.Regular(
                50,
                0,
                1,
                name="discr",
                label=f"NoPIDParT {name}"
                + (" (not reweighted)" if name in NOPID_DISCRIMINANTS else ""),
            ),
            Hist.storage.Weight(),
        )
    return hists
