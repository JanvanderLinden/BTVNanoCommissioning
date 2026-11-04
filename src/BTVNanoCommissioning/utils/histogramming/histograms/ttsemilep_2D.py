import hist as Hist

from .sf2d import SCHEMES, TAG_LADDER_THRESHOLDS, TAG_SEL_CODES


def get_histograms(axes, **kwargs):
    hists = {}

    hists[f"nbjet"] = Hist.Hist(axes["syst"], axes["n"], Hist.storage.Weight())
    hists[f"ncjet"] = Hist.Hist(axes["syst"], axes["n"], Hist.storage.Weight())
    hists[f"w_mt"] = Hist.Hist(axes["syst"], axes["mt"], Hist.storage.Weight())

    channel = kwargs.get("channel", "mu")
    hists[f"{channel}_pfRelIso04_all"] = Hist.Hist(
        axes["syst"], axes["iso"], Hist.storage.Weight()
    )
    hists[f"{channel}_dxy"] = Hist.Hist(
        axes["syst"], axes["dxy"], Hist.storage.Weight()
    )
    hists[f"{channel}_dz"] = Hist.Hist(axes["syst"], axes["dz"], Hist.storage.Weight())

    hists["top_pt"] = Hist.Hist(axes["syst"], axes["jpt"], Hist.storage.Weight())
    hists["antitop_pt"] = Hist.Hist(axes["syst"], axes["jpt"], Hist.storage.Weight())

    # Per-scheme tag-multiplicity ladder (Run 2 parity: makePlotsTT1L.C's
    # n_b_loose/medium/tight/VT/VVT + n_c_* control plots, via
    # countFlavTag(n_ak4, ak4_tag, threshold)). One histogram per scheme,
    # counted over every stored SelJet (up to 4), not just the two leading --
    # same convention as nbjet_<scheme>/ncjet_<scheme>. nopid_phase4 only has
    # the two thresholds with an unambiguous geometric meaning (see
    # TAG_SEL_CODES in sf2d.py).
    for scheme in SCHEMES:
        thresholds = [t for t in TAG_LADDER_THRESHOLDS if t in TAG_SEL_CODES[scheme]]
        hists[f"tag_ladder_{scheme}"] = Hist.Hist(
            axes["syst"],
            Hist.axis.StrCategory(thresholds, name="threshold"),
            Hist.axis.Integer(0, 5, name="count", overflow=True),
            Hist.storage.Weight(),
        )

    return hists
