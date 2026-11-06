import re

import hist as Hist

from BTVNanoCommissioning.utils.histogramming import hist_helpers
from BTVNanoCommissioning.helpers.definitions import get_definitions, get_discriminators


def get_histograms(axes, **kwargs):
    hists = {}

    include_osss = kwargs.get("include_osss", False)

    include_npv = kwargs.get("include_npv", True)
    include_njet = kwargs.get("include_njet", True)
    include_nmujet = kwargs.get("include_nmujet", False)
    include_nsoftmu = kwargs.get("include_nsoftmu", False)
    # One-bin, syst-only sum of weights: the total event/jet norm of the
    # channel, filled once per systematic. Present in every "common" channel
    # (Run 2 parity: makePlotsTT1L.C's dedicated "Normalization" histogram),
    # so `checkSFNorm.py`-style checks are a single-number read instead of an
    # integral over a kinematic histogram.
    include_norm = kwargs.get("include_norm", True)

    include_discriminators = kwargs.get("include_discriminators", True)
    include_common_definitions = kwargs.get("include_common_definitions", True)
    include_sv_variables = kwargs.get("include_sv_variables", False)

    if include_osss:
        n_axes = [axes["syst"], axes["osss"], axes["n"]]
        npv_axes = [axes["syst"], axes["osss"], axes["npv"]]
    else:
        n_axes = [axes["syst"], axes["n"]]
        npv_axes = [axes["syst"], axes["npv"]]

    if include_npv:
        hists["npv"] = Hist.Hist(*npv_axes, Hist.storage.Weight())

    if include_njet:
        hists["njet"] = Hist.Hist(*n_axes, Hist.storage.Weight())

    if include_norm:
        hists["norm"] = Hist.Hist(axes["syst"], Hist.storage.Weight())

    if include_nmujet:
        hists["nmujet"] = Hist.Hist(*n_axes, Hist.storage.Weight())

    if include_nsoftmu:
        hists["nsoftmu"] = Hist.Hist(*n_axes, Hist.storage.Weight())

    if include_discriminators:
        hists = hists | _get_discriminators(axes, **kwargs)

    if include_common_definitions:
        hists = hists | _get_common_definitions(axes, **kwargs)

    if include_sv_variables:
        hists = hists | _get_sv_variables(axes, **kwargs)

    return hists


def _get_common_definitions(axes, **kwargs):

    include_osss = kwargs.get("include_osss", False)
    jet_fields = kwargs.get("jet_fields", None)

    if jet_fields is None:
        raise ValueError(
            "Tried to get btag input histograms without passing jet_fields (list of keys in the jet collection)"
        )

    hists = {}

    definitions = get_definitions()

    for d in definitions:
        if d not in jet_fields:
            continue

        ranges = definitions[d]["manual_ranges"]
        binning = definitions[d]["bins"]
        labels = (
            definitions[d]["displayname"]
            + " ["
            + definitions[d]["inputVar_units"]
            + "]"
            if definitions[d]["inputVar_units"] is not None
            else definitions[d]["displayname"]
        )

        if include_osss:
            ax = [
                axes["syst"],
                axes["flav"],
                axes["osss"],
                Hist.axis.Regular(binning, ranges[0], ranges[1], name=d, label=labels),
            ]
        else:
            ax = [
                axes["syst"],
                axes["flav"],
                Hist.axis.Regular(binning, ranges[0], ranges[1], name=d, label=labels),
            ]

        hists[d] = Hist.Hist(*ax, Hist.storage.Weight())

    return hists


# NOTE: This was commented out in the old histogrammer
# Disabled by default in get_histograms()
def _get_sv_variables(axes, **kwargs):

    include_osss = kwargs.get("include_osss", False)

    hists = {}

    definitions = hist_helpers.get_definitions(include_defitions=["SV"])

    for d in definitions:
        ranges = definitions[d]["manual_ranges"]
        binning = definitions[d]["bins"]
        labels = (
            definitions[d]["displayname"]
            + " ["
            + definitions[d]["inputVar_units"]
            + "]"
            if definitions[d]["inputVar_units"] is not None
            else definitions[d]["displayname"]
        )

        if include_osss:
            ax = [
                axes["syst"],
                axes["flav"],
                axes["osss"],
                Hist.axis.Regular(binning, ranges[0], ranges[1], name=d, label=labels),
            ]
        else:
            ax = [
                axes["syst"],
                axes["flav"],
                Hist.axis.Regular(binning, ranges[0], ranges[1], name=d, label=labels),
            ]

        hists[d] = Hist.Hist(*ax, Hist.storage.Weight())

    return hists


def _get_discriminators(axes, **kwargs):
    jet_fields = kwargs.get("jet_fields", None)

    if jet_fields is None:
        raise ValueError(
            "Tried to get discriminator histograms without passing jet_fields (list of keys in the jet collection)"
        )

    include_osss = kwargs.get("include_osss", False)
    njet = kwargs.get("njet", 1)
    c_wf = kwargs.get("c_wf", False)

    disc_list = get_discriminators()

    hists = {}

    common_axes = [axes["syst"], axes["flav"]]
    if include_osss:
        common_axes.append(axes["osss"])

    for d in disc_list:
        if d not in jet_fields:
            continue

        disc_axes = {
            "btag": Hist.axis.Regular(50, 0.0, 1, name="discr", label=d),
            "Bprob": Hist.axis.Regular(50, 0, 10, name="discr", label=d),
            "Res": Hist.axis.Regular(40, 0, 1, name="discr", label=d),
            "Corr": Hist.axis.Regular(40, 0, 2, name="discr", label=d),
            "2Dbin": Hist.axis.IntCategory(
                # phase4 categories (SFbc-2D/OptimizedWP) use dense codes
                # 0..10; legacy taggers use the sparse 40s/50s codes. Select
                # by branch name so phase4 values aren't silently dropped
                # into the (unlisted) overflow of the legacy axis.
                (
                    list(range(11))
                    if "_phase4" in d
                    else [0, 40, 41, 42, 43, 44, 50, 51, 52, 53, 54]
                ),
                name="discr",
                label=d,
            ),
        }

        if c_wf:
            if ("btag" in d and "2Dbin" not in d) or "ProbaN" == d:
                hists[d] = Hist.Hist(
                    *common_axes,
                    disc_axes["btag"],
                    Hist.storage.Weight(),
                )
            elif "2Dbin" in d:
                hists[d] = Hist.Hist(
                    *common_axes,
                    disc_axes["2Dbin"],
                    Hist.storage.Weight(),
                )
            elif "Bprob" in d:
                hists[d] = Hist.Hist(
                    *common_axes,
                    disc_axes["Bprob"],
                    Hist.storage.Weight(),
                )
            elif "PNetRegPtRawRes" == d:
                hists[d] = Hist.Hist(
                    *common_axes,
                    disc_axes["PNetRegPtRawRes"],
                    Hist.storage.Weight(),
                )
            elif "PNetRegPtRawCorr" in d:
                hists[d] = Hist.Hist(
                    *common_axes,
                    disc_axes["PNetRegPtRawCorr"],
                    Hist.storage.Weight(),
                )

        for i in range(njet):
            if ("btag" in d and "2Dbin" not in d) or "ProbaN" == d:
                hists[f"{d}_{i}"] = Hist.Hist(
                    *common_axes,
                    disc_axes["btag"],
                    Hist.storage.Weight(),
                )
            elif "2Dbin" in d:
                hists[f"{d}_{i}"] = Hist.Hist(
                    *common_axes,
                    disc_axes["2Dbin"],
                    Hist.storage.Weight(),
                )
            elif "Bprob" in d:
                hists[f"{d}_{i}"] = Hist.Hist(
                    *common_axes,
                    disc_axes["Bprob"],
                    Hist.storage.Weight(),
                )

            if include_osss:
                if "Res" in d:
                    hists[f"{d}_{i}"] = Hist.Hist(
                        *common_axes,
                        disc_axes["Res"],
                        Hist.storage.Weight(),
                    )
                elif "Corr" in d:
                    hists[f"{d}_{i}"] = Hist.Hist(
                        *common_axes,
                        disc_axes["Corr"],
                        Hist.storage.Weight(),
                    )

            else:
                if "PNetRegPtRawRes" == d:
                    hists[f"{d}_{i}"] = Hist.Hist(
                        *common_axes,
                        disc_axes["Res"],
                        Hist.storage.Weight(),
                    )
                elif "PNetRegPtRawCorr" in d:
                    hists[f"{d}_{i}"] = Hist.Hist(
                        *common_axes,
                        disc_axes["Corr"],
                        Hist.storage.Weight(),
                    )

    # Tagger data/MC comparison (runner --tagger-compare): every stored score
    # matching _LOGIT_RE on a fine logit axis, so tails can be rebinned offline;
    # nominal only (histo_writer skips "_logit_" for shifts).
    if kwargs.get("tagger_compare", False):
        for d in jet_fields:
            if _LOGIT_RE.fullmatch(d):
                for i in range(njet):
                    hists[f"{d}_logit_{i}"] = Hist.Hist(
                        *common_axes,
                        Hist.axis.Regular(
                            600, -14, 16, name="discr", label=f"logit({d})"
                        ),
                        Hist.storage.Weight(),
                    )
        for t1, t2, d in (
            ("UParTAK4", "UParTAK4NoAdv", "HFvLF"),
            ("UParTAK4", "UParTAK4NoAdv", "BvC"),
            ("UParTAK4", "PNet", "HFvLF"),
            ("UParTAK4", "PNet", "BvC"),
            ("UParTAK4", "DeepFlav", "HFvLF"),
            ("UParTAK4", "DeepFlav", "BvC"),
            ("UParTAK4", "DeepFlav", "B"),
        ):
            if {f"btag{t1}{d}", f"btag{t2}{d}"} <= set(jet_fields):
                for i in range(njet):
                    hists[f"btagVs_{t1}_{t2}_{d}_logit_{i}"] = Hist.Hist(
                        *common_axes,
                        Hist.axis.Regular(
                            52, -12, 14, name="t1", label=f"logit({t1} {d})"
                        ),
                        Hist.axis.Regular(
                            52, -12, 14, name="t2", label=f"logit({t2} {d})"
                        ),
                        Hist.storage.Weight(),
                    )

    return hists


_LOGIT_RE = re.compile(
    r"btag(UParTAK4|UParTAK4NoAdv|PNet|DeepFlav)(HFvLF|BvC)(_pt\d+to(\d+|inf))?"
    r"|btag(UParTAK4|UParTAK4NoAdv|PNet)(B|CvL|CvB|CvNotB|QvG)"
    r"|btagDeepFlav(B|CvL|CvB|QG)"
)
