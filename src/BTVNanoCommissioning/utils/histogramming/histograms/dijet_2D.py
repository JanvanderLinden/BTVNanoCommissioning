import itertools
import awkward as ak
import numpy as np
import hist as Hist

from BTVNanoCommissioning.helpers.func import flatten

# Legacy 2D category codes (utils/selection.py _mapping). The tag-side
# requirement is always expressed in the legacy scheme: the phase4 scheme has
# no calibrated b/c working points to cut on.
_LEGACY_CODES = [0, 40, 41, 42, 43, 44, 50, 51, 52, 53, 54]

# Tag-jet class, as stored in the "tag_cls_<tagger>" branches by the workflow.
TAG_CLASSES = {"tagl": 0, "tagc": 1, "tagb": 2}

# Taggers the away jet is classified with, short name -> tagger name (the
# btag<name>... Jet branch prefix, and what btag_wp() takes). Order is priority
# order: the first available one is the primary, i.e. the tagger the tag-side
# event selection itself cuts on.
TAG_TAGGERS = {"nopid": "NoPIDParTAK4", "upart": "UParTAK4"}


def available_tag_taggers(jet_fields):
    """Short names of the taggers that can classify the away jet, in priority
    order. NoPIDParTAK4 only exists in the private CustomSummer24 tuples, so
    central NanoAOD gets a UParTAK4-only split.
    """
    return [s for s, full in TAG_TAGGERS.items() if f"btag{full}2Dbin" in jet_fields]


def get_histograms(axes, **kwargs):
    """Dijet tag-and-probe control distributions: the variables the region is
    defined by (pT average, third-jet activity, balance, back-to-backness) plus
    the tag jet itself, which drives the probe-side flavour composition.
    """
    hists = {}

    hists["ptave"] = Hist.Hist(axes["syst"], axes["ljpt"], Hist.storage.Weight())
    hists["alpha"] = Hist.Hist(
        axes["syst"],
        Hist.axis.Regular(30, 0.0, 0.3, name="alpha", label=r"$p_{T,3}/p_{T,ave}$"),
        Hist.storage.Weight(),
    )
    hists["asym"] = Hist.Hist(
        axes["syst"],
        Hist.axis.Regular(25, 0.0, 0.5, name="asym", label="dijet asymmetry"),
        Hist.storage.Weight(),
    )
    hists["dphi_jj"] = Hist.Hist(
        axes["syst"],
        Hist.axis.Regular(32, 2.4, 3.2, name="dphi", label=r"$\Delta\phi(j_1,j_2)$"),
        Hist.storage.Weight(),
    )
    hists["tag_pt"] = Hist.Hist(
        axes["syst"], axes["flav"], axes["ljpt"], Hist.storage.Weight()
    )
    # Control plot only, and only for the primary tagger -- the away jet's
    # category in the tagger the selection cut on. The per-tagger categories
    # themselves reach the fit through the tag-class-split templates below.
    hists["tag_2Dbin"] = Hist.Hist(
        axes["syst"],
        axes["flav"],
        Hist.axis.IntCategory(_LEGACY_CODES, name="discr", label="tag jet 2D category"),
        Hist.storage.Weight(),
    )
    # One-bin normalisation per (tagger, tag class): the away-tag
    # normalisation spread (27 % on the same events, plan section 9c) is the
    # dijet's headline anomaly, so re-checking it after any change should be a
    # constant-time read, not an integral over kinematic histograms.
    hists["norm_tagcls"] = Hist.Hist(
        axes["syst"],
        Hist.axis.StrCategory([], name="tagger", growth=True),
        Hist.axis.IntCategory(
            list(TAG_CLASSES.values()), name="tagcls", label="tag-jet class"
        ),
        Hist.storage.Weight(),
    )

    return hists


def _is_split_template(name):
    return name.startswith("sf2d_") or (name.startswith("btag") and "2Dbin" in name)


def book_tag_split(output, taggers):
    """Clone every probe-side SF template once per (tag tagger, tag-jet class).

    The fit region is (probe category | tag category), so the templates the SF
    fit consumes have to be split by what the away jet was tagged as. Cloning
    the already-booked histograms keeps the axis layout and the variable names
    that SFbc-2D/scripts/makeTemplates.py expects, with only a
    `_<tagger>_<tagl|tagc|tagb>` suffix added, instead of introducing a
    category axis that the downstream tooling would have to learn about.

    Called at booking time, so the clones are empty.
    """
    extra = {}
    for name, h in output.items():
        if not _is_split_template(name):
            continue
        for tagger in taggers:
            for cls in TAG_CLASSES:
                extra[f"{name}_{tagger}_{cls}"] = h.copy()
    output.update(extra)
    return output


def _split_targets(output, taggers, seljet_fields):
    """Resolve every (tagger, class) clone set to what fills it, once.

    A clone's name is its base template name plus `_<tagger>_<class>`, so the
    base is recovered by stripping that suffix -- and, for the per-jet-index
    discriminator templates, the trailing `_0` on top of it. Resolved here
    rather than inside the systematic loop so the string matching runs once per
    call instead of once per systematic per class.

    Returns (tagger, class) -> (sf2d templates keyed by base name, list of
    (histogram, SelJet field) for the discriminator templates).
    """
    targets = {}
    for tagger in taggers:
        for cls in TAG_CLASSES:
            suffix = f"_{tagger}_{cls}"
            sf2d_view, discr = {}, []
            for name, h in output.items():
                if not name.endswith(suffix):
                    continue
                base = name[: -len(suffix)]
                if base.startswith("sf2d_"):
                    sf2d_view[base] = h
                    continue
                field = base[:-2] if base.endswith("_0") else base
                if field in seljet_fields:
                    discr.append((h, field))
            targets[(tagger, cls)] = (sf2d_view, discr)
    return targets


def _genflavour(jets):
    if "hadronFlavour" not in jets.fields:
        return ak.zeros_like(jets.pt, dtype=int)
    return ak.values_astype(
        jets.hadronFlavour
        + 1 * ((jets.partonFlavour == 0) & (jets.hadronFlavour == 0)),
        int,
    )


def dijet_writer(pruned_ev, output, weights, systematics, isSyst, taggers):
    """Fill the control distributions above and the tag-class-split templates
    booked by book_tag_split(). Kept out of histo_writer because neither the
    dijet topology variables nor the tag-class split exist in any other channel.
    """
    # local import: histogrammer imports this module (via hist_helpers), so a
    # module-level import here would be circular
    from BTVNanoCommissioning.utils.histogramming.histogrammer import (
        _fill_sf2d_histograms,
        replay_fills,
    )

    seljet = pruned_ev.SelJet
    genflavor = _genflavour(seljet)
    tagflav = _genflavour(pruned_ev.TagJet)
    class_masks = {
        (tagger, cls): ak.to_numpy(pruned_ev[f"tag_cls_{tagger}"]) == code
        for tagger in taggers
        for cls, code in TAG_CLASSES.items()
    }
    # the primary tagger is the one the tag-side selection cut on, so it is the
    # one the inclusive tag-category control plot is meaningful for
    tag_2Dbin = pruned_ev[f"tag_2Dbin_{taggers[0]}"]

    def fill_all(output, weight, syst):
        targets = _split_targets(output, taggers, seljet.fields)
        syst_arr = np.full(len(weight), syst)

        output["ptave"].fill(syst_arr, flatten(pruned_ev.ptave), weight=weight)
        output["alpha"].fill(syst_arr, flatten(pruned_ev.alpha), weight=weight)
        output["asym"].fill(syst_arr, flatten(pruned_ev.asym), weight=weight)
        output["dphi_jj"].fill(syst_arr, flatten(pruned_ev.dphi_jj), weight=weight)
        output["tag_pt"].fill(
            syst_arr, tagflav, flatten(pruned_ev.TagJet.pt), weight=weight
        )
        output["tag_2Dbin"].fill(syst_arr, tagflav, flatten(tag_2Dbin), weight=weight)

        for key, mask in class_masks.items():
            if not mask.any():
                continue
            tagger_name, cls_name = key
            wgt = weight[mask]
            syst_cls = np.full(len(wgt), syst)
            output["norm_tagcls"].fill(
                syst=syst_cls,
                tagger=np.full(len(wgt), tagger_name),
                tagcls=np.full(len(wgt), TAG_CLASSES[cls_name]),
                weight=wgt,
            )
            sf2d_view, discr = targets[key]
            for h, field in discr:
                h.fill(
                    syst=syst_cls,
                    flav=genflavor[mask],
                    discr=seljet[field][mask],
                    weight=wgt,
                )
            if sf2d_view:
                _fill_sf2d_histograms(
                    sf2d_view, seljet[mask], genflavor[mask], wgt, syst_cls
                )

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
    replay_fills(output, fill_all, systs, weight_for, len(weights.weight()))
    return output
