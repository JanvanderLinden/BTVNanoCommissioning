import awkward as ak
import hist
import numpy as np
from coffea import processor

import collections


from pathlib import Path

# Campaigns that aren't backed by their own CVMFS metadata directory and should
# instead resolve to another campaign's corrections (e.g. a private NanoAOD
# production reusing the same MiniAOD conditions as an official campaign).
_CAMPAIGN_ALIASES = {
    "CustomSummer24": "Summer24",
    "CustomSummer24UParTPaper": "Summer24",
    "CustomSummer25": "Prompt25",
}


def campaign_map():
    dirs = [
        Path("/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/"),
        Path("/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/"),
        Path("/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/"),
        Path("/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/"),
        Path("/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/"),
    ]
    subdirs = [p.name for d in dirs if d.is_dir() for p in d.iterdir() if p.is_dir()]
    dirnames = {}
    for i in range(len(subdirs)):
        if "Run3" in subdirs[i]:
            if "25Prompt" in subdirs[i]:
                dirnames["Prompt25"] = subdirs[i]
            elif "26Prompt" in subdirs[i]:
                dirnames["Prompt26"] = subdirs[i]
            else:
                dirnames[subdirs[i].split("-")[2]] = subdirs[i]
        elif "Run2" in subdirs[i]:
            key = subdirs[i].split("-")[1] + "-UL"
            dirnames[key] = subdirs[i]
        elif "Run3" not in subdirs[i] and "Run2" not in subdirs[i]:
            continue
        else:
            raise ValueError("Unknown campaign name")
    for alias, target in _CAMPAIGN_ALIASES.items():
        if target in dirnames:
            dirnames[alias] = dirnames[target]
    return dirnames


def flatten(ar):  # flatten awkward into a 1d array to hist
    return ak.flatten(ar, axis=None)


def normalize(val, cut):
    if cut is None:
        ar = ak.to_numpy(ak.fill_none(val, np.nan))
        return ar
    else:
        ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
        return ar


def update(events, collections):
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out


# return run & lumiblock in pairs
def dump_lumi(events, output):
    pairs = np.vstack((events.run.to_numpy(), events.luminosityBlock.to_numpy()))
    # remove replicas
    pairs = np.unique(np.transpose(pairs), axis=0)
    pairs = pairs[
        np.lexsort(([pairs[:, i] for i in range(pairs.shape[1] - 1, -1, -1)]))
    ]
    output["fname"] = processor.set_accumulator([events.metadata["filename"]])
    output["run"] = processor.column_accumulator(pairs[:, 0])
    output["lumi"] = processor.column_accumulator(pairs[:, 1])
    return output


## Based on https://github.com/CoffeaTeam/coffea/discussions/735
def _inner_type(t):
    # awkward1 wraps the inner type as `.type`; awkward2 renamed it to `.content`
    return t.content if hasattr(t, "content") else t.type


# awkward1 names the leaf type PrimitiveType; awkward2 renamed it to NumpyType
_LEAF_TYPES = tuple(
    cls
    for cls in (
        getattr(ak.types, "PrimitiveType", None),
        getattr(ak.types, "NumpyType", None),
    )
    if cls is not None
)


def _is_rootcompat(a):
    """Is it a flat or 1-d jagged array?"""
    t = ak.type(a)
    if isinstance(t, ak.types.ArrayType):
        inner = _inner_type(t)
        if isinstance(inner, _LEAF_TYPES):
            return True
        if isinstance(inner, ak.types.ListType) and isinstance(
            _inner_type(inner), _LEAF_TYPES
        ):
            return True
    return False


def PFCand_link(events, event_level, jetindx):
    if str(ak.type(jetindx)).count("*") > 1:
        jetindx = jetindx[event_level]
        spfcands = collections.defaultdict(dict)
        for i in range(len(jetindx[0])):
            spfcands[i] = events[event_level].PFCands[
                events[event_level]
                .JetPFCands[events[event_level].JetPFCands.jetIdx == jetindx[:, i]]
                .pFCandsIdx
            ]

    else:
        spfcands = events[event_level].PFCands[
            events[event_level]
            .JetPFCands[events[event_level].JetPFCands.jetIdx == jetindx[event_level]]
            .pFCandsIdx
        ]
    return spfcands


def _local_pick(arr, idx, default):
    """Select arr[i][idx[i]] for each event i (idx may be a masked/padded
    per-event scalar index), filling `default` where idx is out of range."""
    mask = ak.local_index(arr) == idx[:, None]
    return ak.fill_none(ak.firsts(arr[mask]), default)


def SV_link_ragged(events, event_level, jetindx):
    """Attach the primary secondary vertex to a ragged (variable jets/event)
    list of raw jet indices, e.g. a SelJet collection with a variable number
    of jets per event capped at some maximum. Unlike SV_link, this does not
    need a fixed per-event slot count and keeps the output jagged (no
    padding), matching the shape of `jetindx` itself.

    Uses the Jet->SV index (Jet_svIdx1) when available. Some private NanoAOD
    productions (e.g. CustomSummer24) keep the SV collection but drop that
    index; in that case, falls back to matching each jet to its closest SV by
    dR, with no cone restriction. Validated against a sample that has
    Jet_svIdx1: an initial "highest-dlenSig SV within dR < 0.4" heuristic only
    reproduced the true link for 89% of jets (capping out ~92% at any cone
    size); closest-by-dR with no cut reproduces it for 96%, because the
    actual Jet<->SV association in MiniAOD is track-based, not a local
    geometric+dlenSig cut. Either way, jets with no matched SV get -1.
    """
    jet = events[event_level].Jet
    sv = events[event_level].SV
    if "svIdx1" in jet.fields:
        # pad SV collection with a dummy entry so index 0 is always valid,
        # even for events with zero SVs (ak.where evaluates both branches
        # eagerly)
        nsv = ak.num(sv.mass, axis=1)
        sv_mass = ak.fill_none(ak.pad_none(sv.mass, 1, axis=1), -1.0)
        sv_dlensig = ak.fill_none(ak.pad_none(sv.dlenSig, 1, axis=1), -1.0)
        svidx1 = jet.svIdx1[jetindx]
        # Jet_svIdx1 can reference a secondary vertex that failed the SV
        # collection's own selection cuts and is therefore missing from `SV`
        # even though svIdx1 >= 0; guard against that out-of-range case too.
        has_sv = (svidx1 >= 0) & (svidx1 < nsv[:, None])
        safe_svidx = ak.where(has_sv, svidx1, 0)
        mass = ak.where(has_sv, sv_mass[safe_svidx], -1.0)
        dlensig = ak.where(has_sv, sv_dlensig[safe_svidx], -1.0)
        return mass, dlensig

    # No Jet->SV index in this production: match by dR instead. Both the
    # selected jets and the SVs are ragged per event, so build the full
    # per-jet x per-SV pairing explicitly. No cone cut -- see validation note
    # above.
    pair = ak.cartesian([jet[jetindx], sv], axis=1, nested=True)  # (event, jet, sv)
    # SecondaryVertex is a Vertex (position x/y/z), not a Lorentz vector -- coffea's
    # old duck-typed delta_r() tolerated it directly, the new vector package doesn't;
    # use the .p4 property coffea provides for exactly this
    dr = pair["0"].delta_r(pair["1"].p4)
    best = ak.argmin(dr, axis=-1, keepdims=True)
    mass = ak.fill_none(ak.flatten(pair["1"].mass[best], axis=-1), -1.0)
    dlensig = ak.fill_none(ak.flatten(pair["1"].dlenSig[best], axis=-1), -1.0)
    return mass, dlensig


def SV_link(events, event_level, jetindx):
    """Attach the primary secondary vertex (Jet_svIdx1, an index into the
    SV collection) to the given jet index/indices. Jets with no matched SV
    (svIdx1 == -1), or jet slots with no jet at all (padded jetindx), get -1.

    Some private NanoAOD productions (e.g. CustomSummer24) keep the SV
    collection but drop the Jet->SV index; when that index is unavailable,
    falls back to matching each jet slot to its closest SV by dR (no cone
    restriction -- see the validation note in SV_link_ragged).
    """
    jet = events[event_level].Jet
    sv = events[event_level].SV
    nsv = ak.num(sv.mass, axis=1)
    has_idx = "svIdx1" in jet.fields

    def sv_for(safe_jetidx, has_jet):
        if has_idx:
            svidx1 = _local_pick(jet.svIdx1, safe_jetidx, -1)
            # Jet_svIdx1 can reference a secondary vertex that failed the SV
            # collection's own selection cuts and is therefore missing from
            # `SV` even though svIdx1 >= 0; guard against that out-of-range
            # case too.
            has_sv = has_jet & (svidx1 >= 0) & (svidx1 < nsv)
            safe_svidx = ak.where(has_sv, svidx1, 0)
            mass = ak.where(has_sv, _local_pick(sv.mass, safe_svidx, -1.0), -1.0)
            dlensig = ak.where(has_sv, _local_pick(sv.dlenSig, safe_svidx, -1.0), -1.0)
            return mass, dlensig

        # No Jet->SV index in this production: match by dR instead. The
        # jet slot is a single jet per event here, so it broadcasts
        # directly against the per-event SV list without needing a cartesian.
        # `jet[safe_jetidx]` cannot be used directly: safe_jetidx is a flat
        # (non-jagged) per-event index, and awkward interprets a flat index
        # array indexing a jagged array as axis-0 (event) fancy indexing, not
        # a per-event local pick -- same local-index mask as `_local_pick`,
        # kept inline since the record (not a single field) is needed here.
        # No cone cut -- see the validation note in SV_link_ragged.
        sel_jet = ak.firsts(jet[ak.local_index(jet.pt) == safe_jetidx[:, None]])
        # sv (not sv.p4) is a Vertex, not a Lorentz vector -- see SV_link_ragged
        dr = sel_jet.delta_r(sv.p4)
        best = ak.argmin(dr, axis=-1, keepdims=True)
        mass = ak.fill_none(ak.flatten(sv.mass[best], axis=-1), -1.0)
        dlensig = ak.fill_none(ak.flatten(sv.dlenSig[best], axis=-1), -1.0)
        mass = ak.where(has_jet, mass, -1.0)
        dlensig = ak.where(has_jet, dlensig, -1.0)
        return mass, dlensig

    if str(ak.type(jetindx)).count("*") > 1:
        jetindx = jetindx[event_level]
        svmass = collections.defaultdict(dict)
        svd3dsig = collections.defaultdict(dict)
        for i in range(len(jetindx[0])):
            idx = jetindx[:, i]
            has_jet = ~ak.is_none(idx)
            safe_jetidx = ak.fill_none(idx, 0)
            svmass[i], svd3dsig[i] = sv_for(safe_jetidx, has_jet)
    else:
        idx = jetindx[event_level]
        has_jet = ~ak.is_none(idx)
        safe_jetidx = ak.fill_none(idx, 0)
        svmass, svd3dsig = sv_for(safe_jetidx, has_jet)
    return svmass, svd3dsig


def uproot_writeable(events, include=["events", "run", "luminosityBlock"]):
    ev = {}
    include = np.array(include)
    no_filter = False
    if len(include) == 1 and include[0] == "*":
        no_filter = False
    for bname in events.fields:
        # Check relevance against `include` from the bname string alone, before
        # touching events[bname] at all: accessing any nested/record field
        # triggers awkward2's eager __awkward_validation__, so an irrelevant
        # collection elsewhere in the file that's malformed for this
        # production (e.g. TrigObj on some private NanoAODs) would otherwise
        # crash the write even though it was never going to be included.
        if (
            not no_filter
            and bname not in include
            and all(np.char.startswith(include, bname) == False)
        ):
            continue
        if not events[bname].fields:
            if not no_filter and bname not in include:
                continue
            ev[bname] = ak.fill_none(
                ak.to_packed(ak.without_parameters(events[bname])), -99
            )
        else:
            no_filter_nest = False
            if all(np.char.startswith(include, bname) == False):
                continue
            include_nest = [
                i[i.find(bname) + len(bname) + 1 :]
                for i in include
                if i.startswith(bname)
            ]

            if len(include_nest) == 1 and include_nest[0] == "*":
                no_filter_nest = True

            if not no_filter_nest:
                mask_wildcard = np.char.find(include_nest, "*") != -1
                include_nest = np.char.replace(include_nest, "*", "")
            for n in events[bname].fields:
                ## make selections to the filter case, keep cross-ref ("Idx")
                if (
                    not no_filter_nest
                    and all(np.char.find(n, include_nest) == -1)
                    and "Idx" not in n
                    and "Flavor" not in n
                ):
                    continue
                evnums = ak.num(events[bname][n], axis=0)
                # awkward1 returns a plain int here; awkward2 returns a 0-d
                # numpy.ndarray of the same count -- check dimensionality, not
                # type, so this isn't awkward-version-sensitive
                if np.ndim(evnums) != 0:
                    continue
                if not _is_rootcompat(events[bname][n]) and evnums != len(
                    flatten(events[bname][n])
                ):
                    continue
                # skip IdxG
                if "IdxG" in n:
                    continue
                # write flat "<bname>_<n>" keys directly, matching BTA_producer's
                # own convention and uproot4's old auto-flattened schema -- a
                # nested ev[bname] = ak.zip(...) record would make uproot5 write
                # genuinely nested dotted branches ("SelJet.pt") instead of the
                # flat ones downstream consumers (SFbc-2D) expect ("SelJet_pt")
                ev[f"{bname}_{n}"] = ak.fill_none(
                    ak.to_packed(ak.without_parameters(events[bname][n])), -99
                )
    return ev


def _stack_syst(hs):
    """One histogram from per-shift histograms whose "syst" categories are
    disjoint: the categories in the order sequential accumulation gives, the
    contents copied once. None where this does not apply."""
    h0 = hs[0]
    if not all(isinstance(h, hist.Hist) for h in hs) or "syst" not in h0.axes.name:
        return None
    i = h0.axes.name.index("syst")
    ax0 = h0.axes[i]
    if not (
        isinstance(ax0, hist.axis.StrCategory)
        and ax0.traits.growth
        and not ax0.traits.overflow
    ):
        return None
    cats = []
    for h in hs:
        if h.axes.name != h0.axes.name or h.storage_type != h0.storage_type:
            return None
        if any(h.axes[j] != h0.axes[j] for j in range(len(h0.axes)) if j != i):
            return None
        cats.extend(h.axes[i])
    if len(set(cats)) != len(cats):
        return None
    axes = list(h0.axes)
    axes[i] = hist.axis.StrCategory(cats, growth=True)
    axes[i].__dict__.update(ax0.__dict__)  # name and label exactly as stored
    out = hist.Hist(*axes, storage=h0.storage_type(), name=h0.name, label=h0.label)
    view, off = out.view(flow=True), 0
    for h in hs:
        n = len(h.axes[i])
        if n:
            sl = [slice(None)] * view.ndim
            sl[i] = slice(off, off + n)
            view[tuple(sl)] = h.view(flow=True)
        off += n
    return out


def _merge_shifts(items):
    if all(isinstance(x, dict) for x in items):
        keys = list(dict.fromkeys(k for x in items for k in x))
        return {k: _merge_shifts([x[k] for x in items if k in x]) for k in keys}
    if len(items) > 1:
        stacked = _stack_syst(items)
        if stacked is not None:
            return stacked
    return processor.accumulate(items)


def accumulate_shifts(outputs):
    """processor.accumulate for the per-shift outputs of one chunk. Each shift
    fills its own "syst" categories, so a histogram is built once from all
    shifts instead of by ~30 successive additions (each a copy). Same result."""
    outputs = [o for o in outputs if o is not None]
    if len(outputs) < 3:
        return processor.accumulate(outputs)
    return _merge_shifts(outputs)
