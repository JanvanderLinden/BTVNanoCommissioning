import numpy as np

np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))
import fnmatch
import argparse, os, glob, sys
import matplotlib.pyplot as plt
import mplhep as hep
import hist
from coffea.util import load
from matplotlib.offsetbox import AnchoredText

plt.style.use(hep.style.CMS)
fs = 25
plt.rcParams.update({"font.size": fs})

from BTVNanoCommissioning.workflows import workflows
from BTVNanoCommissioning.helpers.xs_scaler import collate, scaleSumW
from BTVNanoCommissioning.helpers.definitions import (
    get_definitions,
    axes_name,
)
from BTVNanoCommissioning.utils.plot_utils import (
    plotratio,
    SFerror,
    MCerrorband,
    poisson_interval,
    autoranger,
    rebin_hist,
    sample_mergemap,
    color_map,
)
from BTVNanoCommissioning.helpers.stag_helpers import STAG_FLAVOURS


def flav_components(h, sel, flav_set, hf_split=False):
    """[(flavour label, histogram)] of the MC flavour split. Histograms with the
    s-tagging flavour axis (stag_* workflows: b, bb, c, cc, s, ud, g, PU) are shown as
    b (b+bb), c (c+cc), s, ud, g, PU, or with bb and cc separately (hf_split); the
    other histograms are split by flav_set, one entry per flav-axis bin."""
    sel = dict(sel)
    if len(h.axes["flav"]) == len(STAG_FLAVOURS):
        # bottom of the stack first: from the top s, c (cc), b (bb), g, ud, PU
        groups = [("PU", [7]), ("ud", [5]), ("g", [6])]
        if hf_split:
            groups += [("bb", [1]), ("b", [0]), ("cc", [3]), ("c", [2])]
        else:
            groups += [("b", [0, 1]), ("c", [2, 3])]
        groups += [("s", [4])]
    else:
        groups = [(t, [i]) for i, t in enumerate(flav_set)]
    out = []
    for label, indices in groups:
        comp = None
        for i in indices:
            sel["flav"] = i
            comp = h[sel] if comp is None else comp + h[sel]
        out.append((label, comp))
    return out


# tt1L-s control histograms: flavour class of the sJet (stag_helpers.flav_class:
# 0 = g/unmatched, 1 = ud, 3 = s, 4 = c, 5 = b) -> s-tagging flavour index (STAG_FLAVOURS)
JFLAV_TO_STAGFLAV = {5: 0, 4: 2, 3: 4, 1: 5, 0: 6}


def jflav_to_stagflav(h):
    """Histogram with the tt1L-s `jflav` axis -> the same histogram with the s-tagging
    flavour axis `flav` (b, bb, c, cc, s, ud, g, PU), so that the flavour split of the
    stag_* workflows applies (g contains the unmatched jets)."""
    iax = h.axes.name.index("jflav")
    new_ax = hist.axis.IntCategory(list(range(len(STAG_FLAVOURS))), name="flav",
                                   label="Jet flavour (" + ", ".join(STAG_FLAVOURS) + ")")
    axes = list(h.axes)
    axes[iax] = new_ax
    hn = hist.Hist(*axes, storage=h.storage_type())
    old, new = h.view(flow=True), hn.view(flow=True)
    for i, val in enumerate(h.axes["jflav"]):
        if val not in JFLAV_TO_STAGFLAV:
            continue
        so = [slice(None)] * h.ndim
        sn = [slice(None)] * h.ndim
        so[iax], sn[iax] = i, JFLAV_TO_STAGFLAV[val]
        new[tuple(sn)] = old[tuple(so)]
    return hn


def _flow_layout(h1d, arr_flow, flow):
    """1D array with all flow bins of the 1D histogram h1d -> the bins as plotted with
    `flow` (none: inner bins, sum: flow added to the first/last bin, show: with flow bins)"""
    ax = h1d.axes[0]
    lo = 1 if ax.traits.underflow else 0
    hi = 1 if ax.traits.overflow else 0
    inner = np.array(arr_flow[lo : len(arr_flow) - hi], dtype=float)
    if flow == "sum":
        if lo:
            inner[0] += arr_flow[0]
        if hi:
            inner[-1] += arr_flow[-1]
        return inner
    if flow == "show":
        under = arr_flow[:1] if lo else np.zeros(1)
        over = arr_flow[-1:] if hi else np.zeros(1)
        return np.concatenate([under, inner, over])
    return inner


def syst_band(h, sel, nominal, flow, exclude=()):
    """Total systematic uncertainty of the MC sum: per variation pair <name>Up/<name>Down
    of the `syst` axis the larger upward and downward deviation from `nominal` per bin,
    added in quadrature. Returns (down, up, [names]) in the bin layout of `flow`."""
    names = [str(s) for s in h.axes["syst"]]
    pairs = sorted(
        n[:-2] for n in names
        if n.endswith("Up") and f"{n[:-2]}Down" in names
        and not any(fnmatch.fnmatch(n[:-2], p) for p in exclude)
    )

    def vals(label):
        h1 = h[{**sel, "syst": label}]
        return _flow_layout(h1, h1.view(flow=True)["value"], flow)

    nom = vals(nominal)
    up2 = np.zeros_like(nom)
    dn2 = np.zeros_like(nom)
    for p in pairs:
        u, d = vals(f"{p}Up") - nom, vals(f"{p}Down") - nom
        up2 += np.maximum(np.maximum(u, d), 0.0) ** 2
        dn2 += np.maximum(np.maximum(-u, -d), 0.0) ** 2
    return np.sqrt(dn2), np.sqrt(up2), pairs


# uncertainty bands: MC stat. filled grey, stat. + syst. hatched black
STAT_BAND = {"facecolor": "#9494AE", "alpha": 0.45, "lw": 0, "fill": True}
TOTAL_BAND = {"facecolor": "none", "edgecolor": "black", "hatch": "////", "lw": 0,
              "fill": True}


def _view(h):
    v = h.view(flow=True)
    if v.dtype.names and "variance" in v.dtype.names:
        return np.asarray(v["value"]), np.asarray(v["variance"])
    return np.asarray(v), np.asarray(v)


def bands_and_ratio(ax, rax, hmc, hdata, flow, ms, err_syst=None):
    """MC stat. band (and stat. + syst. band, err_syst = (down, up)) in the main and the
    ratio panel, and the Data/MC points with x errors (marker size ms)"""
    val, var = _view(hmc)
    nom = _flow_layout(hmc, val, flow)
    stat = np.sqrt(np.abs(_flow_layout(hmc, var, flow)))
    edges = plot_edges(hmc, flow)
    with np.errstate(divide="ignore", invalid="ignore"):
        safe = np.where(nom != 0, nom, np.nan)
        ax.stairs(nom + stat, edges=edges, baseline=nom - stat, label="Stat. unc.",
                  **STAT_BAND)
        if rax is not None:
            rax.stairs(1 + stat / safe, edges=edges, baseline=1 - stat / safe, **STAT_BAND)
        if err_syst is not None:
            dn, up = np.hypot(err_syst[0], stat), np.hypot(err_syst[1], stat)
            ax.stairs(nom + up, edges=edges, baseline=nom - dn,
                      label=r"Stat. $\oplus$ syst. unc.", **TOTAL_BAND)
            if rax is not None:
                rax.stairs(1 + up / safe, edges=edges, baseline=1 - dn / safe, **TOTAL_BAND)
        if rax is None or hdata is None:
            return
        rax.axhline(1.0, ls="--", color="black", lw=1, alpha=0.5)
        dval, dvar = _view(hdata)
        d = _flow_layout(hdata, dval, flow)
        dv = _flow_layout(hdata, dvar, flow)
        r = d / safe
        err = np.abs(poisson_interval(r, dv / safe**2) - r)
        centers = 0.5 * (edges[1:] + edges[:-1])
        half = 0.5 * (edges[1:] - edges[:-1])
        rax.errorbar(centers, r, xerr=half, yerr=err, color="black", marker="o",
                     markersize=ms, linestyle="none", lw=1)


# W+c: charge combinations of the OS/SS axis (osss = 1: OS, -1: SS)
OSSS_MODES = {
    "os-ss": ("OSminusSS", "OS-SS"),
    "os+ss": ("OSplusSS", "OS+SS"),
    "os": ("OS", "OS"),
    "ss": ("SS", "SS"),
}


def osss_reduce(h, mode):
    if mode == "os":
        return h[{"osss": hist.loc(1)}]
    if mode == "ss":
        return h[{"osss": hist.loc(-1)}]
    if mode == "os+ss":
        return h[{"osss": sum}]
    return h[{"osss": hist.loc(1)}] + h[{"osss": hist.loc(-1)}] * -1


def report_variations(h, sel, nominal, names, discr, threshold=0.2):
    """relative change of the MC sum per variation; prints the variations changing it by
    more than `threshold` (broken variations, e.g. a missing weight or a wrong sum of
    weights of the renormalisation)"""
    nom = h[{**sel, "syst": nominal}].sum(flow=True)
    nom = nom.value if hasattr(nom, "value") else float(nom)
    if not nom:
        return
    large = []
    for n in names:
        for d in ("Up", "Down"):
            v = h[{**sel, "syst": f"{n}{d}"}].sum(flow=True)
            v = v.value if hasattr(v, "value") else float(v)
            if abs(v / nom - 1) > threshold:
                large.append(f"{n}{d} {100 * (v / nom - 1):+.0f}%")
    if large:
        print(f"WARNING {discr}: variations changing the MC sum by more than "
              f"{100 * threshold:.0f}%: " + ", ".join(large))


def plot_edges(h1d, flow):
    edges = h1d.axes[0].edges
    if flow == "show":
        w0, w1 = edges[1] - edges[0], edges[-1] - edges[-2]
        edges = np.array([edges[0] - 2 * w0, *edges, edges[-1] + 2 * w1])
    return edges

bininfo = get_definitions()
SV_bininfo = get_definitions(include_definitions=["SV"])


def get_parser():
    parser = argparse.ArgumentParser(description="Hist plotter for commissioning")
    parser.add_argument("--lumi", required=True, type=float, help="Luminosity in /pb")
    parser.add_argument("--com", default="13.6", type=str, help="Sqrt(s) in TeV")
    parser.add_argument(
        "-p",
        "--phase",
        required=True,
        choices=list(workflows.keys()),
        dest="phase",
        help="Which phase space",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="",
        help="Input coffea files (str), split different files with ','. Wildcard option * available as well.",
    )
    parser.add_argument("--log", "--logy", action="store_true", help="Log on y axis")
    parser.add_argument("--logx", action="store_true", help="Log on x axis")
    parser.add_argument(
        "--norm",
        default=False,
        type=bool,
        help="Use for reshape SF, scale to same yield as no SFs case",
    )
    parser.add_argument(
        "-v",
        "--variable",
        type=str,
        default="all",
        help="Variables to plot, split by ,. Wildcard option * available as well. Specifying `all` will run through all variables.",
    )
    parser.add_argument(
        "--xlabel",
        type=str,
        help="Rename the label of variables to plot, split by ,.  Wildcard option * NOT available here",
    )
    parser.add_argument(
        "--ylabel", type=str, default=None, help="Modify y-axis label of plot"
    )
    parser.add_argument(
        "--SF", action="store_true", default=False, help="Make with SF comparisons"
    )
    parser.add_argument(
        "--xrange",
        type=str,
        default=None,
        help="Custom x-range. Usage: --xrange xmin,xmax",
    )
    parser.add_argument(
        "--ratio-range",
        type=str,
        default="0.5,1.5",
        help="y-range for ratio plot, default is 0.5 to 1.5. Usage: --ratio-range ymin,ymax",
    )
    parser.add_argument("--ext", type=str, default="", help="Prefix name")
    parser.add_argument(
        "--hf-split",
        action="store_true",
        help="s-tagging flavours (stag_* workflows): show bb and cc separately from b and c",
    )
    parser.add_argument(
        "--mc-only",
        action="store_true",
        help="MC only: the MC stack with its stat. uncertainty, without data and ratio panel "
        "(used automatically when the inputs contain no data)",
    )
    parser.add_argument(
        "--autorebin",
        default=None,
        help="Rebin the plotting variables, input `int` or `list`. int: merge N bins. list of number: rebin edges(non-uniform bin is possible)",
    )
    parser.add_argument(
        "--flow",
        type=str,
        default="none",
        help="str, optional {'none', 'show', 'sum'} Whether plot the under/overflow bin. If 'show', add additional under/overflow bin. If 'sum', add the under/overflow bin content to first/last bin.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="flavor",
        choices=["flavor", "sample", "sample_flav"],
        help="Decomposition of MC samples. Default is split to jet flavor, possible to split by group of MC samples. Combination of jetflavor+ sample split is also possible",
    )
    parser.add_argument(
        "--osss",
        type=str,
        default="os-ss,os+ss,os,ss",
        help="Histograms with an OS/SS axis (W+c): charge combinations to plot, one plot "
        "each, split by ',': os-ss, os+ss, os, ss. Default: all four",
    )
    parser.add_argument(
        "--splitOSSS",
        type=int,
        default=None,
        choices=[1, -1],
        help="Only for W+c phase space: only opposite sign (1) or only same sign (-1) "
        "events (replaces --osss)",
    )
    parser.add_argument(
        "--save-as",
        type=str,
        default="png,pdf",
        help="File format to save the plots, split by ,. Default is png and pdf",
    )
    parser.add_argument(
        "--CMS-label",
        type=str,
        default="Preliminary",
        help="CMS label to put on the plot, default is Preliminary",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=None,
        help="Output directory; the plots are then named <variable>[__<category>]_<split>"
        "[_mconly][_log].<ext>. Default: plot/<phase>_<ext>/unc_<variable>_inclusive_*.<ext>",
    )
    parser.add_argument(
        "--syst-band",
        action="store_true",
        help="Draw the total uncertainty (MC stat. + all <name>Up/<name>Down variations of "
        "the syst axis, added in quadrature) as a band in the main and the ratio panel",
    )
    parser.add_argument(
        "--syst-exclude",
        type=str,
        default="",
        help="Variations (without Up/Down) left out of --syst-band, split by ',', wildcards "
        "allowed, e.g. 'PDF*,UEPS_*'",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="*",
        help="Histograms with a `category` axis (selections, e.g. tt1L-s): one plot per "
        "category matching these patterns (split by ','). Default: all categories",
    )
    parser.add_argument(
        "--also-log",
        action="store_true",
        help="Save every plot a second time with log scale on the y axis (suffix _log)",
    )

    return parser.parse_args()


def main(args):
    inputs = args.input.split(",")

    # Load the inputs
    output = {}
    for inp in inputs:
        if not os.path.exists(inp) and "*" not in inp:
            print(f"{inp} does not exist!")
            return 1

        files = glob.glob(inp)

        for i in files:
            if ".coffea" not in i:
                print(f"{i} is not a valid coffea file")
                return 1
            fdict = load(i)
            for sample in fdict:
                if sample not in output:
                    output[sample] = fdict[sample]
                else:
                    for key in fdict[sample]:
                        if key not in output[sample]:
                            continue
                        output[sample][key] += fdict[sample][key]

    # variables=(): no per-variation renormalised copies (<hist>_<variation>) for outputs
    # without sumw_<variation>; the plots use the syst axis
    output = scaleSumW(output, args.lumi, variables=())

    mergemap = {}
    ## create merge map from sample set/data MC
    outdir = args.outdir if args.outdir else f"plot/{args.phase}_{args.ext}"
    os.makedirs(outdir, exist_ok=True)
    if not any(".coffea" in o for o in output.keys()):
        if "sample" in args.split:
            mergemap = sample_mergemap
        mergemap["data"] = [m for m in output.keys() if "Run" in m]
        mergemap["mc"] = [m for m in output.keys() if "Run" not in m]
    else:
        datalist = []
        mclist = []
        for f in output.keys():
            datalist.extend([m for m in output[f].keys() if "Run" in m])
            mclist.extend([m for m in output[f].keys() if "Run" not in m])
        if "sample" in args.split:
            mergemap = sample_mergemap
        mergemap["mc"] = mclist
        mergemap["data"] = datalist

    mc_only = args.mc_only or not mergemap["data"]
    if mc_only:
        if not args.mc_only:
            print("No data in the inputs: MC-only plots")
        mergemap.pop("data", None)

    collated = collate(output, mergemap)
    collated = {
        key: value for key, value in collated.items() if isinstance(collated[key], dict)
    }
    print(collated.keys())

    # Which flavor to split, default is udsg+pu+c+b, for QG workflow, split to ud+s+c+b+g+other
    flav_set = ["udsg", "pu", "c", "b"]
    if "qg" in args.phase.lower():
        flav_set = ["ud", "s", "c", "b", "g", "other"]

    ### input text settings
    phase_label_map = {
        "Wc": "W+c",
        "DY": "DY+jets",
        "QCD": "QCD",
        "semilep": r"t$\bar{t}$ semileptonic",
        "tt1L": r"t$\bar{t}$ semileptonic",
        "dilep": r"t$\bar{t}$ dileptonic",
        "dijet": "QCD dijet",
    }
    phase_nj_map = {"semilep": 4, "dilep": 2}

    input_txt = next(
        (label for key, label in phase_label_map.items() if key in args.phase),
        "placeholder",
    )
    for key, nj_val in phase_nj_map.items():
        if key in args.phase:
            nj = nj_val
            break

    if (
        "njet" in args.variable.split(",")
        or "nbjet" in args.variable.split(",")
        or "mu" in args.variable.split(",")
    ):
        nj = 1

    if "emctag" in args.phase:
        input_txt += r" (e$\mu$)"
    elif "ectag" in args.phase or "2D_e_" in args.phase or "stag_e_" in args.phase:
        input_txt += " (e)"
    elif args.phase == "ttdilep_sf" or "emu_ttdilep" in args.phase:
        input_txt += r" (e$\mu$)"
    elif args.phase == "QCD" or "qg" in args.phase.lower():
        pass
    else:
        input_txt += r" ($\mu$)"

    if "ctag" in args.phase and "DY" not in args.phase:
        input_txt += "\nw/ soft-$\mu$"

    # Find the variables to plot
    if args.variable == "all":
        var_set = [var for var in collated["mc"].keys()]
    else:
        all_vars = collated["mc"].keys()
        var_set = [
            var
            for pattern in args.variable.split(",")
            for var in fnmatch.filter(all_vars, pattern)
        ]

    # tt1L-s: sJet flavour classes -> s-tagging flavour axis; histograms with a category
    # axis (selections) -> one entry <variable>__<category> per selected category;
    # histograms with an OS/SS axis (W+c) -> one entry <variable>__<OSminusSS|OSplusSS|OS|SS>
    # per charge combination. The expanded histograms replace the original ones.
    cat_patterns = args.categories.split(",")
    category_of = {}
    base_of = {}
    expanded_set = []
    for discr in var_set:
        hmc0 = collated["mc"].get(discr)
        if not isinstance(hmc0, hist.Hist):
            expanded_set.append(discr)
            continue
        if "jflav" in hmc0.axes.name:
            for sample in collated:
                h0 = collated[sample].get(discr)
                if isinstance(h0, hist.Hist) and "jflav" in h0.axes.name:
                    collated[sample][discr] = jflav_to_stagflav(h0)
        if "category" not in collated["mc"][discr].axes.name:
            expanded_set.append(discr)
            continue
        cats = [
            c for c in collated["mc"][discr].axes["category"]
            if any(fnmatch.fnmatch(c, p) for p in cat_patterns)
        ]
        for c in cats:
            key = f"{discr}__{c}"
            for sample in collated:
                h0 = collated[sample].get(discr)
                if isinstance(h0, hist.Hist) and "category" in h0.axes.name:
                    collated[sample][key] = h0[{"category": c}]
            category_of[key] = c
            base_of[key] = discr
            expanded_set.append(key)
        for sample in collated:
            collated[sample].pop(discr, None)

    if args.splitOSSS is not None:
        osss_modes = ["os" if args.splitOSSS == 1 else "ss"]
    else:
        osss_modes = [m for m in args.osss.split(",") if m]
        for m in osss_modes:
            if m not in OSSS_MODES:
                raise SystemExit(f"--osss: unknown {m} (choices: {', '.join(OSSS_MODES)})")
    osss_label = {}
    var_set = []
    for key in expanded_set:
        h0 = collated["mc"].get(key)
        if not isinstance(h0, hist.Hist) or "osss" not in h0.axes.name:
            var_set.append(key)
            continue
        for m in osss_modes:
            key2 = f"{key}__{OSSS_MODES[m][0]}"
            for sample in collated:
                hs = collated[sample].get(key)
                if isinstance(hs, hist.Hist) and "osss" in hs.axes.name:
                    collated[sample][key2] = osss_reduce(hs, m)
            osss_label[key2] = OSSS_MODES[m][1]
            if key in category_of:
                category_of[key2] = category_of[key]
            base_of[key2] = base_of.get(key, key)
            var_set.append(key2)
        for sample in collated:
            collated[sample].pop(key, None)
    syst_exclude = [p for p in args.syst_exclude.split(",") if p]

    for index, discr in enumerate(var_set):
        # text box of this plot: phase-space label (+ jet pT bin of pT-binned discriminants)
        var_txt = input_txt
        # Skip non-histogram objects
        try:
            if not isinstance(collated["mc"][discr], hist.hist.Hist):
                continue
        except:
            print(f"{discr} not found. Variable must be in", collated["mc"].keys())
            return 1

        # Skip missing and empty hists
        check_flow = args.flow in ("show", "sum")
        if discr not in collated["mc"].keys() or (
            not mc_only and discr not in collated["data"].keys()
        ):
            print(discr, "not in files, skipping")
            continue
        elif (collated["mc"][discr].values(flow=check_flow) == 0).all() or (
            not mc_only
            and (collated["data"][discr].values(flow=check_flow) == 0).all()
        ):
            print(discr, "is empty, skipping")
            continue

        ## additional legend info
        if "2Dbin_pt" in discr or "HFvLF_pt" in discr or "BvC_pt" in discr:
            tmp = discr.strip().split("_")[1][2:]
            tmp = tmp.split("to")
            low_edge = tmp[0]
            up_edge = tmp[1]

        ## axis info
        allaxis = {}  # (OS/SS axis: already reduced in the expansion above)

        if "flav" in collated["mc"][discr].axes.name:
            allaxis["flav"] = sum
            SF_axis = allaxis
            noSF_axis = allaxis

        if "syst" in collated["mc"][discr].axes.name:
            # Get list of available systematics
            systlist = [
                collated["mc"][discr]
                .axes[collated["mc"][discr].axes.name.index("syst")]
                .value(i)
                for i in range(
                    collated["mc"][discr]
                    .axes[collated["mc"][discr].axes.name.index("syst")]
                    .size
                )
            ]
            print(f"Available systematics: {systlist}")

            # Choose the appropriate systematic name
            if "nominal" in systlist:
                allaxis["syst"] = "nominal"
            elif "SF" in systlist:  # For ctag_Wc_sf workflow
                allaxis["syst"] = "SF"
            elif len(systlist) > 0:
                # Fallback to first available systematic
                allaxis["syst"] = systlist[0]
                print(
                    f"Warning: 'nominal' not found in systematics, using '{systlist[0]}' instead"
                )

            SF_axis = allaxis.copy()
            noSF_axis = allaxis.copy()
            if "noSF" in systlist:
                noSF_axis["syst"] = "noSF"

        ## rebin config, add xerr
        do_xerr = False
        if args.autorebin is not None:
            if args.autorebin.isdigit():
                rebin = int(args.autorebin)
            else:
                rebin = np.array([float(i) for i in args.autorebin.split(",")])
                do_xerr = True
            for s in collated.keys():
                collated[s][discr] = rebin_hist(
                    collated[s][discr], collated[s][discr].axes[-1].name, rebin
                )

        # check if plottable, skip otherwise
        hmc = collated["mc"][discr][allaxis]
        if not isinstance(hmc, hist.Hist) or len(hmc.axes) == 0:
            print(f"Skipping {discr}: no axis left to plot after the selections")
            continue
        if len(hmc.axes) > 2:
            print(f"Skipping {discr}: after selections still has {len(hmc.axes)} axes")
            continue

        ## Rescale noSF & SF to same MC yields
        if (
            "flav" in collated["mc"][discr].axes.name
            and "syst" in collated["mc"][discr].axes.name
            and args.SF
        ):
            scale_sf = np.sum(collated["mc"][discr][SF_axis].values()) / np.sum(
                collated["mc"][discr][noSF_axis].values()
            )
        else:
            scale_sf = 1.0
        if (
            "flav" in collated["mc"][discr].axes.name
            and "syst" in collated["mc"][discr].axes.name
            and args.SF
        ):
            print("============> fraction of each flavor in MC")
            print(
                "b:%.3f",
                np.sum(collated["mc"][discr][{"syst": "SF", "flav": 3}].values())
                / np.sum(collated["mc"][discr][{"syst": "SF", "flav": sum}].values()),
            )
            print(
                "c:%.3f",
                np.sum(collated["mc"][discr][{"syst": "SF", "flav": 2}].values())
                / np.sum(collated["mc"][discr][{"syst": "SF", "flav": sum}].values()),
            )
            print(
                "pu:%.3f",
                np.sum(collated["mc"][discr][{"syst": "SF", "flav": 1}].values())
                / np.sum(collated["mc"][discr][{"syst": "SF", "flav": sum}].values()),
            )
            print(
                "l:%.3f",
                np.sum(collated["mc"][discr][{"syst": "SF", "flav": 0}].values())
                / np.sum(collated["mc"][discr][{"syst": "SF", "flav": sum}].values()),
            )
            print("============")

        own_bands = False  # bands and ratio drawn by bands_and_ratio (all but --SF)
        hdata_plot = None
        stack_labels = []
        # data points: same marker in both panels
        marker_size = plt.rcParams["lines.markersize"] + 1
        if mc_only:
            fig, ax = plt.subplots(figsize=(10, 10))
            rax = None
            fig.subplots_adjust(top=0.92, bottom=0.1, right=0.97)
        else:
            fig, ((ax), (rax)) = plt.subplots(
                2,
                1,
                figsize=(10, 10),
                gridspec_kw={"height_ratios": (3, 1)},
                sharex=True,
            )
            fig.subplots_adjust(hspace=0.06, top=0.92, bottom=0.1, right=0.97)
        hep.cms.label(
            args.CMS_label,
            data=not mc_only,
            lumi=f"{args.lumi / 1000.0:.3g}",  # 3 significant digits
            com=args.com,
            loc=0,
            ax=ax,
        )

        ## w/ & w/o btag SF
        if (
            "flav" in collated["mc"][discr].axes.name
            and "syst" in collated["mc"][discr].axes.name
            and args.SF
            and args.split != "sample"
        ):
            splitflav_stack, labels, color_config = [], [], {}
            color_config["color"], color_config["alpha"] = [], []
            splitflav_axis = SF_axis
            for sample in collated.keys():
                if sample == "data":
                    continue
                if args.split == "flavor" and sample != "mc":
                    continue
                elif args.split == "sample_flav" and sample == "mc":
                    continue
                for i, (t, hcomp) in enumerate(
                    flav_components(
                        collated[sample][discr], splitflav_axis, flav_set, args.hf_split
                    )
                ):
                    splitflav_stack.append(hcomp)
                    if args.split == "flavor":
                        labels.append(t)
                        color_config["color"].append(color_map[t])
                        color_config["alpha"].append(0.55 if t in ("bb", "cc") else 1.0)
                    else:
                        labels.append(f"{sample} ({t})")
                        color_config["color"].append(color_map[sample])
                        color_config["alpha"].append(1.0 - 0.2 * i)

            SF_axis["flav"] = sum
            hep.histplot(
                splitflav_stack,
                stack=True,
                label=labels,
                histtype="fill",
                edgecolor="black",  # black borders of the stacked histograms
                linewidth=1,
                yerr=True,
                ax=ax,
                flow=args.flow,
                sort="y",  # sort by yield
                **color_config,
            )
            hep.histplot(
                collated["mc"][discr][noSF_axis],
                label=["w/o SF"],
                color="#9494AE",
                width=2,
                yerr=True,
                ax=ax,
                flow=args.flow,
                sort="y",  # sort by yield
            )
            if not mc_only:
                hep.histplot(
                    collated["data"][discr][noSF_axis],
                    histtype="errorbar",
                    color="black",
                    label="Data",
                    yerr=True,
                    ax=ax,
                    xerr=True,
                    marker="o",
                    markersize=marker_size,
                    flow=args.flow,
                )
            hmc = collated["mc"][discr][SF_axis]
            MCerrorband(hmc, ax=ax, flow=args.flow)  # stat. unc. errorband
            sf_err = SFerror(collated, discr, flow=args.flow)
            other = {"hatch": "\\\\", "lw": 0, "color": "r", "alpha": 0.4}
            MCerrorband(
                hmc,
                ax=ax,
                flow=args.flow,
                ext_error=sf_err,
                label="SF unc.",
                fill_opts=other,
            )  # stat. unc. errorband
            if not mc_only:
                plotratio(
                    collated["data"][discr][noSF_axis],
                    collated["mc"][discr][noSF_axis],
                    ax=rax,
                    flow=args.flow,
                    xerr=do_xerr,
                )
            if not mc_only:
                plotratio(
                    collated["data"][discr][noSF_axis],
                    hmc,
                    ax=rax,
                    ext_denom_error=sf_err,
                    error_opts={"color": "r", "marker": "v"},
                    denom_fill_opts=other,
                    clear=False,
                    label="SF unc.",
                    flow=args.flow,
                    xerr=do_xerr,
                )

        elif (
            "syst" in collated["mc"][discr].axes.name
            and not args.SF
            and "noSF" in systlist
            and args.split != "sample"
        ):
            splitflav_stack, labels, color_config = [], [], {}
            color_config["color"], color_config["alpha"] = [], []
            splitflav_axis = noSF_axis
            for sample in collated.keys():
                if sample == "data":
                    continue
                if args.split == "flavor" and sample != "mc":
                    continue
                elif args.split == "sample_flav" and sample == "mc":
                    continue
                for i, (t, hcomp) in enumerate(
                    flav_components(
                        collated[sample][discr], splitflav_axis, flav_set, args.hf_split
                    )
                ):
                    splitflav_stack.append(hcomp)
                    if args.split == "flavor":
                        labels.append(t)
                        color_config["color"].append(color_map[t])
                        color_config["alpha"].append(0.55 if t in ("bb", "cc") else 1.0)
                    else:
                        labels.append(f"{sample} ({t})")
                        color_config["color"].append(color_map[sample])
                        color_config["alpha"].append(1.0 - 0.2 * i)
            noSF_axis["flav"] = sum
            hep.histplot(
                splitflav_stack,
                stack=True,
                histtype="fill",
                edgecolor="black",  # black borders of the stacked histograms
                linewidth=1,
                label=labels,
                yerr=True,
                ax=ax,
                flow=args.flow,
                **color_config,
            )
            stack_labels = list(labels)
            if not mc_only:
                hep.histplot(
                    collated["data"][discr][noSF_axis],
                    histtype="errorbar",
                    color="black",
                    label="Data",
                    yerr=True,
                    ax=ax,
                    xerr=True,
                    marker="o",
                    markersize=marker_size,
                    flow=args.flow,
                )
            hmc = collated["mc"][discr][noSF_axis]
            hdata_plot = None if mc_only else collated["data"][discr][noSF_axis]
            own_bands = True
        elif "flav" in collated["mc"][discr].axes.name and args.split != "sample":
            splitflav_stack, labels, color_config = [], [], {}
            color_config["color"], color_config["alpha"] = [], []
            splitflav_axis = allaxis
            for sample in collated.keys():
                if sample == "data":
                    continue
                if args.split == "flavor" and sample != "mc":
                    continue
                elif args.split == "sample_flav" and sample == "mc":
                    continue
                for i, (t, hcomp) in enumerate(
                    flav_components(
                        collated[sample][discr], splitflav_axis, flav_set, args.hf_split
                    )
                ):
                    splitflav_stack.append(hcomp)
                    if args.split == "flavor":
                        labels.append(t)
                        color_config["color"].append(color_map[t])
                        color_config["alpha"].append(0.55 if t in ("bb", "cc") else 1.0)
                    else:
                        labels.append(f"{sample} ({t})")
                        color_config["color"].append(color_map[sample])

                        color_config["alpha"].append(1.0 - 0.2 * i)
            allaxis["flav"] = sum
            hep.histplot(
                splitflav_stack,
                stack=True,
                histtype="fill",
                edgecolor="black",  # black borders of the stacked histograms
                linewidth=1,
                label=labels,
                yerr=True,
                ax=ax,
                flow=args.flow,
                **color_config,
            )
            stack_labels = list(labels)
            if not mc_only:
                hep.histplot(
                    collated["data"][discr][allaxis],
                    histtype="errorbar",
                    color="black",
                    label="Data",
                    yerr=True,
                    ax=ax,
                    xerr=True,
                    marker="o",
                    markersize=marker_size,
                    flow=args.flow,
                )
            hmc = collated["mc"][discr][allaxis]
            hdata_plot = None if mc_only else collated["data"][discr][allaxis]
            own_bands = True
        else:
            hmc = collated["mc"][discr][allaxis]
            if "sample" in args.split:
                hep.histplot(
                    [
                        collated[s][discr][allaxis]
                        for s in collated.keys()
                        if s != "mc" and s != "data"
                    ],
                    histtype="fill",
                    edgecolor="black",  # black borders of the stacked histograms
                    linewidth=1,
                    stack=True,
                    label=[s for s in collated.keys() if s != "mc" and s != "data"],
                    yerr=True,
                    ax=ax,
                    color=[
                        color_map[s]
                        for s in collated.keys()
                        if s != "mc" and s != "data"
                    ],
                    flow=args.flow,
                    sort="y",  # sort by yield
                )
                stack_labels = [s for s in collated.keys() if s != "mc" and s != "data"]
            else:
                hep.histplot(
                    hmc,
                    color="#3f90da",
                    histtype="fill",
                    edgecolor="black",  # black borders of the stacked histograms
                    linewidth=1,
                    label=["MC"],
                    yerr=True,
                    ax=ax,
                    flow=args.flow,
                    sort="y",  # sort by yield
                )
            if not mc_only:
                hep.histplot(
                    collated["data"][discr][allaxis],
                    histtype="errorbar",
                    color="black",
                    label="Data",
                    yerr=True,
                    ax=ax,
                    xerr=True,
                    marker="o",
                    markersize=marker_size,
                    flow=args.flow,
                )
            hdata_plot = None if mc_only else collated["data"][discr][allaxis]
            own_bands = True

        # uncertainty bands (MC stat.; stat. + syst. with --syst-band: all variations of
        # the syst axis) and Data/MC points
        err_syst = None
        if own_bands and args.syst_band:
            hfull = collated["mc"][discr]
            if "syst" not in hfull.axes.name:
                print(f"{discr}: no syst axis, only the MC stat. uncertainty is drawn")
            else:
                nominal = allaxis["syst"]
                sel = {k: v for k, v in allaxis.items() if k != "syst"}
                if "flav" in hfull.axes.name:
                    sel["flav"] = sum
                err_dn, err_up, used = syst_band(hfull, sel, nominal, args.flow, syst_exclude)
                err_syst = (err_dn, err_up)
                print(f"{discr}: syst. band from {len(used)} variations"
                      + (f" ({', '.join(used)})" if used else " (none found)"))
                report_variations(hfull, sel, nominal, used, discr)
        if own_bands:
            bands_and_ratio(ax, rax, hmc, hdata_plot, args.flow, marker_size, err_syst)

        xax = ax if rax is None else rax  # axis with the x label and range
        ax.set_xlabel(None)
        ax.set_ylabel("Events")
        if rax is not None:
            rax.set_ylabel("Data/MC")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        ax.get_yaxis().get_offset_text().set_position((-0.065, 1.05))

        # FIXME: add wildcard option for xlabel
        xlabel = (
            args.xlabel
            if args.xlabel is not None
            else collated["mc"][discr].axes[-1].label
        )  # Use label from stored hists

        ## FIXME: Set temporary fix for the x-axis
        if args.xlabel is not None:
            args.xlabel.split(",")[index]
        elif discr in category_of:
            # one category of a histogram with selections: the label of the plotted axis
            xlabel = collated["mc"][discr].axes[-1].label
        elif base_of.get(discr, discr) != discr:
            # one charge combination: as the original histogram
            base = base_of[discr]
            xlabel = (
                collated["mc"][discr].axes[-1].label
                if base.startswith(("stagdisc_", "stagscore_"))
                or collated["mc"][discr].axes[-1].name == "discr"
                or "DeepJet" in base or "DeepCSV" in base or "PFCands" in base
                or "JetSVs_" in base
                else axes_name(base)
            )
        elif "DeepJet" in discr or "DeepCSV" in discr or "PFCands" in discr:
            xlabel = (
                bininfo[discr]["displayname"]
                + " ["
                + bininfo[discr]["inputVar_units"]
                + "]"
                if (bininfo[discr]["inputVar_units"] is not None)
                and (bininfo[discr]["inputVar_units"] == "")
                else bininfo[discr]["displayname"]
            )
        elif "JetSVs_" in discr:
            xlabel = (
                SV_bininfo[discr]["displayname"]
                + " ["
                + SV_bininfo[discr]["inputVar_units"]
                + "]"
                if (SV_bininfo[discr]["inputVar_units"] is not None)
                and (SV_bininfo[discr]["inputVar_units"] == "")
                else SV_bininfo[discr]["displayname"]
            )
        elif discr.startswith(("stagdisc_", "stagscore_")) or (
            collated["mc"][discr].axes[-1].name == "discr"
        ):
            # tagger outputs: the axis label is the branch name
            xlabel = collated["mc"][discr].axes[-1].label
        else:
            xlabel = axes_name(discr)

        xax.set_xlabel(xlabel)

        if "2D" in discr:
            xtickpos = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
            xticklabels = [
                "L0",
                "C0",
                "C1",
                "C2",
                "C3",
                "C4",
                "B0",
                "B1",
                "B2",
                "B3",
                "B4",
            ]
            xax.set_xticks(xtickpos, xticklabels)

        # legend: stacked histograms from the top of the stack, then the rest
        handles, leg_labels = ax.get_legend_handles_labels()
        top_first = [leg_labels.index(lab) for lab in reversed(stack_labels) if lab in leg_labels]
        order = top_first + [i for i in range(len(leg_labels)) if i not in top_first]
        handles = [handles[i] for i in order]
        leg_labels = [leg_labels[i] for i in order]
        if "sample" in args.split:
            ax.legend(handles, leg_labels, ncols=2, prop={"size": 16})
        else:
            ax.legend(
                handles,
                leg_labels,
                ncols=(2 if len(flav_set) > 3 else 1),
            )

        if rax is not None:
            rax.set_ylim(
                float(args.ratio_range.split(",")[0]),
                float(args.ratio_range.split(",")[1]),
            )
        ax.set_ylim(bottom=0.0)

        xax.autoscale(True, axis="x", tight=True)
        xmin, xmax = autoranger(
            (
                collated["mc"][discr][allaxis]
                if mc_only
                else collated["data"][discr][allaxis] + collated["mc"][discr][allaxis]
            ),
            flow=args.flow,
        )

        if args.xrange is not None:
            xmin, xmax = float(args.xrange.split(",")[0]), float(
                args.xrange.split(",")[1]
            )

        xax.set_xlim(xmin, xmax)

        if "2Dbin_pt" in discr or "HFvLF_pt" in discr or "BvC_pt" in discr:
            var_txt += "\n" + low_edge + " < jet $p_T$ < " + up_edge + " GeV"
        if discr in category_of:
            var_txt += "\n" + category_of[discr]
        if discr in osss_label:
            var_txt += "\n" + osss_label[discr]

        at = AnchoredText(var_txt + "\n" + args.ext, loc=2, frameon=False)
        ax.add_artist(at)

        scale = ""
        if args.norm:
            scale = "_norm"
        if args.outdir:
            # <outdir>/<variable>[__<category>]_<split>[_mconly][_log].<ext>
            name = {"flavor": "flavour", "sample": "process",
                    "sample_flav": "process_flavour"}[args.split]
        else:
            name = "all"
            if args.split == "sample":
                name += "_sample"
        if mc_only:
            name += "_mconly"

        try:
            hep.mpl_magic(ax=ax)
        except RuntimeError as e:
            print(f"Warning: {e}")
            print("Using soft_fail=True for legend placement")
            try:
                # Try with soft_fail=True
                hep.mpl_magic(ax=ax, soft_fail=True)
            except Exception as e2:
                print(f"Still failed: {e2}")
                # Continue anyway - the plot will still be usable

        if args.log:
            name += "_log"
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.1)
            hep.mpl_magic(ax=ax)

        if args.logx:
            name += "_logx"
            ax.set_xscale("log")

        # Save the plots
        def save(name):
            stem = (
                f"{outdir}/{discr}{scale}_{name}"
                if args.outdir
                else f"{outdir}/unc_{discr}_inclusive{scale}_{name}"
            )
            for filetype in args.save_as.split(","):
                if filetype not in ["png", "pdf"]:
                    print(
                        f"Unsupported file type: {filetype}. Only 'png' and 'pdf' are supported."
                    )
                    continue
                print("creating:", f"{stem}.{filetype}")
                fig.savefig(
                    f"{stem}.{filetype}",
                    bbox_inches="tight",
                    dpi=200 if filetype == "png" else None,
                )

        save(name)
        if args.also_log and not args.log:
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.1)
            try:
                hep.mpl_magic(ax=ax, soft_fail=True)
            except Exception as e:
                print(f"Warning: {e}")
            save(name + "_log")

        plt.close(fig)


if __name__ == "__main__":
    args = get_parser()
    sys.exit(main(args))
