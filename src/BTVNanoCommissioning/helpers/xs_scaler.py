import copy
import math
import hist
from coffea.processor import accumulate
import os
from BTVNanoCommissioning.helpers.xsection import xsection
from alive_progress import alive_bar

"""
Scale histograms to corresponding cross-section. Merge mutiple `.coffea` and collate the MC samples into sub-class in this function.
"""
from BTVNanoCommissioning.helpers.xsection_13TeV import xsection_13TeV


def scale_xs(hist, lumi, events):
    xs_dict = {}
    for obj in xsection + xsection_13TeV:
        xs_dict[obj["process_name"]] = float(obj["cross_section"])
    scales = {}
    for key in events:
        if type(key) != str or "Run" in key:
            continue
        scales[key] = xs_dict[key] * lumi / events[key]
    hist.scale(scales, axis="dataset")
    return hist


def _renormalise_syst_slices(h, factors):
    """Multiply each `syst`-axis slice whose label is in `factors` by that factor
    (value by f, variance by f^2), in place. Slices not listed -- detector
    shifts, SF variations, the nominal -- are left as they are."""
    if "syst" not in h.axes.name:
        return h
    iax = h.axes.name.index("syst")
    ax = h.axes["syst"]
    view = h.view(flow=True)
    for label in ax:
        f = factors.get(label)
        if f is None or f == 1.0:
            continue
        sl = [slice(None)] * h.ndim
        sl[iax] = ax.index(label)
        sl = tuple(sl)
        if view.dtype.names and "variance" in view.dtype.names:
            view["value"][sl] *= f
            view["variance"][sl] *= f * f
        else:
            view[sl] *= f
    return h


def scaleSumW(output, lumi, syst_shapes=True, variables=None):
    scaled = {}
    xs_dict = {}
    for obj in xsection + xsection_13TeV:
        xs_dict[obj["process_name"]] = float(obj["cross_section"])
        # if k-factor in the xsection: multiply by the k-factor
        if "kFactor" in obj.keys() and obj["kFactor"] != "":
            xs_dict[obj["process_name"]] = xs_dict[obj["process_name"]] * float(
                obj["kFactor"]
            )
    # Maps output histogram suffix -> sumw accumulator key used for normalisation.
    # When syst_shapes=False every variation is normalised with the nominal sumw instead.
    syst_sumw_map = {
        "PDF_weightUp": "PDF_sumwUp",
        "PDF_weightDown": "PDF_sumwDown",
        "aS_weightUp": "aS_sumwUp",
        "aS_weightDown": "aS_sumwDown",
        "scalevar_muRUp": "muR_sumwUp",
        "scalevar_muRDown": "muR_sumwDown",
        "scalevar_muFUp": "muF_sumwUp",
        "scalevar_muFDown": "muF_sumwDown",
        "UEPS_ISRUp": "ISR_sumwUp",
        "UEPS_ISRDown": "ISR_sumwDown",
        "UEPS_FSRUp": "FSR_sumwUp",
        "UEPS_FSRDown": "FSR_sumwDown",
        "ttbar_weightUp": "ttbar_weight_sumwUp",
        "ttbar_weightDown": "ttbar_weight_sumwDown",
        "PDFaS_weightUp": "PDFaS_sumwUp",
        "PDFaS_weightDown": "PDFaS_sumwDown",
        "scalevar_muR_muFUp": "muR_muF_sumwUp",
        "scalevar_muR_muFDown": "muR_muF_sumwDown",
    }

    merged_output = merge_output(output)

    with alive_bar(
        len(merged_output.keys()), title="Scaling by the XS, lumi, and sum of weights"
    ) as bar:
        for sample, accu in merged_output.items():
            scaled[sample] = {}
            if "sumw" not in accu.keys():
                bar()
                continue
            scaled[sample]["sumw"] = merged_output[sample]["sumw"]
            # New outputs carry one sum per generator-level variation
            # (`sumw_<variation>`, correction.reweighting). Then EVERY such
            # variation is renormalised to its own inclusive sum, slice by slice
            # on the `syst` axis, and no per-syst companion histograms are needed.
            var_sumw = {
                k[len("sumw_"):]: v
                for k, v in accu.items()
                if isinstance(k, str) and k.startswith("sumw_")
            }
            for k, v in var_sumw.items():
                scaled[sample][f"sumw_{k}"] = v

            # Per-sample setup done once (NOT once per histogram key): the sum-of-
            # weights accumulators are per-sample scalars, so collecting them inside
            # the key loop repeated the work ~200x and spammed the warnings.
            is_mc = sample in xs_dict
            if is_mc:
                xs = xs_dict[sample] * lumi
                nominal_sumw = merged_output[sample]["sumw"]
            if is_mc and not var_sumw:
                # legacy outputs: per-syst sumw accumulators (fall back to nominal)
                for syst in [
                    "PDF",
                    "aS",
                    "PDFaS",
                    "muR",
                    "muF",
                    "ISR",
                    "FSR",
                    "ttbar_weight",
                    "muR_muF",
                ]:
                    for var in ["Up", "Down"]:
                        key_reweight = f"{syst}_sumw{var}"
                        if key_reweight in merged_output[sample].keys():
                            scaled[sample][key_reweight] = merged_output[sample][
                                key_reweight
                            ]
                        else:
                            scaled[sample][key_reweight] = nominal_sumw
                            print(f"WARNING: {key_reweight} not found!")

            for key, h_obj in accu.items():
                if not isinstance(h_obj, hist.Hist):
                    continue

                if is_mc and var_sumw:
                    h = h_obj * xs / nominal_sumw
                    if syst_shapes:
                        _renormalise_syst_slices(
                            h,
                            {
                                label: nominal_sumw / s
                                for label, s in var_sumw.items()
                                if math.isfinite(s) and s != 0
                            },
                        )
                    scaled[sample][key] = h
                elif is_mc:
                    # legacy outputs without `sumw_<variation>`
                    scaled[sample][key] = h_obj * xs / nominal_sumw

                    # Per-syst renormalised copies are only ever read for the
                    # variables actually plotted; building them for every one of
                    # the ~200 histogram keys is what makes scaleSumW run out of
                    # memory. `variables=None` keeps the old all-keys behaviour.
                    if variables is None or key in variables:
                        for suffix, sumw_key in syst_sumw_map.items():
                            sumw = (
                                scaled[sample][sumw_key]
                                if syst_shapes
                                else nominal_sumw
                            )
                            # A present-but-zero (or non-finite) varied sumw is as
                            # useless as a missing one -- dividing by it yields NaN
                            # that poisons the collated MC. Fall back to nominal.
                            if not math.isfinite(sumw) or sumw == 0:
                                sumw = nominal_sumw
                            scaled[sample][f"{key}_{suffix}"] = h_obj * xs / sumw
                elif ("data" in sample) or ("Run" in sample) or ("Double" in sample):
                    scaled[sample][key] = copy.deepcopy(h_obj)
                else:
                    raise KeyError(
                        sample,
                        "is not found in xsection.py. If you're using 13TeV samples, please use xsection_13TeV.py",
                    )
            bar()
    return scaled


## Additional rescale for MC
def additional_scale(output, scale, sample_to_scale):
    scaled = {}
    for files in output.keys():
        scaled[files] = {}
        if "sumw" not in output[files].keys():
            for sample, accu in output[files].items():
                scaled[files][sample] = {}
                for key, h_obj in accu.items():
                    if isinstance(h_obj, hist.Hist):
                        h = copy.deepcopy(h_obj)
                        if sample in sample_to_scale:
                            h = h * scale
                        else:
                            h = h
                        scaled[files][sample][key] = h
        else:
            for sample, accu in output.items():
                scaled[sample] = {}
                for key, h_obj in accu.items():
                    if isinstance(h_obj, hist.Hist):
                        h = copy.deepcopy(h_obj)
                        if sample in sample_to_scale:
                            h = h * scale
                        else:
                            h = h
                        scaled[sample][key] = h
    return scaled


def dict_depth(d):
    if isinstance(d, dict):
        return 1 + (max(map(dict_depth, d.values())) if d else 0)
    return 0


def merge_output(output):
    if dict_depth(output) <= 2:
        return output
    else:
        return accumulate([output[f] for f in output.keys()])


def collate(merged_output, mergemap):
    out = {}
    merged_output = merge_output(merged_output)
    for group, names in mergemap.items():
        out[group] = accumulate(
            [v for k, v in merged_output.items() if k.split("_FNAME_")[0] in names]
        )
    return out


def getSumW(accumulator):
    sumw = {}
    for key, accus in accumulator.items():
        sumw[key] = accus["sumw"]
    return sumw
