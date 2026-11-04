"""Pileup weights for another minimum-bias cross section (e.g. 75.3 mb).

w_x(n) = w_official_nominal(n) * P_x(n) / P_69.2(n), with P from the BRIL pileup JSON
(pileupCalc 'true' mode). The ratio cancels any small difference to the official histogram;
--check compares P_69.2 and P_72.4 with the official data histograms.
"""

import argparse
import gzip
import json

import numpy as np
import uproot
from scipy.special import erf

PU = "/eos/user/c/cmsdqm/www/CAF/certification"
LUM = "/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM"
YEARS = {
    "2017": dict(
        pujson=f"{PU}/Collisions17/13TeV/PileUp/UltraLegacy/pileup_latest.txt",
        golden=f"{PU}/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt",
        official=f"{PU}/Collisions17/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-{{}}ub-99bins.root",
        weights=f"{LUM}/Run2-2017-UL-NanoAODv9/latest/puWeights.json.gz",
        nbins=99,
    ),
    "2024": dict(
        pujson=f"{PU}/Collisions24/PileUp/pileup_JSON-2024CDEFGHI_Golden.txt",
        golden=None,
        official=f"{PU}/Collisions24/PileUp/dataPileupHistogram-2024CDEFGHI_Golden-{{}}ub.root",
        weights=f"{LUM}/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/latest/puWeights_CDEFGHI.json.gz",
        nbins=100,
    ),
    "2025": dict(
        pujson=f"{PU}/Collisions25/PileUp/pileup_JSON-2025_Golden.txt",
        golden=None,
        official=f"{PU}/Collisions25/PileUp/dataPileupHistogram-2025pp_Golden-{{}}ub.root",
        weights=f"{LUM}/Run3-25Prompt-Summer24-NanoAODv15/latest/puWeights_2025pp_Golden_Summer24_25ns_69200ub.json.gz",
        nbins=100,
    ),
}


def profile(cfg, xsec_ub):
    pu = json.load(open(cfg["pujson"]))
    good = json.load(open(cfg["golden"])) if cfg["golden"] else None
    edges = np.arange(cfg["nbins"] + 1, dtype=float)
    h = np.zeros(cfg["nbins"])
    for run, lss in pu.items():
        rng = good.get(run) if good is not None else None
        if good is not None and rng is None:
            continue
        a = np.asarray(lss, dtype=float)
        if rng is not None:
            ls = a[:, 0]
            a = a[np.any([(ls >= lo) & (ls <= hi) for lo, hi in rng], axis=0)]
        if len(a) == 0:
            continue
        lumi, rms, mean = a[:, 1], a[:, 2] * xsec_ub, a[:, 3] * xsec_ub
        ok = rms > 0
        above = erf((mean[ok, None] - edges[None, :]) / np.sqrt(2) / rms[ok, None])
        h += (0.5 * (above[:, :-1] - above[:, 1:]) * lumi[ok, None]).sum(0)
        idx = np.floor(mean[~ok]).astype(int)
        sel = (idx >= 0) & (idx < cfg["nbins"]) & (mean[~ok] >= 1e-5)
        np.add.at(h, idx[sel], lumi[~ok][sel])
    return h / h.sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", choices=YEARS, required=True)
    p.add_argument("--xsec", type=float, default=75300.0, help="minimum-bias xsec in ub")
    p.add_argument("--check", action="store_true")
    p.add_argument("-o", "--output", required=True)
    a = p.parse_args()
    cfg = YEARS[a.year]

    p69 = profile(cfg, 69200.0)
    if a.check:
        for x in (69200, 72400):
            off = uproot.open(cfg["official"].format(x))["pileup"].values()[: cfg["nbins"]]
            off = off / off.sum()
            mine = p69 if x == 69200 else profile(cfg, float(x))
            print(
                f"{a.year} {x} ub: max |diff| = {np.abs(mine - off).max():.2e}, "
                f"mean mine {np.dot(mine, np.arange(cfg['nbins']) + 0.5):.3f} "
                f"official {np.dot(off, np.arange(cfg['nbins']) + 0.5):.3f}"
            )
    px = profile(cfg, a.xsec)

    d = json.load(gzip.open(cfg["weights"]))
    corr = d["corrections"][0]
    ratio = np.divide(px, p69, out=np.ones_like(px), where=p69 > 0)
    for c in corr["data"]["content"]:
        node = c["value"]
        n = len(node["content"])
        r = np.concatenate([ratio, np.ones(max(0, n - len(ratio)))])[:n]
        node["content"] = list(np.asarray(node["content"]) * r)
    corr["description"] = f"official weights x P({a.xsec:.0f} ub)/P(69200 ub)"
    d["corrections"] = [corr]
    with gzip.open(a.output, "wt") as f:
        json.dump(d, f)
    print(f"wrote {a.output}: <PU> {np.dot(px, np.arange(len(px)) + 0.5):.2f} at {a.xsec:.0f} ub")


if __name__ == "__main__":
    main()
