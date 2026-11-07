"""Expected width of 95% exact CIs for theta at the fitted aircraft model GOLD(1.6247, 19):
SRS vs RSS (m = 2, 3, 4). The law of T is computed by FFT convolution (conv.py), which agrees with
exact Talbot inversion to < 5e-7; endpoints are spot-checked against exact inversion for n = 9."""
import numpy as np, json
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator
from scipy.integrate import simpson
from multiprocessing import Pool
from pe import GOLD
from conv import total_cdf

K, TH = 19, 1.6246943901300896
fam = GOLD(K)


def F_on(design, th, t):
    pos, cdf = total_cdf(fam, th, design)
    return np.interp(t, pos, cdf)


def ends_conv(t, design):
    L = brentq(lambda th: F_on(design, th, t) - 0.975, TH / 6, TH * 6, xtol=1e-9)
    U = brentq(lambda th: F_on(design, th, t) - 0.025, TH / 6, TH * 6, xtol=1e-9)
    return L, U


import os
def exp_width(design):
    fn = "cache/design_%s.json" % "_".join(str(v) for v in design[:3])
    if os.path.exists(fn):
        return json.load(open(fn))
    o = _exp_width(design); json.dump(o, open(fn, "w"), default=float); return o


def _exp_width(design):
    n = fam.n_of(design)
    mu, sd = n * fam.mean(TH), np.sqrt(n * fam.var(TH))
    pos, cdf0 = total_cdf(fam, TH, design)
    tg = np.linspace(mu - 6 * sd, mu + 6 * sd, 61)
    LU = np.array([ends_conv(t, design) for t in tg])
    w = LU[:, 1] - LU[:, 0]
    F0 = np.interp(tg, pos, cdf0)
    # E[w(T)] = int w dF ; integrate by parts-free: Stieltjes sum on fine grid
    tt = np.linspace(tg[0], tg[-1], 2001)
    Ft = np.interp(tt, pos, cdf0)
    wt = PchipInterpolator(tg, w)(tt)
    ew = np.sum(0.5 * (wt[1:] + wt[:-1]) * np.diff(Ft)) / (Ft[-1] - Ft[0])
    out = dict(design=[design[0], design[1], design[2] if design[0] == 'rss' else None], n=n,
               exp_width=ew, rel_width=ew / TH, mass=Ft[-1] - Ft[0])
    if n == 9:   # spot check against exact inversion
        chk = []
        for t in (mu - sd, mu, mu + sd):
            eL, _, eU = fam.exact_ci(t, design if design[0] == 'rss' else ('srs', n), lo=TH / 2.5, hi=TH * 2.5)
            cL, cU = ends_conv(t, design)
            chk.append(max(abs(eL - cL) / eL, abs(eU - cU) / eU))
        out["max_rel_err_vs_exact"] = max(chk)
    return out


if __name__ == "__main__":
    jobs = []
    for n in (12, 24, 36, 48):
        jobs.append(('srs', n))
        for m in (2, 3, 4):
            jobs.append(('rss', m, n // m, None))
    jobs += [('srs', 9), ('rss', 3, 3, None)]
    with Pool(2) as p:
        R = p.map(exp_width, jobs)
    json.dump(R, open("design.json", "w"), indent=1, default=float)
    for o in R:
        print(o, flush=True)
