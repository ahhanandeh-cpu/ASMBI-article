"""Application: aircraft window strength (Fuller et al. 1994), GOLD(theta, k=4)."""
import numpy as np, json
from scipy.optimize import brentq
from pe import GOLD
from study_c import P_lambda

DATA = np.array([18.83, 20.80, 21.657, 23.03, 23.23, 24.05, 24.321, 25.50, 25.52, 25.80, 26.69, 26.77,
                 26.78, 27.05, 27.67, 29.90, 31.11, 33.20, 33.73, 33.76, 33.89, 34.76, 35.75, 35.91,
                 36.98, 37.08, 37.09, 39.58, 44.045, 45.29, 45.381])
K, M, R = 19, 3, 3
fam = GOLD(K)
Z = 1.959964
out = {}
from scipy.stats import kstest
prof = []
for k in (1, 2, 4, 10, 15, 17, 19, 21, 25, 30):
    g = GOLD(k); th = g.mu_inv(DATA.mean()); ll = float(np.log(g.pdf(DATA, th)).sum())
    ks = kstest(DATA, lambda x, g=g, th=th: 1 - g.sf(np.asarray(x), th))
    prof.append(dict(k=k, theta=th, loglik=ll, ks=ks.statistic, ks_p=ks.pvalue))
out["profile"] = prof

# full-data SRS fit and exact SRS CI from all 31
th_full = fam.mu_inv(DATA.mean())
out["full"] = dict(n=len(DATA), mle=th_full, exact=fam.exact_ci(DATA.sum(), ('srs', len(DATA)), lo=0.2, hi=20))

# one documented RSS (m=3, r=3) and one SRS of size 9, drawn from the 31 values
rng = np.random.default_rng(2026)
perm = rng.permutation(DATA)
sets = perm[:M * M * R].reshape(R, M, M)
rss = [(c + 1, i + 1, float(np.sort(sets[c, i])[i]), [float(v) for v in sets[c, i]]) for c in range(R) for i in range(M)]
x = np.array([v[2] for v in rss])
srs = rng.choice(DATA, size=M * R, replace=False)
out["rss_sample"] = rss
out["srs_sample"] = [float(v) for v in srs]
n = M * R
des = ('rss', M, R, None)

def wald_me(t, design, mm=None):
    th = fam.mu_inv(t / n)
    dmu = fam.var(th) / th ** 2
    if design[0] == 'srs':
        v = fam.var(th)
    else:
        v = np.mean([fam.os_moments(th, M, i)[1] for i in range(1, M + 1)])
    se = np.sqrt(v / n) / dmu
    return th, th - Z * se, th + Z * se

res = {}
res["rss_exact"] = fam.exact_ci(x.sum(), des, lo=0.2, hi=20)
res["rss_wald_me"] = wald_me(x.sum(), des)
for l0 in (0.25, 1.0):
    res[f"rss_exact_l0_{l0}"] = fam.exact_ci(x.sum(), ('rss', M, R, P_lambda(l0)), lo=0.2, hi=20)
res["srs_exact"] = fam.exact_ci(srs.sum(), ('srs', n), lo=0.2, hi=20)
res["srs_wald"] = wald_me(srs.sum(), ('srs', n))
out["intervals"] = res

# exact one-sided 95% lower bounds on reliability R(x0) = S(x0; theta) (increasing in theta)
def lower_theta(t, design):
    return brentq(lambda th: (1 - fam.cdf_T(t, th, design)) - 0.05, 0.2, 20, xtol=1e-10)
rel = {}
for lab, t, d in (("rss", x.sum(), des), ("srs", srs.sum(), ('srs', n)), ("full", DATA.sum(), ('srs', len(DATA)))):
    thl = lower_theta(t, d)
    rel[lab] = dict(theta_lower=thl, **{f"R({x0})": float(fam.sf(x0, thl)) for x0 in (20.0, 25.0)})
rel["point_full"] = {f"R({x0})": float(fam.sf(x0, th_full)) for x0 in (20.0, 25.0)}
rel["point_rss_mu"] = {f"R({x0})": float(fam.sf(x0, res["rss_exact"][1])) for x0 in (20.0, 25.0)}
rel["empirical"] = {f"R({x0})": float((DATA > x0).mean()) for x0 in (20.0, 25.0)}
out["reliability"] = rel
json.dump(out, open("app.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
