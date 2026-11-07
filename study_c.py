"""Study C: exact coverage and expected width of lambda0-calibrated exact CIs under ranking error."""
import numpy as np, json
from scipy.optimize import brentq
from scipy.integrate import quad
from multiprocessing import Pool
from pe import GOLD

M = 3
ALPHA = 0.05


def P_lambda(lam, m=M):
    return (1 - lam) * np.eye(m) + lam / m * np.ones((m, m))


def P_rho(rho, m=M, N=2_000_000, seed=11):
    """P[i,j] = P(true rank j | judged rank i) under a Gaussian-copula concomitant with corr rho."""
    if rho >= 1:
        return np.eye(m)
    rng = np.random.default_rng(seed)
    zx = rng.standard_normal((N, m))
    zy = rho * zx + np.sqrt(1 - rho ** 2) * rng.standard_normal((N, m))
    true_rank = np.argsort(np.argsort(zx, axis=1), axis=1)
    judged_order = np.argsort(zy, axis=1)            # judged_order[:, i] = unit judged i-th
    P = np.zeros((m, m))
    for i in range(m):
        tr = true_rank[np.arange(N), judged_order[:, i]]
        P[i] = np.bincount(tr, minlength=m) / N
    return P


def quantile_T(fam, th, design, p):
    n = fam.n_of(design)
    mu, sd = n * fam.mean(th), np.sqrt(n * fam.var(th))
    lo, hi = max(1e-9, mu - 12 * sd), mu + 25 * sd
    return brentq(lambda t: fam.cdf_T(t, th, design) - p, lo, hi, xtol=1e-10)


def coverage_table(args):
    name, k, th, r = args
    fam = GOLD(k)
    lam0s = [0.0, 0.25, 0.5, 1.0]
    truths = [("lambda", l, P_lambda(l)) for l in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)] + \
             [("rho", rho, P_rho(rho)) for rho in (0.9, 0.75, 0.5)]
    out = {"name": name, "k": k, "theta": th, "r": r, "n": M * r, "P_rho": {}, "rows": []}
    for kind, v, P in truths:
        if kind == "rho":
            out["P_rho"][str(v)] = P.tolist()
    qs = {}
    for l0 in lam0s:
        des0 = ('rss', M, r, None if l0 == 0 else P_lambda(l0))
        qs[l0] = (quantile_T(fam, th, des0, ALPHA / 2), quantile_T(fam, th, des0, 1 - ALPHA / 2))
    for kind, v, P in truths:
        des = ('rss', M, r, None if (kind == "lambda" and v == 0) else P)
        row = {"kind": kind, "value": v}
        for l0, (a, b) in qs.items():
            Fa, Fb = fam.cdf_T(a, th, des), fam.cdf_T(b, th, des)
            row[str(l0)] = dict(cover=Fb - Fa, low=Fa, high=1 - Fb)
        out["rows"].append(row)
    return out


if __name__ == "__main__":
    jobs = [("Lindley", 2, 1.0, 3), ("Lindley", 2, 1.0, 10), ("Amarendra", 4, 1.2, 3), ("Amarendra", 4, 1.2, 10)]
    with Pool(2) as p:
        R = p.map(coverage_table, jobs)
    json.dump(R, open("study_c.json", "w"), indent=1, default=float)
    for o in R:
        print(o["name"], "n=", o["n"])
        for row in o["rows"]:
            print("  %-6s %-5s" % (row["kind"], row["value"]), " ".join("l0=%s: %.4f" % (l0, row[l0]["cover"]) for l0 in ("0.0", "0.25", "0.5", "1.0")))
    print("P_rho:", json.dumps(R[0]["P_rho"], indent=0))
