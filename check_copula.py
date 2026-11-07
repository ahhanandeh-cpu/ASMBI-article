"""Direct simulation of Gaussian-copula concomitant ranking vs the matrix-model exact coverage."""
import numpy as np, json
from scipy.stats import norm
from pe import GOLD
from study_c import quantile_T
fam = GOLD(2); th = 1.0; m = 3; out = {}
C = json.load(open("study_c.json"))
for r in (3, 10):
    des0 = ('rss', m, r, None)
    a, b = quantile_T(fam, th, des0, 0.025), quantile_T(fam, th, des0, 0.975)
    for rho in (0.9, 0.75, 0.5):
        rng = np.random.default_rng(int(rho * 100) + r)
        N = 200000
        X = fam.rvs(N * r * m * m, th, rng).reshape(N, r, m, m)
        zx = norm.ppf(np.clip(1 - fam.sf(X, th), 1e-15, 1 - 1e-15))
        zy = rho * zx + np.sqrt(1 - rho ** 2) * rng.standard_normal(zx.shape)
        idx = np.argsort(zy, axis=3)                                  # judged order
        T = sum(np.take_along_axis(X[:, :, i, :], idx[:, :, i, i:i + 1], axis=2)[..., 0] for i in range(m)).sum(1)
        cov = np.mean((T >= a) & (T <= b)); se = np.sqrt(cov * (1 - cov) / N)
        blk = [o for o in C if o["name"] == "Lindley" and o["r"] == r][0]
        ex = [row["0.0"]["cover"] for row in blk["rows"] if row["kind"] == "rho" and row["value"] == rho][0]
        out[f"r{r}_rho{rho}"] = dict(sim=cov, se=se, matrix=ex)
        print(f"n={m*r} rho={rho}: direct simulation {cov:.4f} (se {se:.4f}) vs matrix model {ex:.4f}", flush=True)
json.dump(out, open("check_copula.json", "w"), indent=1)
