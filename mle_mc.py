"""Monte Carlo coverage of the RSS-MLE Wald interval (perfect ranking), GOLD cases, m=3."""
import numpy as np, json, sys
sys.path.insert(0, ".")
import gold as g
Z = 1.959964; M = 3; N = 20000
out = {}
for name, th, k in (("Lindley", 1.0, 2), ("Sujatha", 0.6, 3), ("Amarendra", 1.2, 4)):
    grid = np.exp(np.linspace(np.log(th / 6), np.log(th * 6), 61))
    D = np.array([g.Delta(t, k) for t in grid])
    Dint = lambda t: np.interp(np.log(t), np.log(grid), D)
    for n in (6, 9, 15, 30):
        r = n // M; rng = np.random.default_rng(1000 + 10 * k + n)
        ranks = np.tile(np.arange(1, M + 1), r)
        est = np.empty(N); cov = 0; lo_miss = hi_miss = 0
        for rep in range(N):
            pool = np.sort(g.rgold(M * M * r, th, k, rng).reshape(r, M, M), axis=2)
            x = np.array([pool[c, i, i] for c in range(r) for i in range(M)])
            tme = g.mu_inv(x.mean(), k)
            tml = g.rss_mle(x, ranks, M, k, tme)
            tml = min(max(tml, grid[0]), grid[-1])
            se = 1 / np.sqrt(n * (g.info_srs(tml, k) + (M - 1) * Dint(tml)))
            if tml - Z * se > th: lo_miss += 1        # interval entirely above theta
            elif tml + Z * se < th: hi_miss += 1      # interval entirely below theta
            else: cov += 1
            est[rep] = tml
        out[f"{name}|{n}"] = dict(cover=cov / N, above=lo_miss / N, below=hi_miss / N,
                                  bias=est.mean() - th, mse=np.mean((est - th) ** 2), N=N)
        print(name, n, out[f"{name}|{n}"], flush=True)
json.dump(out, open("mle_mc.json", "w"), indent=1)
