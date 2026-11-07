"""Write paper/numbers.tex: every number quoted in the prose, computed from the JSON outputs."""
import json, numpy as np
from pe import GOLD

AB = {(o["name"], o["n"], o["m"]): o for o in json.load(open("study_ab.json"))}
MC = json.load(open("mle_mc.json"))
C = {(o["name"], o["n"]): o for o in json.load(open("study_c.json"))}
A = json.load(open("app.json"))
D = json.load(open("design.json"))
CC = json.load(open("check_copula.json"))
f3 = lambda v: "%.3f" % v
f2 = lambda v: "%.2f" % v
pct = lambda v: "%.1f\\%%" % (100 * v)

def ab(name, n, m=3): return AB[(name, n, m)]
def re_mle(name, n): return ab(name, n)["mse_srs"] / MC[f"{name}|{n}"]["mse"]
def rng_re(name): v = [ab(name, n)["re_exact"] for n in (6, 9, 15, 30)]; return min(v), max(v)

def are_inf(fam, th, m):
    return fam.var(th) / np.mean([fam.os_moments(th, m, i)[1] for i in range(1, m + 1)])

lines = []
def mac(name, val): lines.append("\\newcommand{\\%s}{%s}" % (name, val))

# ---------- efficiency ----------
lo_L, hi_L = rng_re("Lindley"); lo_S, hi_S = rng_re("Sujatha"); lo_A, hi_A = rng_re("Amarendra")
ak6, ak30 = ab("Akash", 6), ab("Akash", 30)
remle = [re_mle(nm, n) for nm in ("Lindley", "Sujatha", "Amarendra") for n in (6, 9, 15, 30)]
reme = [ab(nm, n)["re_exact"] for nm in ("Lindley", "Sujatha", "Amarendra") for n in (6, 9, 15, 30)]
gain = [a / b - 1 for a, b in zip(remle, reme)]
ms = [ab("Lindley", 60, m)["re_exact"] for m in (2, 3, 4, 5)]
rr = lambda lo, hi: f2(lo) if f2(lo) == f2(hi) else f"{f2(lo)}--{f2(hi)}"
eff = (f"For the three GOLD laws the exact relative efficiency of RSS is essentially constant in $n$: "
       f"{rr(lo_L, hi_L)} for Lindley, {rr(lo_S, hi_S)} for Sujatha and {rr(lo_A, hi_A)} for Amarendra, from $n=6$ to $n=30$. "
       f"These values already equal the large-sample limit $\\sigma^2/\\bar\\sigma^2$ at $n=6$, so the asymptotic efficiency is a reliable guide even for very small RSS samples. "
       f"For the Akash law the exact efficiency is {f2(ak6['re_exact'])} at $n=6$ and falls to {f2(ak30['re_exact'])} at $n=30$. At small $n$ it exceeds both the large-sample value and the bound $(m+1)/2=2$ that applies to the RSS mean, because RSS also roughly halves the bias of the nonlinear estimator $\\mu^{{-1}}(\\bar x)$ ({f3(ak6['bias_srs'])} under SRS against {f3(ak6['bias_rss'])} under RSS at $n=6$). "
       f"The simulated RSS-MLE improves on the moment estimator by only {round(100*min(gain))}--{round(100*max(gain))}\\% in MSE. "
       f"With $n=60$, the exact efficiency increases with the set size: {', '.join(f2(v) for v in ms)} for $m=2,\\dots,5$.")
mac("EffSummary", eff)

# ---------- coverage ----------
L6, L30 = ab("Lindley", 6), ab("Lindley", 30)
ak9 = ab("Akash", 9)
mc6 = MC["Lindley|6"]
cov = (f"At $n=6$ the Wald interval for the Lindley law covers {f3(L6['wald_srs']['cover'])} under SRS and {f3(L6['wald_rss']['cover'])} under RSS. "
       f"More seriously, its errors fall almost entirely in one tail. Under RSS, {pct(L6['wald_rss']['T_low_miss'])} of intervals lie entirely below $\\theta$ and only {pct(L6['wald_rss']['T_high_miss'])} entirely above, against a nominal 2.5\\% each. "
       f"The imbalance persists at $n=30$ ({pct(L30['wald_rss']['T_low_miss'])} against {pct(L30['wald_rss']['T_high_miss'])}) and is shared by the simulated Wald interval of the RSS-MLE ({pct(mc6['below'])} against {pct(mc6['above'])} at $n=6$). "
       f"For the Akash law the Wald interval slightly overcovers, but it is again lopsided in the opposite direction ({pct(ak9['wald_rss']['T_high_miss'])} of intervals lie entirely below $\\lambda$ and {pct(ak9['wald_rss']['T_low_miss'])} above at $n=9$). "
       f"One-sided statements, such as lower confidence bounds for reliability, are therefore badly calibrated by Wald intervals at the sample sizes typical of RSS. The exact interval puts exactly 2.5\\% in each tail at every $n$.")
mac("CovSummary", cov)

# ---------- ranking ----------
def cv(name, n, kind, val, g0):
    for row in C[(name, n)]["rows"]:
        if row["kind"] == kind and abs(row["value"] - val) < 1e-9:
            return row[g0]["cover"]
diff_n = max(abs(cv(nm, 9, k, v, g) - cv(nm, 30, k, v, g)) for nm in ("Lindley", "Amarendra")
             for k, v in [("lambda", x) for x in (0.1, 0.25, 0.5, 0.75, 1.0)] + [("rho", x) for x in (0.9, 0.75, 0.5)]
             for g in ("0.0", "0.25", "0.5", "1.0"))
valid = all(cv(nm, n, "lambda", g, g0) >= 0.95 - 1e-6 for nm in ("Lindley", "Amarendra") for n in (9, 30)
            for g0 in ("0.0", "0.25", "0.5", "1.0") for g in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0) if g <= float(g0) + 1e-9)
assert valid
rank = (f"The perfect-ranking interval ($\\gamma_0=0$) loses coverage quickly. At $n=9$ its coverage is {f3(cv('Lindley', 9, 'lambda', 0.1, '0.0'))} when one judged rank in ten is replaced by a random one ($\\gamma=0.1$), "
        f"{f3(cv('Lindley', 9, 'lambda', 0.25, '0.0'))} at $\\gamma=0.25$, and {f3(cv('Lindley', 9, 'lambda', 1.0, '0.0'))} (Lindley) or {f3(cv('Amarendra', 9, 'lambda', 1.0, '0.0'))} (Amarendra) under random ranking. "
        f"These losses do not shrink with the sample size: coverages at $n=9$ and $n=30$ differ by at most {f3(diff_n)} throughout Table~\\ref{{tab:rank}}. "
        f"In every case computed, an interval calibrated at $\\gamma_0$ has coverage at least $0.95$ whenever the true $\\gamma\\le\\gamma_0$, so the SRS-calibrated interval ($\\gamma_0=1$) is valid under any level of mixture ranking error. "
        f"Under the concomitant model, $\\rho=0.9$ causes about the same loss as $\\gamma=0.25$, and $\\rho=0.5$ a loss between those of $\\gamma=0.5$ and $\\gamma=0.75$.")
mac("RankSummary", rank)
cop = (f"To check the matrix approximation for concomitant ranking, we simulated RSS with units ranked directly by a Gaussian-copula concomitant (Lindley, 200{{,}}000 replications). "
       f"The coverage of the perfect-ranking interval at $n=9$ was {f3(CC['r3_rho0.9']['sim'])}, {f3(CC['r3_rho0.75']['sim'])} and {f3(CC['r3_rho0.5']['sim'])} at $\\rho=0.9,0.75,0.5$, "
       f"against {f3(CC['r3_rho0.9']['matrix'])}, {f3(CC['r3_rho0.75']['matrix'])} and {f3(CC['r3_rho0.5']['matrix'])} from the matrix model, with similar results at $n=30$. "
       f"The matrix model therefore overstates the harm of concomitant ranking error by one to two percentage points. Calibration based on it errs on the safe side, but ranking error by a concomitant still costs real coverage.")
mac("CopulaCheck", cop)

# ---------- application ----------
I = A["intervals"]; Rl = A["reliability"]; full = A["full"]
mac("AppThetaFull", f3(full["mle"])); mac("AppFullL", f3(full["exact"][0])); mac("AppFullU", f3(full["exact"][2]))
mac("AppRtwentyFit", f3(Rl["point_full"]["R(20.0)"])); mac("AppRtwentyfiveFit", f3(Rl["point_full"]["R(25.0)"]))
mac("AppRtwentyEmp", f3(Rl["empirical"]["R(20.0)"])); mac("AppRtwentyfiveEmp", f3(Rl["empirical"]["R(25.0)"]))
w = lambda c: c[2] - c[0]
wr, ws = w(I["rss_exact"]), w(I["srs_exact"])
mu_ = I["rss_exact"][1]
app = (f"The exact perfect-ranking interval from the nine ranked measurements is $[{f3(I['rss_exact'][0])},{f3(I['rss_exact'][2])}]$, about {round(100*(1-wr/ws))}\\% narrower than the exact interval $[{f3(I['srs_exact'][0])},{f3(I['srs_exact'][2])}]$ from the nine-window SRS, and it contains the full-data estimate. "
       f"The Wald interval $[{f3(I['rss_wald_me'][1])},{f3(I['rss_wald_me'][2])}]$ has a similar width but is symmetric about the estimate. "
       f"The exact interval is not symmetric: it extends {f3(I['rss_exact'][2]-mu_)} above the median-unbiased estimate and {f3(mu_-I['rss_exact'][0])} below, reflecting the skewness of the sampling distribution. "
       f"Allowing for ranking error widens the interval to $[{f3(I['rss_exact_l0_0.25'][0])},{f3(I['rss_exact_l0_0.25'][2])}]$ at $\\gamma_0=0.25$ and to $[{f3(I['rss_exact_l0_1.0'][0])},{f3(I['rss_exact_l0_1.0'][2])}]$ under SRS calibration. "
       f"In engineering terms, the exact one-sided 95\\% lower confidence bound for the probability that a window survives a stress of 20 is {f3(Rl['rss']['R(20.0)'])} from the RSS, against {f3(Rl['srs']['R(20.0)'])} from the SRS of the same size and {f3(Rl['full']['R(20.0)'])} from all 31 windows. "
       f"At a stress of 25 the bounds are {f3(Rl['rss']['R(25.0)'])}, {f3(Rl['srs']['R(25.0)'])} and {f3(Rl['full']['R(25.0)'])}. "
       f"These comparisons rest on a single draw of each design; Table~\\ref{{tab:design}} gives the expected behaviour.")
mac("AppSummary", app)

def dget(n, kd):
    for o in D:
        if o["n"] == n and ((kd == "srs" and o["design"][0] == "srs") or (kd != "srs" and o["design"][0] == "rss" and o["design"][1] == kd)):
            return o["rel_width"]
chk = max(o.get("max_rel_err_vs_exact", 0) for o in D)
mac("DesignChk", "%.1f\\times10^{-8}" % (chk * 1e8) if chk < 1e-7 else "%.1e" % chk)
fam19 = GOLD(19); th19 = full["mle"]
are19 = are_inf(fam19, th19, 3)
des = (f"Under SRS, 24 destructive tests give an expected relative width of {f3(dget(24,'srs'))}. RSS with $m=3$ achieves {f3(dget(12,3))} with only 12 tests, and $m=4$ achieves {f3(dget(12,4))}. "
       f"Similarly, 48 SRS tests ({f3(dget(48,'srs'))}) are matched by 24 RSS tests with $m=3$ ({f3(dget(24,3))}). "
       f"For a target precision, RSS with sets of three therefore needs about half as many destroyed windows, consistent with the large-sample efficiency {f2(are19)} of the moment estimator at the fitted model. "
       f"The price is that $m$ windows must be inspected and ranked for every one destroyed. When inspection is cheap and destruction is not, this trade is favourable.")
mac("DesignSummary", des)
open("paper/numbers.tex", "w").write("\n".join(lines) + "\n")
print("\n\n".join(lines))
