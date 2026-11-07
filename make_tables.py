"""Generate LaTeX tables and figures from the JSON outputs (no hand-copied numbers)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pe import GOLD, AKASH
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
T = "paper/tables/"
import os; os.makedirs(T, exist_ok=True)
FAMS = {"Lindley": (GOLD(2), 1.0, r"Lindley $(\theta=1)$"), "Sujatha": (GOLD(3), 0.6, r"Sujatha $(\theta=0.6)$"),
        "Amarendra": (GOLD(4), 1.2, r"Amarendra $(\theta=1.2)$"), "Akash": (AKASH, 1.5, r"Akash $(\lambda=1.5)$")}
def f4(v): return ("$-$%.4f" % -v) if v < -5e-5 else "%.4f" % abs(v)
def f5(v): return "%.5f" % v

# ---------- Table: exact MSE ----------
AB = json.load(open("study_ab.json"))
MC = json.load(open("mle_mc.json"))
def are_me(fam, th, m):
    v = np.mean([fam.os_moments(th, m, i)[1] for i in range(1, m + 1)])
    return fam.var(th) / v
L = [r"\begin{table}[H]\centering\small\setlength{\tabcolsep}{4pt}",
     r"\caption{Exact bias and MSE of the moment estimator under SRS and RSS ($m=3$ unless stated), and its exact finite-sample relative efficiency $\mathrm{RE}=\mathrm{MSE}_{\SRS}/\mathrm{MSE}_{\RSS}$; ARE$_\infty$ is the large-sample limit $\sigma^2/\bar\sigma^2$. These columns involve no simulation. The last column is the simulated efficiency of the RSS-MLE relative to the exact SRS MSE (20{,}000 replications).}\label{tab:mse}",
     r"\begin{tabular}{llrrrrrrr}\toprule",
     r"Case & $n$ & Bias(SRS) & MSE(SRS) & Bias(RSS) & MSE(RSS) & RE & ARE$_\infty$ & RE$_{\mathrm{MLE}}$\\\midrule"]
prev = None
for o in AB:
    if o["m"] != 3 or o["n"] == 60: continue
    fam, th, lab = FAMS[o["name"]]
    first = o["name"] != prev; prev = o["name"]
    if first and len(L) > 4: L.append(r"\addlinespace")
    mc = MC.get(f"{o['name']}|{o['n']}")
    remle = "%.3f" % (o['mse_srs'] / mc['mse']) if mc else "--"
    arec = "%.3f" % are_me(fam, th, 3) if first else ""
    L.append(f"{lab if first else ''} & {o['n']} & {f4(o['bias_srs'])} & {f5(o['mse_srs'])} & {f4(o['bias_rss'])} & {f5(o['mse_rss'])} & {o['re_exact']:.3f} & {arec} & {remle} \\\\")
L.append(r"\addlinespace\multicolumn{9}{l}{\emph{Effect of set size, Lindley, $n=60$}}\\")
for o in AB:
    if o["n"] != 60: continue
    fam, th, lab = FAMS[o["name"]]
    L.append(f"$m={o['m']}$ & 60 & {f4(o['bias_srs'])} & {f5(o['mse_srs'])} & {f4(o['bias_rss'])} & {f5(o['mse_rss'])} & {o['re_exact']:.3f} & {are_me(fam, th, o['m']):.3f} & -- \\\\")
L += [r"\bottomrule\end{tabular}\end{table}"]
open(T + "tab_mse.tex", "w").write("\n".join(L))

# ---------- Table: coverage ----------
MC = json.load(open("mle_mc.json"))
L = [r"\begin{table}[H]\centering\small",
     r"\caption{Coverage of nominal 95\% Wald intervals, $m=3$. Wald(ME): moment estimator with model-based standard error, coverage computed exactly; ``below'' and ``above'' are the probabilities that the RSS interval lies entirely below or entirely above the true parameter. Wald(RSS-MLE): simulated (20{,}000 replications; Monte Carlo s.e.\ $\approx0.0017$ for coverage), GOLD cases only. The exact interval of Theorem~\ref{thm:exact} has coverage exactly $0.950$ with $0.025$ in each tail.}\label{tab:cov}",
     r"\begin{tabular}{llrrrrrrr}\toprule",
     r"& & \multicolumn{4}{c}{Wald(ME), exact} & \multicolumn{3}{c}{Wald(RSS-MLE), simulated}\\\cmidrule(lr){3-6}\cmidrule(lr){7-9}",
     r"Case & $n$ & SRS & RSS & below & above & cover & below & above\\\midrule"]
prev = None
for o in AB:
    if o["m"] != 3 or o["n"] == 60: continue
    fam, th, lab = FAMS[o["name"]]
    first = o["name"] != prev; prev = o["name"]
    if first and len(L) > 5: L.append(r"\addlinespace")
    w = o["wald_rss"]
    if o["name"] == "Akash":     # parameter decreasing in T: T-low -> estimate too large -> interval above
        below, above = w["T_high_miss"], w["T_low_miss"]
    else:
        below, above = w["T_low_miss"], w["T_high_miss"]
    mc = MC.get(f"{o['name']}|{o['n']}")
    mcs = ("%.3f & %.3f & %.3f" % (mc["cover"], mc["below"], mc["above"])) if mc else "-- & -- & --"
    L.append(f"{lab if first else ''} & {o['n']} & {o['wald_srs']['cover']:.3f} & {w['cover']:.3f} & {below:.3f} & {above:.3f} & {mcs} \\\\")
L += [r"\bottomrule\end{tabular}\end{table}"]
open(T + "tab_cov.tex", "w").write("\n".join(L))

# ---------- Table + figure: ranking error ----------
C = json.load(open("study_c.json"))
g0s = ["0.0", "0.25", "0.5", "1.0"]
def blk(name):
    return [o for o in C if o["name"] == name]
L = [r"\begin{table}[H]\centering\small",
     r"\caption{Exact coverage of 95\% exact intervals calibrated at $\gamma_0$ (columns) when the true ranking follows the mixture model $P_\gamma$ or the concomitant model $P^{(\rho)}$ (rows), $m=3$. $\gamma_0=0$: perfect-ranking interval; $\gamma_0=1$: SRS-calibrated interval.}\label{tab:rank}",
     r"\begin{tabular}{ll" + "r" * 8 + r"}\toprule"]
for name in ("Lindley", "Amarendra"):
    b = blk(name)
    L.append(r"\multicolumn{10}{l}{\emph{%s}}\\" % FAMS[name][2])
    L.append(r" & & \multicolumn{4}{c}{$n=%d$} & \multicolumn{4}{c}{$n=%d$}\\\cmidrule(lr){3-6}\cmidrule(lr){7-10}" % (b[0]["n"], b[1]["n"]))
    L.append(r"\multicolumn{2}{l}{True ranking} & " + " & ".join([r"$\gamma_0=%s$" % g.rstrip('0').rstrip('.') if g != "0.0" else r"$\gamma_0=0$" for g in g0s] * 2) + r"\\\midrule")
    for k_, row in enumerate(b[0]["rows"]):
        lab = (r"$\gamma=%s$" % row["value"]) if row["kind"] == "lambda" else (r"$\rho=%s$" % row["value"])
        vals = [b[0]["rows"][k_][g]["cover"] for g in g0s] + [b[1]["rows"][k_][g]["cover"] for g in g0s]
        L.append(" & " + lab + " & " + " & ".join("%.3f" % v for v in vals) + r"\\")
    L.append(r"\addlinespace")
L[-1] = r"\bottomrule\end{tabular}\end{table}"
open(T + "tab_rank.tex", "w").write("\n".join(L))

fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
cols = plt.cm.viridis(np.linspace(0, 0.85, 4))
for a, o in zip(ax, blk("Lindley")):
    rows = [r for r in o["rows"] if r["kind"] == "lambda"]
    gs = [r["value"] for r in rows]
    for c, g in zip(cols, g0s):
        a.plot(gs, [r[g]["cover"] for r in rows], marker="o", color=c, lw=2, label=r"$\gamma_0=%s$" % (g.rstrip('0').rstrip('.') if g != "0.0" else "0"))
    a.axhline(0.95, color="grey", ls=":", lw=1)
    a.set_xlabel(r"true ranking-error level $\gamma$"); a.set_title(f"n = {o['n']}")
    a.set_ylim(0.70, 1.0)
ax[0].set_ylabel("exact coverage"); ax[0].legend(frameon=False, fontsize=10, loc="lower left")
fig.tight_layout(); fig.savefig("paper/figures/fig_rank.png")

# ---------- Application tables ----------
A = json.load(open("app.json"))
L = [r"\begin{table}[H]\centering\small",
     r"\caption{The ranked set sample ($m=3$, $r=3$) constructed from the aircraft window data: the three values in each set, the rank retained, and the measured (destructively tested) value.}\label{tab:appsample}",
     r"\begin{tabular}{cclcr}\toprule Cycle & Set & Values in set & Rank retained & Measured\\\midrule"]
for c, i, v, s in A["rss_sample"]:
    L.append(f"{c} & {i} & {', '.join('%g' % u for u in s)} & {i} & {v:g}\\\\")
L += [r"\bottomrule\end{tabular}\end{table}"]
open(T + "tab_appsample.tex", "w").write("\n".join(L))

I = A["intervals"]
def row(lab, est, lo, hi):
    return f"{lab} & {est:.3f} & [{lo:.3f}, {hi:.3f}] & {hi - lo:.3f}\\\\"
L = [r"\begin{table}[H]\centering\small",
     r"\caption{Inference on $\theta$ for the aircraft window data, GOLD$(\theta,19)$, $n=9$ destructive tests. Exact intervals from Theorem~\ref{thm:exact}; the estimate shown with exact intervals is the median-unbiased estimate.}\label{tab:appci}",
     r"\begin{tabular}{lrcr}\toprule Method & Estimate & 95\% interval & Width\\\midrule",
     r"\multicolumn{4}{l}{\emph{Ranked set sample}}\\",
     row("Exact, perfect ranking ($\\gamma_0=0$)", I["rss_exact"][1], I["rss_exact"][0], I["rss_exact"][2]),
     row("Exact, calibrated $\\gamma_0=0.25$", I["rss_exact_l0_0.25"][1], I["rss_exact_l0_0.25"][0], I["rss_exact_l0_0.25"][2]),
     row("Exact, SRS-calibrated ($\\gamma_0=1$)", I["rss_exact_l0_1.0"][1], I["rss_exact_l0_1.0"][0], I["rss_exact_l0_1.0"][2]),
     row("Wald, moment estimator", I["rss_wald_me"][0], I["rss_wald_me"][1], I["rss_wald_me"][2]),
     r"\addlinespace\multicolumn{4}{l}{\emph{Simple random sample of nine windows}}\\",
     row("Exact", I["srs_exact"][1], I["srs_exact"][0], I["srs_exact"][2]),
     row("Wald, MLE", I["srs_wald"][0], I["srs_wald"][1], I["srs_wald"][2]),
     r"\addlinespace\multicolumn{4}{l}{\emph{All 31 windows (SRS)}}\\",
     row("Exact", A["full"]["exact"][1], A["full"]["exact"][0], A["full"]["exact"][2]),
     r"\bottomrule\end{tabular}\end{table}"]
open(T + "tab_appci.tex", "w").write("\n".join(L))

P = A["profile"]
L = [r"\begin{table}[H]\centering\small",
     r"\caption{GOLD$(\theta,k)$ fits to all 31 windows for selected $k$: maximum likelihood estimate, maximized log-likelihood, and Kolmogorov--Smirnov statistic with its $p$-value.}\label{tab:profile}",
     r"\begin{tabular}{rrrrr}\toprule $k$ & $\hat\theta$ & $\log L$ & KS & $p$\\\midrule"]
for p_ in P:
    L.append("%d & %.4f & %.3f & %.3f & %.3f\\\\" % (p_["k"], p_["theta"], p_["loglik"], p_["ks"], p_["ks_p"]))
L += [r"\bottomrule\end{tabular}\end{table}"]
open(T + "tab_profile.tex", "w").write("\n".join(L))

# ---------- Design table + figure ----------
D = json.load(open("design.json"))
ns = [12, 24, 36, 48]
def get(n, kind):
    for o in D:
        if o["n"] == n and ((kind == "srs" and o["design"][0] == "srs") or (kind != "srs" and o["design"][0] == "rss" and o["design"][1] == kind)):
            return o
L = [r"\begin{table}[H]\centering\small",
     r"\caption{Exact expected width of the 95\% exact interval for $\theta$ (relative width $E(\theta_U-\theta_L)/\theta$ in parentheses) at the fitted model GOLD$(1.625,19)$.}\label{tab:design}",
     r"\begin{tabular}{rcccc}\toprule $n$ & SRS & RSS, $m=2$ & RSS, $m=3$ & RSS, $m=4$\\\midrule"]
for n in ns:
    cells = []
    for kd in ("srs", 2, 3, 4):
        o = get(n, kd); cells.append("%.3f (%.3f)" % (o["exp_width"], o["rel_width"]))
    L.append(f"{n} & " + " & ".join(cells) + r"\\")
L += [r"\bottomrule\end{tabular}\end{table}"]
open(T + "tab_design.tex", "w").write("\n".join(L))
fig, a = plt.subplots(figsize=(7, 4.3), dpi=300)
for kd, c, lab in (("srs", "grey", "SRS"), (2, "tab:blue", "RSS, m = 2"), (3, "tab:orange", "RSS, m = 3"), (4, "tab:green", "RSS, m = 4")):
    a.plot(ns, [get(n, kd)["rel_width"] for n in ns], marker="o", color=c, lw=2, label=lab)
a.set_xlabel("number of destructive tests n"); a.set_ylabel("expected relative width"); a.set_xticks(ns)
a.legend(frameon=False)
fig.tight_layout(); fig.savefig("paper/figures/fig_design.png")
print("tables and figures written")
