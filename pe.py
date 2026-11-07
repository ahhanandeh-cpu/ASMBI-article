"""Exact finite-sample inference under RSS for polynomial-exponential (PE) lifetime families.

A PE family has density f(x; lam) = h(x) exp(-lam x) / C(lam), h(x) = sum_{j=1}^K a_j x^{j-1}, a_j >= 0,
i.e. an Erlang(j, lam) mixture with common rate lam and weights w_j(lam) = a_j (j-1)! lam^{-j} / C(lam).
GOLD(theta,k): a_j = 1 (j<=k), lam = 1/theta.  Akash(theta): a = (1,0,1), lam = theta.
"""
import numpy as np
import mpmath as mp
from math import comb, factorial
from numpy.polynomial import polynomial as Pn
from scipy.optimize import brentq

mp.mp.dps = 30


class PE:
    def __init__(self, a, lam_of, name=""):
        self.a = np.asarray(a, float)          # coefficients of h (ascending powers)
        self.K = len(a)
        self.lam_of = lam_of                    # parameter -> rate
        self.name = name
        self._dps = {}                          # calibrated working precision per design

    # ---- basic quantities ----
    def weights(self, th):
        lam = self.lam_of(th)
        w = np.array([self.a[j - 1] * factorial(j - 1) / lam ** j for j in range(1, self.K + 1)])
        return w / w.sum()

    def mean(self, th):
        lam = self.lam_of(th); w = self.weights(th)
        return float(sum(w[j - 1] * j / lam for j in range(1, self.K + 1)))

    def var(self, th):
        lam = self.lam_of(th); w = self.weights(th)
        m2 = sum(w[j - 1] * j * (j + 1) / lam ** 2 for j in range(1, self.K + 1))
        return float(m2 - self.mean(th) ** 2)

    def pdf_poly(self, th):
        """f(x) = exp(-lam x) * poly(x)."""
        lam = self.lam_of(th); w = self.weights(th)
        c = np.zeros(self.K)
        for j in range(1, self.K + 1):
            c[j - 1] += w[j - 1] * lam ** j / factorial(j - 1)
        return c

    def sf_poly(self, th):
        """S(x) = exp(-lam x) * poly(x)."""
        lam = self.lam_of(th); w = self.weights(th)
        c = np.zeros(self.K)
        for j in range(1, self.K + 1):
            for l in range(j):
                c[l] += w[j - 1] * lam ** l / factorial(l)
        return c

    def pdf(self, x, th):
        return np.exp(-self.lam_of(th) * x) * Pn.polyval(x, self.pdf_poly(th))

    def sf(self, x, th):
        return np.exp(-self.lam_of(th) * x) * Pn.polyval(x, self.sf_poly(th))

    def rvs(self, n, th, rng):
        lam = self.lam_of(th); w = self.weights(th)
        sh = rng.choice(np.arange(1, self.K + 1), size=n, p=w)
        return rng.gamma(sh, 1 / lam)

    # ---- exact order statistics (Theorem 1) ----
    def _mp_polys(self, th):
        """g and q (Proposition 1b) with multiprecision coefficients."""
        lam = mp.mpf(self.lam_of(th))
        a = [mp.mpf(v) for v in self.a]
        C = sum(a[j - 1] * mp.factorial(j - 1) / lam ** j for j in range(1, self.K + 1))
        w = [a[j - 1] * mp.factorial(j - 1) / lam ** j / C for j in range(1, self.K + 1)]
        g = [a[j] / C for j in range(self.K)]
        q = [mp.mpf(0)] * self.K
        for j in range(1, self.K + 1):
            for l in range(j):
                q[l] += w[j - 1] * lam ** l / mp.factorial(l)
        return g, q

    @staticmethod
    def _pmul(p, q):
        out = [mp.mpf(0)] * (len(p) + len(q) - 1)
        for i, x in enumerate(p):
            for j, y in enumerate(q):
                out[i + j] += x * y
        return out

    def os_terms_mp(self, th, m, i):
        g, q = self._mp_polys(th)
        ci = m * comb(m - 1, i - 1); out = {}
        for s_ in range(i):
            p = m - i + s_
            poly = list(g)
            for _ in range(p):
                poly = self._pmul(poly, q)
            b = p + 1
            coef = ci * comb(i - 1, s_) * (-1) ** s_
            cur = out.get(b, [mp.mpf(0)])
            L = max(len(cur), len(poly))
            cur = cur + [mp.mpf(0)] * (L - len(cur))
            out[b] = [cur[k] + (coef * poly[k] if k < len(poly) else 0) for k in range(L)]
        return out

    def os_terms(self, th, m, i):
        """float version (for densities/moments at moderate precision)."""
        with mp.workdps(50):
            T = self.os_terms_mp(th, m, i)
            return {b: np.array([float(c) for c in p]) for b, p in T.items()}

    def os_pdf(self, x, th, m, i):
        lam = self.lam_of(th)
        return sum(np.exp(-b * lam * x) * Pn.polyval(x, p) for b, p in self.os_terms(th, m, i).items())

    def os_moments(self, th, m, i):
        with mp.workdps(50):
            lam = mp.mpf(self.lam_of(th)); e1 = e2 = mp.mpf(0)
            for b, p in self.os_terms_mp(th, m, i).items():
                be = b * lam
                for a_, c in enumerate(p):
                    e1 += c * mp.factorial(a_ + 1) / be ** (a_ + 2)
                    e2 += c * mp.factorial(a_ + 2) / be ** (a_ + 3)
            return float(e1), float(e2 - e1 ** 2)

    # ---- Laplace transforms ----
    def _lt_terms(self, terms, lam, s):
        tot = mp.mpf(0)
        for b, p in terms.items():
            be = mp.mpf(b) * lam
            for a_, c in enumerate(p):
                if c != 0:
                    tot += mp.mpf(c) * mp.factorial(a_) / (be + s) ** (a_ + 1)
        return tot

    def lt_total(self, th, design):
        """Return s -> E exp(-s T); coefficients are built at the current mp precision.
        design = ('srs', n) or ('rss', m, r, Pmat or None)."""
        lam = mp.mpf(self.lam_of(th))
        if design[0] == 'srs':
            n = design[1]
            a = [mp.mpf(v) for v in self.a]
            C = sum(a[j - 1] * mp.factorial(j - 1) / lam ** j for j in range(1, self.K + 1))
            w = [a[j - 1] * mp.factorial(j - 1) / lam ** j / C for j in range(1, self.K + 1)]
            def L(s):
                return sum(w[j - 1] * (lam / (lam + s)) ** j for j in range(1, self.K + 1)) ** n
            return L
        _, m, r, Pm = design
        # precompute, per order statistic, {b*lam: [c_a * a!] in descending powers of u=1/(b lam+s)}
        pre = []
        for i in range(1, m + 1):
            T_i = self.os_terms_mp(th, m, i)
            pre.append([(mp.mpf(b) * lam, [c * mp.factorial(a_) for a_, c in enumerate(p)][::-1])
                        for b, p in T_i.items()])
        Pm = None if Pm is None else [[mp.mpf(float(v)) for v in row] for row in Pm]
        def lt_os(P_, s):
            tot = 0
            for be, cs in P_:
                u = 1 / (be + s)
                tot += u * mp.polyval(cs, u)       # sum_a c_a a! u^{a+1}
            return tot
        def L(s):
            Lj = [lt_os(P_, s) for P_ in pre]
            out = mp.mpf(1)
            for i in range(m):
                li = Lj[i] if Pm is None else sum(Pm[i][j] * Lj[j] for j in range(m))
                out *= li ** r
            return out
        return L

    @staticmethod
    def _key(design):
        return (design[0], design[1], design[2] if design[0] == 'rss' else 0,
                design[0] == 'rss' and design[3] is not None)

    def _raw(self, th, design, t, d, cdf):
        with mp.workdps(d):
            L = self.lt_total(th, design)
            f = (lambda s: L(s) / s) if cdf else L
            return mp.invertlaplace(f, mp.mpf(t), method='talbot')

    def calibrate(self, th, design, tol=1e-13):
        """Smallest working precision d at which Talbot inversions at d and d+15 digits agree
        to `tol` at five points spanning the law of T; the result is stored as d + 10."""
        key = self._key(design)
        if key in self._dps:
            return self._dps[key]
        n = self.n_of(design)
        mu, sd = n * self.mean(th), np.sqrt(n * self.var(th))
        pts = [max(mu * 1e-2, mu - 3.5 * sd), mu - sd, mu, mu + sd, mu + 5 * sd]
        for d in (30, 40, 50, 60, 75, 90, 110, 140, 180, 230, 290, 360, 450):
            if all(abs(self._raw(th, design, t, d, c) - self._raw(th, design, t, d + 15, c))
                   <= tol * max(1, abs(self._raw(th, design, t, d + 15, c)))
                   for t in pts for c in (True, False)):
                self._dps[key] = d + 10
                return d + 10
        raise RuntimeError(f"precision calibration failed for {design[:3]}")

    def _invert(self, th, design, t, cdf=True):
        d = self.calibrate(th, design)
        return float(self._raw(th, design, t, d, cdf))

    def cdf_T(self, t, th, design):
        """Exact CDF of T. If a value falls outside [0,1] (precision insufficient at this theta),
        the working precision for the design is raised and the value recomputed."""
        v = self._invert(th, design, t, cdf=True)
        tries = 0
        while (v < -1e-12 or v > 1 + 1e-12) and tries < 6:
            key = self._key(design)
            self._dps[key] = int(self._dps[key] * 1.4) + 10
            v = self._invert(th, design, t, cdf=True); tries += 1
        if v < -1e-12 or v > 1 + 1e-12:
            raise RuntimeError(f"cdf out of range {v}")
        return min(max(v, 0.0), 1.0)

    def pdf_T(self, t, th, design):
        return self._invert(th, design, t, cdf=False)

    # ---- estimators and exact inference (Section 4) ----
    def mu_inv(self, xbar):
        """Unique root of mean(th) = xbar (mean is strictly monotone in th)."""
        f = lambda lth: self.mean(np.exp(lth)) - xbar
        lo, hi = -2.0, 2.0
        while f(lo) * f(hi) > 0 and hi < 60:
            lo -= 2.0; hi += 2.0
        return float(np.exp(brentq(f, lo, hi, xtol=1e-13)))

    def n_of(self, design):
        return design[1] if design[0] == 'srs' else design[1] * design[2]

    def exact_ci(self, tobs, design, alpha=0.05, increasing=True, lo=1e-4, hi=1e4):
        """Invert the exact law of T. `increasing`: T stochastically increasing in the parameter."""
        F = lambda th: self.cdf_T(tobs, th, design)
        if increasing:   # F decreasing in th
            thL = brentq(lambda th: (1 - F(th)) - alpha / 2, lo, hi, xtol=1e-10)
            thU = brentq(lambda th: F(th) - alpha / 2, lo, hi, xtol=1e-10)
            med = brentq(lambda th: F(th) - 0.5, lo, hi, xtol=1e-10)
        else:
            thU = brentq(lambda th: (1 - F(th)) - alpha / 2, lo, hi, xtol=1e-10)
            thL = brentq(lambda th: F(th) - alpha / 2, lo, hi, xtol=1e-10)
            med = brentq(lambda th: F(th) - 0.5, lo, hi, xtol=1e-10)
        return thL, med, thU


def GOLD(k):
    return PE(np.ones(k), lambda th: 1.0 / th, name=f"GOLD(k={k})")


AKASH = PE(np.array([1.0, 0.0, 1.0]), lambda th: th, name="Akash")
