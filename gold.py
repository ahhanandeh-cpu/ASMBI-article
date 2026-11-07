"""GOLD(theta,k) distribution: density, survival, NEF quantities, RSS Fisher information."""
import numpy as np
from scipy.special import gammaincc, gamma as G, zeta
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar


def Gk(t, k):
    return sum(G(j) * t ** (j - 1) for j in range(1, k + 1))


def mu(t, k):
    return sum(G(j + 1) * t ** j for j in range(1, k + 1)) / Gk(t, k)


def m2(t, k):
    return sum(G(j + 2) * t ** (j + 1) for j in range(1, k + 1)) / Gk(t, k)


def var(t, k):
    return m2(t, k) - mu(t, k) ** 2


def info_srs(t, k):
    """Per-observation Fisher information for theta under SRS: Var(X)/theta^4."""
    return var(t, k) / t ** 4


def pdf(x, t, k):
    x = np.asarray(x, float)
    h = sum(x ** (j - 1) for j in range(1, k + 1))
    return h * np.exp(-x / t) / (t * Gk(t, k))


def logpdf(x, t, k):
    x = np.asarray(x, float)
    h = sum(x ** (j - 1) for j in range(1, k + 1))
    return np.log(h) - x / t - np.log(t) - np.log(Gk(t, k))


def sf(x, t, k):
    x = np.asarray(x, float)
    A = sum(t ** (j - 1) * G(j) * gammaincc(j, x / t) for j in range(1, k + 1))
    return A / Gk(t, k)


def B(x, t, k):
    """int_x^inf u f(u) du = sum_j theta^j Gamma(j+1, x/theta) / Gamma_k(theta)."""
    x = np.asarray(x, float)
    return sum(t ** j * G(j + 1) * gammaincc(j + 1, x / t) for j in range(1, k + 1)) / Gk(t, k)


def dF(x, t, k):
    """dF/dtheta = -(1/theta^2) [B(x) - mu S(x)]  (Lemma 1)."""
    return -(B(x, t, k) - mu(t, k) * sf(x, t, k)) / t ** 2


def dF_thm1(x, t, k):
    """dF/dtheta from Theorem 1 of the original manuscript (cross-check)."""
    x = np.asarray(x, float)
    gk = Gk(t, k)
    gkp = sum((j - 1) * G(j) * t ** (j - 2) for j in range(2, k + 1))
    dA = sum((j - 1) * t ** (j - 2) * G(j) * gammaincc(j, x / t) for j in range(1, k + 1)) \
        + np.exp(-x / t) / t ** 2 * sum(x ** j for j in range(1, k + 1))
    dS = dA / gk - sf(x, t, k) * gkp / gk
    return -dS


def Delta(t, k):
    """E[(dF/dtheta)^2 / (F S)]."""
    def integrand(x):
        S = sf(x, t, k)
        F = 1.0 - S
        if F <= 0 or S <= 0:
            return 0.0
        return dF(x, t, k) ** 2 / (F * S) * pdf(x, t, k)
    up = mu(t, k) + 40 * np.sqrt(var(t, k))
    pts = [mu(t, k)]
    val, _ = quad(integrand, 0, up, points=pts, limit=400, epsabs=0, epsrel=1e-10)
    return val


def kappa(t, k):
    return Delta(t, k) / info_srs(t, k)


def are_mle(t, k, m):
    return 1 + (m - 1) * kappa(t, k)


def os_moments(t, k, m):
    """Means and variances of the m order statistics of a sample of size m."""
    from math import comb
    mus, vs = [], []
    up = mu(t, k) + 40 * np.sqrt(var(t, k))
    for i in range(1, m + 1):
        c = m * comb(m - 1, i - 1)
        def dens(x):
            S = sf(x, t, k)
            return c * (1 - S) ** (i - 1) * S ** (m - i) * pdf(x, t, k)
        e1 = quad(lambda x: x * dens(x), 0, up, limit=400, epsrel=1e-11)[0]
        e2 = quad(lambda x: x * x * dens(x), 0, up, limit=400, epsrel=1e-11)[0]
        mus.append(e1)
        vs.append(e2 - e1 ** 2)
    return np.array(mus), np.array(vs)


def are_me(t, k, m):
    """ARE of the RSS moment estimator vs the SRS-MLE (perfect ranking)."""
    mus, vs = os_moments(t, k, m)
    return var(t, k) / vs.mean()


def kappa_gamma(k):
    """kappa for Gamma(shape k, scale theta) with k known (scale family: free of theta)."""
    from scipy.stats import gamma as gd
    I = k  # theta=1: info for scale = k
    def integrand(u):
        F = gd.cdf(u, k); S = gd.sf(u, k)
        if F <= 0 or S <= 0:
            return 0.0
        dFt = -u * gd.pdf(u, k)  # dF/dtheta at theta=1
        return dFt ** 2 / (F * S) * gd.pdf(u, k)
    return quad(integrand, 0, k + 60 * np.sqrt(k), limit=400, epsrel=1e-11)[0] / I


# ---------------- estimators ----------------
def mu_inv(xbar, k):
    """Solve mu(theta)=xbar (SRS-MLE when xbar is an SRS mean; RSS-ME for an RSS mean)."""
    if k == 1:
        return xbar
    if k == 2:
        return ((xbar - 1) + np.sqrt(xbar ** 2 + 6 * xbar + 1)) / 4
    lo, hi = 1e-8, xbar
    return brentq(lambda t: mu(t, k) - xbar, lo, hi, xtol=1e-12)


def rss_negloglik(t, x, ranks, m, k):
    S = sf(x, t, k)
    S = np.clip(S, 1e-300, 1 - 1e-16)
    F = 1 - S
    return -np.sum((ranks - 1) * np.log(F) + (m - ranks) * np.log(S) + logpdf(x, t, k))


def rss_mle(x, ranks, m, k, start):
    res = minimize_scalar(rss_negloglik, bounds=(start / 4, start * 4), method="bounded",
                          args=(x, ranks, m, k), options={"xatol": 1e-9})
    return res.x


def rgold(n, t, k, rng):
    w = np.array([G(j) * t ** (j - 1) for j in range(1, k + 1)])
    w /= w.sum()
    shapes = rng.choice(np.arange(1, k + 1), size=n, p=w)
    return rng.gamma(shapes, t)


if __name__ == "__main__":
    # sanity checks
    for k in (2, 3, 4):
        t = 1.1
        xs = np.linspace(0.1, 6, 7)
        print("dF lemma vs thm1 max abs diff", k, np.max(np.abs(dF(xs, t, k) - dF_thm1(xs, t, k))))
        print("pdf integrates", quad(lambda x: pdf(x, t, k), 0, np.inf)[0],
              " mean", quad(lambda x: x * pdf(x, t, k), 0, np.inf)[0], mu(t, k))
        h = 1e-5
        num = (sf(2.0, t + h, k) - sf(2.0, t - h, k)) / (2 * h)
        print("numeric dS", num, -dF(2.0, t, k))
    print("kappa exp k=1:", kappa(1.0, 1), kappa(3.0, 1), " theory 2(zeta3-1)=", 2 * (zeta(3) - 1))
