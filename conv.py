"""Fast law of the RSS/SRS total by FFT convolution of order-statistic densities (midpoint masses).
Used for the design study; verified against exact Talbot inversion."""
import numpy as np
from math import comb
from scipy.special import gammaincc, gammaln


def os_masses(fam, th, m, h, xmax):
    lam = fam.lam_of(th); w = fam.weights(th)
    x = (np.arange(int(np.ceil(xmax / h))) + 0.5) * h          # midpoints
    # stable survival and density of the Erlang mixture
    S = sum(w[j - 1] * gammaincc(j, lam * x) for j in range(1, fam.K + 1))
    f = sum(w[j - 1] * np.exp(j * np.log(lam) + (j - 1) * np.log(x) - lam * x - gammaln(j)) for j in range(1, fam.K + 1))
    F = 1 - S
    if m == 1:
        return [f * h]
    return [m * comb(m - 1, i - 1) * F ** (i - 1) * S ** (m - i) * f * h for i in range(1, m + 1)]


def total_cdf(fam, th, design, h=None):
    """Return (t, cdf) on a grid; t are the right edges of cells."""
    if design[0] == 'srs':
        m, r = 1, design[1]
    else:
        m, r = design[1], design[2]
    n = m * r
    mu, sd = fam.mean(th), np.sqrt(fam.var(th))
    xmax = mu + 16 * sd
    if h is None:
        h = sd / 400
    ms = os_masses(fam, th, m, h, xmax)
    L = len(ms[0]); N = 1 << int(np.ceil(np.log2(L * n + 1)))
    prod = np.ones(N // 2 + 1, dtype=complex)
    for p in ms:
        prod *= np.fft.rfft(p, N) ** r
    mass = np.fft.irfft(prod, N)[: L * n]
    mass = np.clip(mass, 0, None)
    # the j-th mass sits at (j + n/2) h ; cumulative mass up to that point, with half-mass correction
    pos = (np.arange(len(mass)) + n / 2) * h
    cdf = np.cumsum(mass) - mass / 2
    return pos, cdf


def cdf_at(fam, th, design, t, h=None):
    pos, cdf = total_cdf(fam, th, design, h)
    return np.interp(t, pos, cdf)
