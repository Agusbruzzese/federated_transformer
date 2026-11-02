"""Site-level differential privacy for the federated system (report 35, M4). Shared VERBATIM by the real ClientApp (fl_system4) and
the in-process simulator (dpsim.py), so both apply exactly the same operations.

privatize: L2-clip a site's whole update to norm C, then add this site's SHARE of Gaussian noise (distributed DP: the noise is
           added by the sites before SecAgg+, so the server only ever sees the noisy SUM). Site k adds sd z*C*n_max/(n_k*sqrt(K));
           after SecAgg+'s weighting by n_k/n_max and the sum over K sites, the decoded mean carries noise sd z*C*n_max/sum(n)
           = z x (sensitivity n_max*C/sum(n) of the weighted mean to one site's clipped update, zero-out adjacency).
secagg_*:  exact in-process emulation of Flower 1.39's SecAgg+ arithmetic (weight factor, stochastic quantisation, integer sum,
           dequantisation, offset, division); pairwise/private masks cancel exactly in modular arithmetic and are omitted.
eps_*:     accountant. T rounds with full participation = T-fold composition of the Gaussian mechanism with multiplier z, which is
           exactly mu-GDP with mu = sqrt(T)/z (Dong, Roth, Su 2022); (eps, delta) by the exact GDP conversion; an RDP bound as a
           cross-check (Mironov 2017; must be >= the exact value). Replace-one adjacency doubles the sensitivity (mu x 2)."""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm as _N

NOISE_TAG = 4242


def privatize(delta, k, rnd, seed, n_k, n_max, K, C, z):
    """delta: list of float32 arrays (one site's update). Returns (new list of float32 arrays, info)."""
    nrm = float(np.sqrt(sum(float((d.astype(np.float64) ** 2).sum()) for d in delta))); info = dict(norm=nrm, scale=1.0, noise_sd=0.0)
    if C > 0:
        f = min(1.0, C / nrm) if nrm > 0 else 1.0; info["scale"] = f
        delta = [d * np.float32(f) for d in delta]
    if z > 0:
        assert C > 0, "noise needs a clipping bound"
        sd = z * C * n_max / (n_k * np.sqrt(K)); info["noise_sd"] = sd; rng = np.random.default_rng([seed, k, rnd, NOISE_TAG])
        delta = [d + (rng.standard_normal(d.shape) * sd).astype(np.float32) for d in delta]
    return [np.asarray(d, dtype=np.float32) for d in delta], info


def qrand_seed(seed, k, rnd):
    """np.random seed set by the site just before secaggplus_mod quantises (stochastic rounding), as in fl_system3."""
    return (seed * 1_000_003 + k * 1_009 + rnd) % (2 ** 32)


def secagg_site(delta, n_k, max_weight, clipping_range, Q, seed, k, rnd):
    """What secaggplus_mod computes from one site's update before masking: (q_ratio, quantised int arrays, n values clipped)."""
    from flwr.common.secure_aggregation.ndarrays_arithmetic import parameters_multiply
    from flwr.common.secure_aggregation.quantization import quantize
    q_ratio = round(n_k / max_weight * Q); dq_ratio = q_ratio / Q
    p = parameters_multiply(delta, dq_ratio); over = int(sum((np.abs(a) > clipping_range).sum() for a in p))
    np.random.seed(qrand_seed(seed, k, rnd))
    return q_ratio, quantize(p, clipping_range, Q), over


def secagg_decode(sites, clipping_range, Q, M):
    """What SecAggPlusWorkflow decodes from K sites: sum mod M, factor extraction, dequantise, offset, divide (same op order)."""
    from flwr.common.secure_aggregation.quantization import dequantize
    K = len(sites); q_total = sum(s[0] for s in sites)
    S = [np.zeros(a.shape, dtype=np.int64) for a in sites[0][1]]
    for _, qa, _ in sites:
        for i, a in enumerate(qa): S[i] += a
    S = [a % M for a in S]
    vec = dequantize(S, clipping_range, Q); offset = -(K - 1) * clipping_range; inv = Q / q_total
    for v in vec:
        v += offset
        v *= inv
    return vec


def secagg_clipping_range(C, z, K, base=0.5, n_sd=7.0):
    """Report 35: SecAgg+ value range = base + n_sd x (per-site noise sd after weighting, = z*C/sqrt(K) for the largest site)."""
    return base + n_sd * z * C / np.sqrt(K)


def eps_gdp(mu, delta):
    if mu <= 0: return 0.0
    f = lambda e: _N.cdf(-e / mu + mu / 2) - np.exp(e + _N.logcdf(-e / mu - mu / 2) - _N.logcdf(-e / mu + mu / 2)) * _N.cdf(-e / mu + mu / 2) - delta
    return float(brentq(f, 0.0, 5e6)) if f(0.0) > 0 else 0.0


def eps_rdp(T, z, delta, sens=1.0):
    a = np.concatenate([np.linspace(1.01, 10, 900), np.linspace(10, 1e6, 200000)])
    return float(np.min(T * a * sens ** 2 / (2 * z ** 2) + np.log(1 / delta) / (a - 1)))


def accountant(T, z, delta=1e-5):
    if z <= 0: return dict(z=z, T=T, delta=delta, mu_zero_out=None, eps_zero_out=float("inf"), eps_replace_one=float("inf"), eps_rdp_zero_out=float("inf"))
    mu = np.sqrt(T) / z
    return dict(z=z, T=T, delta=delta, mu_zero_out=float(mu), eps_zero_out=eps_gdp(mu, delta), eps_replace_one=eps_gdp(2 * mu, delta),
                eps_rdp_zero_out=eps_rdp(T, z, delta))
