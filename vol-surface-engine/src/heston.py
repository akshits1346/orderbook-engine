"""
Heston (1993) stochastic-volatility model: semi-analytic pricing via the
characteristic function, and a Monte Carlo simulator used as an
independent cross-check on that pricer (see tests/test_heston.py).

    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW1_t
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW2_t
    corr(dW1_t, dW2_t) = rho

Two independent implementations of "the price of a European option
under these dynamics" (a Fourier-integral formula and a Monte Carlo
simulation) agreeing to within MC standard error is the actual proof
that the semi-analytic pricer is correct -- not just "the code runs and
returns a plausible-looking number for one input."
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.integrate import quad


@dataclass
class HestonParams:
    kappa: float  # mean-reversion speed of variance
    theta: float  # long-run variance
    xi: float     # vol-of-vol
    rho: float    # corr(dW1, dW2), typically negative (leverage effect)
    v0: float     # initial variance

    def feller_ratio(self):
        """2*kappa*theta / xi^2 >= 1 keeps v_t > 0 a.s. (Feller condition).
        Not enforced -- the MC simulator below uses full truncation
        precisely so parameter sets that violate it (common in practice,
        since real equity smiles often need xi large) don't blow up."""
        return 2 * self.kappa * self.theta / (self.xi ** 2)


def _char_function(u, S0, r, q, T, p: HestonParams):
    """
    Characteristic function of ln(S_T), "little trap" formulation
    (Albrecher, Mayer, Schoutens, Tistaert 2007): algebraically identical
    to Heston's original but with the branch of the complex sqrt/log
    chosen so it stays continuous in T for all model parameters. The
    textbook formula has spurious discontinuities that make quad()
    integrate garbage for long maturities / high vol-of-vol -- this
    reformulation is the standard fix, not an approximation.
    """
    kappa, theta, xi, rho, v0 = p.kappa, p.theta, p.xi, p.rho, p.v0
    x0 = math.log(S0)
    a = kappa * theta
    b = kappa - rho * xi * 1j * u

    d = np.sqrt(b ** 2 + (xi ** 2) * (u ** 2 + 1j * u))
    g = (b - d) / (b + d)  # "little trap" g, not 1/g

    exp_dT = np.exp(-d * T)
    C = (r - q) * 1j * u * T + (a / xi ** 2) * (
        (b - d) * T - 2 * np.log((1 - g * exp_dT) / (1 - g))
    )
    D = ((b - d) / xi ** 2) * ((1 - exp_dT) / (1 - g * exp_dT))

    return np.exp(C + D * v0 + 1j * u * x0)


def _prob_j(j, S0, K, T, r, q, p: HestonParams):
    """P1 or P2 from Heston (1993) eq. 18, via numerical integration of
    the characteristic function. j=1 -> P1 (uses phi shifted by -i),
    j=2 -> P2 (uses phi directly)."""
    x0 = math.log(S0)
    lnK = math.log(K)

    def integrand(u):
        if j == 1:
            phi = _char_function(u - 1j, S0, r, q, T, p) / _char_function(-1j, S0, r, q, T, p)
        else:
            phi = _char_function(u, S0, r, q, T, p)
        val = np.exp(-1j * u * lnK) * phi / (1j * u)
        return val.real

    # integrand has a removable singularity at u=0 (limit = finite); quad
    # handles this fine starting the interval just above 0.
    #
    # The upper limit MUST be np.inf, not a finite cutoff, despite the
    # integrand visibly decaying by u~30-50 for most reasonable strikes:
    # for far-OTM, short-maturity options the true P1/P2 (and hence the
    # price) are correctly close to 0, and a finite cutoff (this used a
    # hardcoded 100 originally) leaves an uncancelled tail contribution
    # that's a tiny absolute integration error but a LARGE relative error
    # on a near-zero probability -- enough to flip heston_price negative,
    # i.e. an outright arbitrage violation in the pricer's own output.
    # Caught by run_surface_experiment.py: deep-OTM, 1-month synthetic
    # quotes were silently dropped (implied_vol correctly rejected the
    # negative price as outside the no-arbitrage band) rather than
    # priced. quad(..., np.inf) lets SciPy adapt the integration range
    # instead of guessing one, and is empirically faster here too, not
    # just more accurate (verified: converges to -pi/2 to 1e-10 in <10ms
    # for the case above, vs. a visibly wrong 4th-significant-digit
    # result at a finite cutoff of 100).
    integral, _ = quad(integrand, 1e-10, np.inf, limit=200, epsabs=1e-12, epsrel=1e-10)
    return 0.5 + integral / math.pi


def heston_price(S0, K, T, r, p: HestonParams, option_type="call", q=0.0):
    """Semi-analytic European option price under Heston (1993)."""
    P1 = _prob_j(1, S0, K, T, r, q, p)
    P2 = _prob_j(2, S0, K, T, r, q, p)
    call = S0 * math.exp(-q * T) * P1 - K * math.exp(-r * T) * P2
    if option_type == "call":
        return call
    if option_type == "put":
        # put-call parity, not a second integral -- also doubles as a
        # correctness check in tests/test_heston.py
        return call - S0 * math.exp(-q * T) + K * math.exp(-r * T)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def simulate_paths(S0, r, p: HestonParams, T, n_steps, n_paths, q=0.0, seed=None, antithetic=True):
    """
    Full-truncation Euler scheme (Lord, Koekkoek, Van Dijk 2010): the
    simplest Heston discretization that stays usable when the Feller
    condition fails (v_t can dip below 0 in discrete time even for
    parameters where the true continuous-time process never hits 0) --
    it substitutes v^+ = max(v, 0) into the diffusion coefficients and
    the drift, rather than reflecting or absorbing at 0. This biases
    the discretization (a known, documented property of the scheme, not
    a bug), but it doesn't crash on sqrt(negative), which is the
    tradeoff this project needs since some SVI/SSVI experiments below
    intentionally probe high-vol-of-vol regimes.

    Returns log-price paths so callers can slice at any point;
    S = S0 * exp(path).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    half = n_paths // 2 if antithetic else n_paths
    total = half * 2 if antithetic else n_paths

    z1 = rng.standard_normal((half, n_steps))
    z2 = rng.standard_normal((half, n_steps))
    if antithetic:
        z1 = np.concatenate([z1, -z1], axis=0)
        z2 = np.concatenate([z2, -z2], axis=0)

    rho = p.rho
    zv = rho * z1 + math.sqrt(1 - rho ** 2) * z2  # correlated driver for variance

    x = np.full(total, math.log(S0), dtype=np.float64)
    v = np.full(total, p.v0, dtype=np.float64)
    x_paths = np.empty((total, n_steps + 1))
    v_paths = np.empty((total, n_steps + 1))
    x_paths[:, 0] = x
    v_paths[:, 0] = v

    for t in range(n_steps):
        v_pos = np.maximum(v, 0.0)
        sqrt_v = np.sqrt(v_pos)
        x = x + (r - q - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * z1[:, t]
        v = v + p.kappa * (p.theta - v_pos) * dt + p.xi * sqrt_v * sqrt_dt * zv[:, t]
        x_paths[:, t + 1] = x
        v_paths[:, t + 1] = v

    return x_paths, v_paths


def mc_price(S0, K, T, r, p: HestonParams, option_type="call", q=0.0,
             n_steps=200, n_paths=200_000, seed=None):
    """Monte Carlo price + standard error, for cross-validating heston_price."""
    x_paths, _ = simulate_paths(S0, r, p, T, n_steps, n_paths, q=q, seed=seed)
    S_T = np.exp(x_paths[:, -1])
    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    else:
        payoff = np.maximum(K - S_T, 0.0)
    disc_payoff = math.exp(-r * T) * payoff
    price = disc_payoff.mean()
    stderr = disc_payoff.std(ddof=1) / math.sqrt(len(disc_payoff))
    return price, stderr
