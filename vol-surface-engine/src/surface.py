"""
VolSurface: wires the option chain -> implied vol -> SVI/SSVI pipeline
together, and runs the arbitrage diagnostics across the whole surface.
"""
import math
import numpy as np

from svi import (
    fit_svi_slice, raw_svi_total_variance, check_butterfly_arbitrage,
    fit_ssvi_surface, ssvi_total_variance, ssvi_calendar_arbitrage_free,
)
from data_gen import rows_by_expiry


class VolSurface:
    def __init__(self, rows):
        """rows: output of data_gen.generate_synthetic_chain (or any list
        of dicts with T, K, F, k, iv)."""
        self.by_expiry = rows_by_expiry(rows)
        self.expiries = list(self.by_expiry.keys())

    def _slice_arrays(self, T):
        rows = self.by_expiry[T]
        k = np.array([r["k"] for r in rows])
        iv = np.array([r["iv"] for r in rows])
        w = iv ** 2 * T
        return k, w, iv

    def fit_raw_svi(self):
        """Fit one raw SVI per expiry independently. Returns dict T -> (params, rmse_vol)."""
        self.svi_params = {}
        for T in self.expiries:
            k, w, iv = self._slice_arrays(T)
            params, fitted_w = fit_svi_slice(k, w)
            fitted_iv = np.sqrt(np.maximum(fitted_w, 0) / T)
            rmse_vol = float(np.sqrt(np.mean((fitted_iv - iv) ** 2)))
            self.svi_params[T] = (params, rmse_vol)
        return self.svi_params

    def butterfly_report(self, k_grid=None):
        """Per-expiry butterfly-arbitrage check on the fitted raw SVI slices."""
        if k_grid is None:
            k_grid = np.linspace(-1.5, 1.5, 301)
        report = {}
        for T, (params, rmse_vol) in self.svi_params.items():
            ok, min_g = check_butterfly_arbitrage(params, k_grid)
            report[T] = {"arb_free": ok, "min_g": min_g, "rmse_vol": rmse_vol}
        return report

    def calendar_report(self, k_common=None, n_k=41):
        """
        Checks whether raw-SVI total variance w(k, T) is non-decreasing
        in T at each fixed k across a common log-moneyness range shared
        by every expiry -- the calendar no-arbitrage condition. Returns
        (violation_count, worst_violation, detail_rows) where detail_rows
        lists every (k, T_i, T_{i+1}, w_i, w_{i+1}) pair that violates it.
        """
        if k_common is None:
            k_common = np.linspace(-0.5, 0.5, n_k)
        Ts = sorted(self.svi_params.keys())
        w_grid = {T: raw_svi_total_variance(k_common, self.svi_params[T][0]) for T in Ts}

        violations = []
        for i in range(len(Ts) - 1):
            T1, T2 = Ts[i], Ts[i + 1]
            w1, w2 = w_grid[T1], w_grid[T2]
            bad = w2 < w1 - 1e-9
            for j in np.where(bad)[0]:
                violations.append({
                    "k": float(k_common[j]), "T1": T1, "T2": T2,
                    "w1": float(w1[j]), "w2": float(w2[j]), "gap": float(w1[j] - w2[j]),
                })
        worst = max((v["gap"] for v in violations), default=0.0)
        return len(violations), worst, violations

    def fit_ssvi(self):
        """Joint arbitrage-aware fit. ATM total variance theta(T) is taken
        directly from each slice's own (interpolated) ATM implied vol --
        NOT from the raw SVI fit, so this fit is independent of fit_raw_svi()."""
        Ts = sorted(self.expiries)
        thetas, ks, ws = [], [], []
        for T in Ts:
            k, w, iv = self._slice_arrays(T)
            atm_w = float(np.interp(0.0, k, w))  # ATM = k=0 (forward, not spot)
            thetas.append(atm_w)
            ks.append(k)
            ws.append(w)
        # enforce theta(T) non-decreasing (a real ATM curve from a
        # no-arbitrage model is; noisy/estimated ATM levels might not be --
        # isotonic projection via cumulative max is the standard fix)
        thetas = np.maximum.accumulate(np.array(thetas))
        rho, eta, gamma_power = fit_ssvi_surface(thetas, ks, ws)
        self.ssvi_theta = dict(zip(Ts, thetas))
        self.ssvi_params = (rho, eta, gamma_power)
        return rho, eta, gamma_power, dict(zip(Ts, thetas))

    def ssvi_fit_quality(self):
        """RMSE in vol terms of the SSVI joint fit, per expiry -- for
        comparing against the raw-SVI per-slice RMSE."""
        rho, eta, gamma_power = self.ssvi_params
        report = {}
        for T in self.expiries:
            k, w, iv = self._slice_arrays(T)
            theta = self.ssvi_theta[T]
            model_w = ssvi_total_variance(k, theta, rho, eta, gamma_power)
            model_iv = np.sqrt(np.maximum(model_w, 0) / T)
            rmse_vol = float(np.sqrt(np.mean((model_iv - iv) ** 2)))
            report[T] = rmse_vol
        return report

    def ssvi_calendar_arb_free(self):
        """The Gatheral-Jacquier condition is SUFFICIENT, not necessary --
        an unconstrained least-squares fit can land outside it while still
        being empirically arbitrage-free (or not). Use ssvi_calendar_report
        below for the direct, model-free grid check; this is the cheap
        theoretical shortcut, not the final word."""
        rho, eta, gamma_power = self.ssvi_params
        return ssvi_calendar_arbitrage_free(rho, eta, gamma_power)

    def ssvi_calendar_report(self, k_common=None, n_k=41):
        """
        Direct, grid-based calendar-arbitrage check on the FITTED SSVI
        surface (same methodology as calendar_report for raw SVI) --
        doesn't rely on the Gatheral-Jacquier sufficient condition holding,
        so it's the real test of whether the joint fit actually delivered
        what it was supposed to.
        """
        if k_common is None:
            k_common = np.linspace(-0.5, 0.5, n_k)
        rho, eta, gamma_power = self.ssvi_params
        Ts = sorted(self.ssvi_theta.keys())
        w_grid = {T: ssvi_total_variance(k_common, self.ssvi_theta[T], rho, eta, gamma_power)
                  for T in Ts}

        violations = []
        for i in range(len(Ts) - 1):
            T1, T2 = Ts[i], Ts[i + 1]
            w1, w2 = w_grid[T1], w_grid[T2]
            bad = w2 < w1 - 1e-9
            for j in np.where(bad)[0]:
                violations.append({
                    "k": float(k_common[j]), "T1": T1, "T2": T2,
                    "w1": float(w1[j]), "w2": float(w2[j]), "gap": float(w1[j] - w2[j]),
                })
        worst = max((v["gap"] for v in violations), default=0.0)
        return len(violations), worst, violations
