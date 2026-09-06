# vol-surface-engine

A derivatives-pricing and volatility-surface research toolkit: a
semi-analytic Heston stochastic-volatility pricer (cross-validated
against independent Monte Carlo), SVI and SSVI implied-vol surface
fitting with explicit no-arbitrage diagnostics, and a delta-hedging
simulator that measures what fixed-volatility Black-Scholes hedging
actually costs you when the true world has stochastic volatility. Built
to answer three concrete questions, not to be a generic pricing
library:

1. Does independently fitting SVI to each option-expiry slice actually
   introduce calendar arbitrage in practice, and does a joint SSVI fit
   fix it?
2. Is the textbook sufficient condition for SSVI being arbitrage-free
   actually necessary in practice?
3. What does naive fixed-vol delta-hedging cost you -- in bias, not
   just variance -- when real volatility is stochastic and correlated
   with returns?

## Architecture

```
black_scholes.py   -- BSM pricing, Greeks, implied-vol solver (Brent's method,
                       bracketed against the no-arbitrage price band)
        |
        v
heston.py          -- semi-analytic Heston (1993) pricer via the "little trap"
                       characteristic function + numerical integration, PLUS
                       an independent full-truncation-Euler Monte Carlo
                       simulator used ONLY to cross-validate the pricer
        |
        v
data_gen.py         -- prices a synthetic option chain under Heston, inverts
                       every price back to Black-Scholes implied vol
        |
        v
svi.py              -- raw SVI (one fit per expiry) + SSVI (one joint fit
                        across all expiries), Durrleman butterfly-arbitrage
                        check, calendar-arbitrage grid check, SSVI's
                        Gatheral-Jacquier sufficient condition
        |
        v
surface.py           -- VolSurface: wires the chain -> IV -> SVI/SSVI pipeline
                        together and runs every arbitrage diagnostic
        |
        v
hedging.py            -- delta-hedges a short option position under TRUE
                        Heston dynamics using BS delta at a FIXED vol (the
                        realistic model-mismatch case), self-financing cash
                        accounting, transaction costs
```

## Why Heston generates the "market" data this project fits SVI to

This sandbox's outbound network access is a package-registry allowlist,
not the general internet (the same constraint the sibling
`orderbook-engine` project documents for LOBSTER data) -- a live listed-
options snapshot is unreachable here. Heston is used as a substitute
market generator, and this is not circular: Heston and SVI are two
independently motivated parameterizations of the same object (an
implied-vol surface), and the SVI/SSVI fitting code never sees Heston's
internal parameters -- it fits generic `(log-moneyness, total variance)`
points, exactly as it would fit real listed quotes. It also gives every
arbitrage check ground truth to check itself against: a Heston surface
is arbitrage-free by construction (one consistent risk-neutral model
generated every price), so if independently-fit SVI slices show
calendar arbitrage, that is the *fitting procedure* introducing it, not
real structure in the underlying data -- which is exactly what section
below finds. No code changes are needed to point this at a real chain:
`generate_synthetic_chain`'s output rows (`T, K, iv`) are the entire
interface `VolSurface` consumes.

## Build & test

```bash
pip install -r requirements.txt
python3 tests/test_black_scholes.py   # 51 checks: pricing, Greeks, IV round-trip, ill-conditioning
python3 tests/test_heston.py          # 28 checks: char-fn identities, MC cross-validation, parity
python3 tests/test_svi.py             # 12 checks: noiseless recovery, arbitrage checks, SSVI condition
python3 tests/test_hedging.py         # 7 checks: matched-model replication, cost drag
```

```bash
python3 scripts/run_surface_experiment.py    # SVI vs SSVI, arbitrage diagnostics (~5s)
python3 scripts/run_hedging_experiment.py    # rehedge-frequency / cost sweep (~3 min, 200k paths x 6-18 sweeps)
```

## Two real bugs found by testing, and what they looked like

**1. The Heston pricer could return a negative (arbitrage-violating)
price for far-OTM, short-dated options.** `_prob_j`'s numerical
integration originally capped the integration range at `u=100` "because
the integrand visibly decays by u~30-50 for most reasonable strikes."
It doesn't for all of them: `run_surface_experiment.py`'s 1-month
option chain silently lost several far-OTM quotes (`implied_vol`
correctly rejected the computed price as outside the no-arbitrage
band, rather than solving a bogus IV). Direct diagnosis, isolating one
such point (`K=165, T=0.083`, deep OTM call):

| integration upper limit | P1/P2 integral | error vs. converged value |
|---|---|---|
| 100 (original) | -1.5707226151 | 7.4e-5 |
| `np.inf` (fixed) | -1.5707963268 | ~1e-10 |

The true probability integral converges to exactly `-pi/2`, i.e. P1=P2=0
(correctly: this option is worth essentially nothing). At the finite
cutoff, the truncated tail contributes a *tiny absolute* error --
but the option's true value is *also* tiny, so the relative error is
large enough to flip the sign of `S0*P1 - K*disc*P2` from ~0 to
negative. **Fix**: integrate to `np.inf` (`scipy.integrate.quad` handles
an infinite upper limit by adapting the range itself) instead of
guessing a cutoff -- and it's empirically *faster* here too (confirmed:
converges to 1e-10 accuracy in under 10ms for the case above, vs. a
visibly wrong result at a finite cutoff of 100).

**2. The delta-hedging simulator's cash account didn't earn interest.**
`test_hedging.py`'s baseline sanity check -- collapse Heston to
constant-variance GBM (`xi -> 0`) exactly matching the hedge model, so
delta-hedging should replicate the option payoff almost perfectly, mean
PnL ~0 -- failed at every rehedge frequency tested, stuck around
**+0.49** even at near-continuous (every-step) rehedging. That it didn't
shrink with rehedge frequency was the tell: discretization error
shrinks toward 0 as rehedging gets more frequent; this didn't move.
The magnitude matched a financing-cost calculation (delta notional x
`r` x `T`), which is exactly what a static (non-interest-accruing) cash
balance misses. **Fix**: cash now compounds at the risk-free rate
between rehedge dates (a proper self-financing-portfolio accounting).
Rerunning the same check after the fix:

| rehedge every N steps | mean PnL (before fix) | mean PnL (after fix) |
|---|---|---|
| 50 | +0.4876 | -0.0049 |
| 10 | +0.5071 | +0.0146 |
| 2 | +0.5079 | +0.0154 |
| 1 | +0.4979 | +0.0053 |

Both are now checked in `tests/test_hedging.py` and `tests/test_heston.py`
so they can't silently regress.

## Finding 1: independent per-slice SVI fits DO introduce calendar
arbitrage, and it's invisible to a per-slice-only check

Setup: a Heston-generated chain across 6 expiries -- including a
closely-spaced weekly/monthly pair (30 and 35 calendar days) -- with
+/-0.5 vol-point Gaussian noise added to every implied vol (a
noiseless arbitrage-free surface fits without introducing arbitrage
almost by definition; noise is what makes this the realistic case, and
matches how real quote noise behaves).

```
Raw SVI, independent per-expiry fits:

| T (yrs) | RMSE (vol pts) | butterfly arb-free | min g(k) |
|---|---|---|---|
| 0.083 | 0.1777 | True | 0.1018 |
| 0.095 | 0.3401 | True | 0.3232 |
| 0.250 | 0.3018 | True | 0.3183 |
| 0.500 | 0.3951 | True | 0.3130 |
| 1.000 | 0.4460 | True | 0.3243 |
| 2.000 | 0.4420 | True | 0.3160 |
```

**Every single slice is individually butterfly-arb-free** (`min_g(k) >
0` everywhere). And yet: **11 calendar-arbitrage violations** between
the T=0.083 and T=0.095 slices (`k=0.5: w(T=0.083)=0.005331 >
w(T=0.095)=0.002798` -- total variance *decreased* with maturity, a
real static arbitrage: a calendar spread here would have negative
cost). This is the actual point of building a calendar-arbitrage check
separate from the butterfly check: **a slice-by-slice arbitrage check
would have reported "all clear" on a surface that isn't** -- calendar
arbitrage is fundamentally a cross-slice property, and closely-spaced
expiries (real weekly-options term structures) are exactly where noise
is most likely to flip the ordering, since their true total variances
are closest together.

## Finding 2: SSVI's textbook condition is sufficient, not necessary --
and that distinction is not just academic

Jointly fitting SSVI (one shared `rho, eta, gamma` across all 6
expiries, unconstrained least-squares) to the same noisy data:

```
SSVI joint fit: rho=-0.6717, eta=1.4004, gamma=0.3127
eta*(1+|rho|) = 2.3410   (Gatheral-Jacquier sufficient condition requires <= 2)
Calendar-arbitrage-free by the Gatheral-Jacquier sufficient condition: False
Direct grid check on the fitted SSVI surface: 0 violations, worst gap = 0.000000
```

The unconstrained fit lands *outside* the textbook sufficient condition
for global calendar-arbitrage-freedom (`eta*(1+|rho|) = 2.34 > 2`) --
and yet the direct grid check (the same methodology that found 11
violations in the raw-SVI case) finds **zero** violations on the actual
fitted surface. Both statements are correct simultaneously: "sufficient,
not necessary" means exactly this -- failing the condition doesn't
prove arbitrage exists, it just stops proving it doesn't. Reporting
only the textbook condition here would have been a false alarm; the
direct check is what actually matters, and is why `surface.py` exposes
both rather than trusting the closed-form condition alone.

The cost of that (empirical) guarantee is real, though: SSVI's RMSE is
markedly worse than raw SVI at every expiry (4.8 vs 0.18 vol points at
the shortest expiry, 0.65 vs 0.09 at the longest) -- one shared
`(rho, eta, gamma)` across 6 expiries is a much more constrained model
than 6 independent 5-parameter fits. This is the real, practitioner-
relevant tradeoff the project set out to quantify: flexibility (and
better per-slice fit) vs. a defensible no-arbitrage guarantee.

## Finding 3: fixed-vol delta-hedging has a real BIAS under stochastic
vol, not just higher variance -- and it doesn't shrink away

Short 1 ATM call (`K=100, T=0.5`), true dynamics Heston
(`kappa=1.5, theta=0.045, xi=0.55, rho=-0.75, v0=0.045` -- realistic
equity-index-like leverage effect and vol-of-vol), hedged with
Black-Scholes delta at a single fixed vol (`sqrt(v0)=21.2%`, never
updated):

| rehedge every N steps | mean PnL | std PnL |
|---|---|---|
| 125 (~quarterly) | 0.4830 | 3.7759 |
| 50 | 0.4836 | 2.8707 |
| 25 | 0.4862 | 2.4274 |
| 10 | 0.4865 | 2.0847 |
| 5 | 0.4851 | 1.9552 |
| 1 (daily) | 0.4864 | 1.8369 |

Std PnL shrinks with rehedge frequency exactly as expected
(discretization error). **Mean PnL does not move at all** -- it sits at
+0.48-0.49 regardless of rehedge frequency, 100+ standard errors from
zero (stderr ~0.004-0.008 per row). This is not a bug: the matched-model
control above (`xi -> 0`, same hedge vol as true vol) drives mean PnL to
~0 at every frequency, isolating the cause to the mismatch between a
fixed-vol hedge and true stochastic, leverage-correlated vol -- exactly
the "volatility risk premium" / correlation-convexity effect real desks
build stochastic-vol-aware Greeks specifically to hedge away. A
naive reading of "hedging error shrinks as I rehedge more" would miss
this entirely, since it's the mean, not the variance, doing the
interesting thing.

Transaction costs (daily rehedging) flip that positive mean negative
well before the highest cost tested:

| cost (bps/trade) | mean PnL | std PnL |
|---|---|---|
| 0.0 | 0.4864 | 1.8369 |
| 1.0 | 0.4388 | 1.8485 |
| 5.0 | 0.2486 | 1.8960 |
| 10.0 | 0.0107 | 1.9587 |
| 25.0 | -0.7027 | 2.1646 |
| 50.0 | -1.8919 | 2.5529 |

And the frequency/cost tradeoff is genuinely two-sided at a realistic
5bps: rehedging more often still reduces variance monotonically
(3.78 -> 1.90) but now *also* erodes the mean (0.44 -> 0.25) -- the
textbook "just rehedge more" answer is no longer free once cost is on
the table, which is exactly the tradeoff a real hedging desk has to
solve for, not a side note.

## Honest limitations

- **No real market data** -- see "Why Heston generates the market data"
  above for why, and that swapping in a real chain requires no code
  changes to `VolSurface`, only different input rows.
- **The full-truncation Euler MC scheme is a known-biased
  discretization** (Lord, Koekkoek, Van Dijk 2010) -- it substitutes
  `v^+ = max(v,0)` into the diffusion whenever the discretized variance
  dips below 0, which happens often here since several tested parameter
  sets deliberately violate the Feller condition. It's the standard
  simplest fix for that regime, not an exact scheme; the MC-vs-analytic
  cross-validation in `test_heston.py` accounts for this by using a
  4-standard-error tolerance band, not an exact match.
- **SSVI's ATM term structure `theta(T)` is projected to be
  non-decreasing via a simple cumulative max**, not fit jointly with
  `(rho, eta, gamma)` -- a real (noisy) ATM curve could need a more
  principled monotonic smoother.
- **Factor loadings in the hedging experiment are static per run** --
  the hedge vol is fixed at inception and never updated from realized
  or implied data during the simulation, which is deliberate (that's
  the scenario being measured), but a vol-surface-aware hedge (updating
  Black-Scholes vol from the current ATM implied vol each rehedge date,
  or hedging vega too) is the natural next comparison and isn't built
  here.
- **The calendar-arbitrage grid check is exactly that -- a grid**,
  evaluated at 41 points over a fixed log-moneyness range; it's a
  strong diagnostic (and the one that actually caught Finding 1), not a
  formal proof for all `k`.

## What I'd build next

- A vol-surface-aware hedge (recompute delta from the current fitted
  SVI/SSVI slice's ATM vol each rehedge date, rather than one fixed
  vol) and re-measure the mean-bias finding above against it directly.
- Constrained SSVI fitting (enforce the Gatheral-Jacquier condition
  as an explicit optimizer constraint rather than checking it
  post-hoc) to see how much RMSE that costs on top of what's already
  measured here.
- A term-structure-aware SSVI variant (theta(T) fit jointly with the
  shared parameters via a monotonic spline, rather than a per-slice ATM
  variance plus cumulative-max projection).
- Real listed-options data, the moment it's reachable from wherever
  this runs next -- no code changes needed, per the note above.
