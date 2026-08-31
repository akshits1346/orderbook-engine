# orderbook-engine

A price-time-priority limit order book engine in C++, with a Python
bridge (pybind11) for strategy research and backtesting. Built to
answer a specific question: does Avellaneda-Stoikov's inventory-aware
quoting actually beat a naive symmetric market maker, when both are
tested against the same replayed order flow with a real fill model?

## Architecture

```
LOBSTER message file (CSV)
        |
        v
lobster_parser (C++)  --  parses into MessageEvent structs
        |
        v
replay layer (C++)     --  maps LOBSTER type codes to book mutations
        |
        v
OrderBook (C++)         --  price-time-priority matching engine,
                             O(log n) add/cancel/execute, queue
                             position tracking via size_ahead_of()
        |
        v
pybind11 bindings       --  exposes OrderBook + replay to Python
        |
        v
ReplayBacktest (Python) --  event-by-event replay with a documented
                             queue-position fill model for OUR OWN
                             resting orders (see backtest/replay_backtest.py)
        |
        v
strategies (Python)     --  NaiveMarketMaker (baseline) and
                             AvellanedaStoikov (inventory-aware),
                             sharing one interface so either can run
                             through the same backtest
        |
        v
calibrate.py (Python)   --  estimates sigma (volatility) and kappa
                             (fill-rate decay vs. distance) from actual
                             replayed data, rather than guessing them
```

## Why a separate C++ core and Python layer

The matching engine is correctness-critical -- it's validated against
a hand-traced reconstruction and against LOBSTER's exact message
semantics, so it stays in C++, compiled once and tested thoroughly.
The strategy layer is where iteration actually happens (trying
different quoting logic, re-running comparisons), so it's in Python,
calling into the validated C++ core through pybind11. This mirrors how
production trading systems are typically split: a hot, correctness-
critical path in a compiled language, a research/iteration layer on
top in something faster to change.

## Build & test

```bash
make test                      # C++ engine: 22 checks (reconstruction + CSV replay)
./build_bindings.sh            # compile the Python extension
python3 tests/test_bindings.py       # 15 checks: Python bridge agrees with C++
python3 tests/test_naive_mm.py       # 6 checks: naive strategy + fill model
python3 tests/test_avellaneda_stoikov.py  # 11 checks: AS formula + rolling horizon
python3 tests/test_calibrate.py      # 4 checks: sigma/kappa fitting math
```

`*.so`, the compiled C++ test binaries, and `__pycache__` are gitignored
build products, not checked in -- run the two build steps above before
`python3 tests/*.py` or the backtest scripts.

## The actual finding

Running `python3 backtest/run_comparison.py` against a synthetic
~125-second order flow (see caveat below), calibrating sigma and kappa
from that same data, and setting AS's naive default `gamma=0.01`:

| Strategy | Fills | Final inventory | Final cash |
|---|---|---|---|
| Naive symmetric (1 tick half-spread) | 4 | 0 | -50,000 |
| Avellaneda-Stoikov (gamma=0.01) | 0 | 0 | 0 |

AS got **zero fills** at its default gamma. Sweeping gamma down:

| gamma | half-spread at session start (ticks) | fills |
|---|---|---|
| 0.01 | 60.9 | 0 |
| 0.001 | 6.1 | 1 |
| 0.0001 | 0.65 | 1 |
| 0.00001 | 0.10 | 2 |

**Why this happens, and why it's a real property of the model, not a
bug:** Avellaneda-Stoikov's spread has an inventory-risk term
proportional to `gamma * sigma^2 * (T - t)`, where `T - t` is time
remaining until session end. Early in a session, `T - t` is large, so
this term dominates and produces a very wide spread -- 60+ ticks wide
here, far outside where any trade in this window actually occurred.
Only gamma values far smaller than the model's own "reasonable
default" produce fills at all, and even then, fewer than the naive
baseline gets.

This is a known, documented practical limitation of the textbook AS
model: it was derived for a single-asset end-of-day liquidation
horizon, and the `T - t` term can dominate badly over short horizons
unless gamma is chosen very small or the horizon is redefined (e.g. a
short rolling window instead of full session end).

**Testing the fix directly: does a rolling horizon restore fills at a
sane gamma, instead of requiring an unrealistically tiny one?**
`strategies/avellaneda_stoikov.py` now takes an optional `horizon`
parameter that caps `T - t` at that many seconds instead of letting it
grow unbounded toward a fixed session close (`horizon=None` reproduces
the original behavior exactly -- see `backtest/run_horizon_comparison.py`).
Sweeping horizon at the *same* `gamma=0.01` that produced zero fills above:

| horizon (s) | half-spread @ session start (ticks) | fills |
|---|---|---|
| None (fixed session end) | 61.0 | 0 |
| 30 | 15.0 | 0 |
| 10 | 5.0 | 1 |
| 5 | 2.0 | 1 |
| 2 | 1.0 | 1 |
| 1 | 1.0 | 1 |

Capping the horizon at 10 seconds or less restores fills at
`gamma=0.01` -- the model's own "reasonable default" -- without touching
gamma at all. This confirms the hypothesis: the pathology is specifically
the *unbounded* `T - t` term from a fixed, far-off session close, not a
flaw in the gamma value itself. It's still fewer fills than the naive
baseline's 4 in this window (a fixed 1-tick half-spread is simply more
willing to trade in a session this short), which is itself worth stating
plainly rather than declaring the fix a win -- see caveats below.

## Honest limitations

- **The comparison above uses synthetic data**
  (`data/synthetic_large.csv`, a random walk generated by
  `backtest/generate_synthetic_data.py`), not real market data. It
  proves the pipeline (parsing, matching, fill simulation, calibration,
  comparison) works end-to-end. It is not a market finding. Real
  LOBSTER sample data (lobsterdata.com) should replace it before any
  conclusion here is treated as meaningful.
- **The fill model is a standard approximation** (queue-position /
  volume-based), not exact L3 matching -- see the detailed caveat in
  `backtest/replay_backtest.py`.
- **The rolling-horizon sweep above uses one fixed `horizon` per run,
  chosen by hand.** It isn't calibrated (e.g. to the timescale over
  which sigma/kappa are themselves stable) -- it demonstrates the
  mechanism, not an optimized horizon length.
- **No transaction costs or adverse selection modeling** in the
  backtest yet.

## What I'd build next

- Swap in real LOBSTER sample data and re-run calibration + comparison
- Add transaction costs to the backtest
- Multi-day robustness: run the same comparison across several
  different days/sessions rather than one window, to see whether the
  naive-beats-AS-at-default-gamma result holds up or was specific to
  this one window
- Calibrate the horizon length itself (rather than hand-picking it),
  e.g. from how quickly the fitted sigma/kappa drift over the session
