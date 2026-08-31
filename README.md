# orderbook-engine

A price-time-priority limit order book engine in C++, with a Python
bridge (pybind11) for strategy research and backtesting. Built to
answer a specific question: does Avellaneda-Stoikov's inventory-aware
quoting actually beat a naive symmetric market maker, when both are
tested against the same replayed order flow with a real fill model?

## Architecture

```
LOBSTER message file (CSV) -- real, or synthetic from
generate_synthetic_data.py (plain random-walk, or
generate_realistic_lobster_data()'s fat-tailed / vol-clustered
version -- see tests/test_synthetic_data_realism.py)
        |
        v
lobster_parser (C++)  --  parses into MessageEvent structs
        |
        v
replay layer (C++)     --  maps LOBSTER type codes to book mutations
        |
        v
OrderBook (C++)         --  price-time-priority matching engine,
                             O(log P) add/cancel/execute (P = number
                             of distinct price levels), queue
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
python3 tests/test_synthetic_data_realism.py  # 7 checks: fat tails + volatility clustering
python3 tests/test_transaction_costs.py       # 7 checks: fee_bps deduction/rebate math
```

`*.so`, the compiled C++ test binaries, and `__pycache__` are gitignored
build products, not checked in -- run the two build steps above before
`python3 tests/*.py` or the backtest scripts.

```bash
make bench                                        # order book latency/throughput benchmark, see below
python3 backtest/run_multi_seed_comparison.py     # 50-session robustness check, see below
```

## A real O(k) bug in the matching engine, found by benchmarking and fixed

The engine was documented (in this README, no less) as "O(log n) add/
cancel/execute" without ever actually being measured. It wasn't true.

`cancelPartial` / `deleteOrder` / `executeOrder` all resolve an order id
to its price level in O(log P) via a hash index (P = number of distinct
price levels) -- fine so far -- but then had to find that SPECIFIC
order within its level's own queue. The original implementation stored
each level's queue as a `std::deque<Order>` and searched it with a
linear scan for the matching id. That makes cancel/execute actually
O(log P + k), k = orders resting at that exact price level, not O(log
P) -- indistinguishable from the correct complexity when k is small
(a few orders per level), and easily missed by hand-traced correctness
tests, which is exactly the failure mode: none of them stress a single
price level with thousands of resting orders, so nothing caught it
until it was specifically benchmarked for.

`bench/order_book_bench.cpp` (`make bench`) measures add/cancel latency
as a function of k directly: build a book with k orders all resting at
ONE price level, delete them in reverse arrival order (the worst case
for a front-scanning search), and check whether per-operation latency
stays flat as k grows or grows with it. Run against the actual git
history of this file, both ways:

| k (orders at one price level) | cancel, OLD (`std::deque` + scan), mean | cancel, FIXED (`std::list` + cached iterator), mean |
|---|---|---|
| 10 | 211 ns | 175 ns |
| 100 | 101 ns | 151 ns |
| 1,000 | 335 ns | 44 ns |
| 10,000 | 2,475 ns | 43 ns |
| 50,000 | 34,268 ns | 46 ns |

The old version's cancel latency grows with k, as the linear-scan
theory predicts (roughly 160x slower at k=50,000 than at k=10); the
fixed version stays flat across the same four-orders-of-magnitude
range -- a ~740x improvement at k=50,000, and genuinely O(1) per
cancel now, not just "small enough not to notice yet."

**The fix**: swap each price level's queue from `std::deque<Order>` to
`std::list<Order>`, and have `OrderLocation` (the value in the id ->
location index) cache a direct `std::list<Order>::iterator` to the
order, set once at `addLimitOrder` time. `std::list::erase(iterator)`
is O(1) and -- the property that actually makes this safe -- erasing
or mutating one element never invalidates any OTHER element's
iterator, unlike `std::deque`, where inserting or erasing anywhere but
the ends invalidates every iterator into it. That's what makes caching
iterators across the whole book's lifetime, not just within one
function call, sound. See the COMPLEXITY note in `include/order_book.hpp`.

**Honest caveat on the benchmark methodology**: this measures wall-clock
time via `std::chrono::steady_clock` in a shared, virtualized
container, not a pinned-core isolated rig -- the absolute nanosecond
figures above are noisier than a real latency-SLA measurement would be
(note the non-monotonic k=10 vs k=100 OLD row, almost certainly
measurement noise, not the algorithm getting faster). What's real and
what this benchmark is actually built to show is the SCALING SHAPE:
flat vs. growing-with-k is visible through that noise once k spans
enough orders of magnitude, and that shape is a property of the
algorithm, not the machine.

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

## Robustness check: does this hold across many sessions, and a second, deeper bug it found

One synthetic file is one random draw. `backtest/run_multi_seed_comparison.py`
reruns naive, `as_fixed` (gamma=0.01, fixed session-end horizon), and
`as_rolling` (gamma=0.01, 10s rolling horizon) across 50 independent
synthetic sessions -- each generated by `generate_realistic_lobster_data()`
(fat-tailed, volatility-clustered order flow; see the data-realism
section below), each recalibrating sigma/kappa from its own data, exactly
as the single-file comparison does:

| strategy | mean fills | sessions w/ >=1 fill | mean final MTM | median final MTM |
|---|---|---|---|---|
| naive | 3.63 | 95.7% | 71,847.83 | 107,500.00 |
| as_fixed | 0.22 | 21.7% | 44,673.91 | 0.00 |
| as_rolling | 0.96 | 95.7% | **-211,086.96** | **-127,500.00** |

(46 of 50 seeds produced usable sessions; the rest were skipped -- too
little data, or `calibrate.py`'s own documented kappa-fit failure mode.)

Two things confirmed, one thing NOT expected:

1. **The zero-fills finding generalizes.** `as_fixed` gets a fill in
   only 21.7% of sessions, consistent with the single-file result --
   not a one-seed artifact.
2. **The rolling horizon does restore participation.** `as_rolling`
   gets at least one fill in 95.7% of sessions, matching naive.
3. **But `as_rolling`'s average PnL is dramatically worse than BOTH
   naive AND the version that barely trades at all.** That's not what
   "fixing the fill problem" was supposed to produce, and it isn't
   noise -- naive beats `as_rolling` on final MTM in 63% of sessions,
   never ties.

**Digging into why** (seed 0, a representative case): AS gets its
first fill (buy 100 shares), and inventory goes from 0 to 100. Directly
computing the NEXT desired quote from that state:

```
sigma=57.25, kappa=0.088, inventory=100, mid=1000800 ($100.08)
desired ask = 968200 ($96.82) -- 326 TICKS below mid
```

The reservation-price skew is `inventory * gamma * sigma^2 * (T-t)` --
note the extra factor of `inventory` that the half-spread term doesn't
have. The rolling horizon caps `T-t` and therefore controls the
HALF-SPREAD term (confirmed above), but the SKEW term still has that
`inventory` multiplier, and once inventory is nonzero it can blow up
independently, even at the same capped horizon. The resulting ask quote
here is priced so far below the actual market that, in a real
continuous double auction, it would cross the book and execute
IMMEDIATELY. But this project's `OrderBook.addLimitOrder` (see
`include/order_book.hpp`) is a resting-order / LOBSTER-replay
reconciliation structure, not a full matching engine -- it never
auto-executes a crossing order on insertion, by design (real fills only
come from historical execution messages landing on our exact quoted
price; see the fill model caveat in `backtest/replay_backtest.py`). So
the quote that SHOULD flatten the position instead just sits, unfilled,
for the rest of the session, because no synthetic trade will occur 326
ticks from the market. The position is stuck long, fully exposed to
wherever the (now fat-tailed, volatility-clustered) price random-walks
by session end -- which is a directional bet, not market-making, and
explains both the sign and the size of the average loss.

This is a second, DIFFERENT bug from the one the single-file gamma
sweep found (that one was in the half-spread term; this one is in the
skew term), and it was only found by stress-testing the fix across many
sessions rather than trusting the one-file result. It's also a genuine
limitation of the backtest's fill model interacting with the strategy,
not a flaw in either piece considered alone -- see "Honest limitations"
and "What I'd build next" below for what actually fixes it.

## Transaction costs

`ReplayBacktest` takes an optional `fee_bps` (basis points of notional,
charged to the strategy's cash on every fill; see
`tests/test_transaction_costs.py`). Rerunning the same 50-seed sweep at
a realistic 1.0 bps per fill:

| strategy | mean final MTM, fee-free | mean final MTM, 1.0 bps |
|---|---|---|
| naive | 71,847.83 | 35,560.13 |
| as_fixed | 44,673.91 | 42,500.02 |
| as_rolling | -211,086.96 | -220,651.72 |

Naive's mean PnL is cut roughly in half by a 1 bp fee; `as_fixed` and
`as_rolling` barely move. That's not surprising once you look at fill
counts (naive: 3.63 fills/session; `as_fixed`: 0.22; `as_rolling`:
0.96) -- fee drag scales with how often a strategy actually trades, and
naive trades roughly 4x more often than `as_rolling` and 16x more often
than `as_fixed` in this data. Naive still wins on average at this fee
level (it started from a much larger edge), but "naive beats AS" is
visibly not fee-invariant, and a high enough per-fill cost would flip
it -- worth remembering before reading either result as a permanent
statement about which approach is better.

## Honest limitations

- **None of the findings above use real market data.**
  `data/synthetic_large.csv` (the single-file comparison) and
  `generate_realistic_lobster_data()` (the multi-seed robustness check)
  are both synthetic -- the latter is statistically validated to
  reproduce fat tails and volatility clustering (see
  `tests/test_synthetic_data_realism.py`), which is a materially
  better stand-in than a plain random walk, but it is still not real
  order flow. This was a deliberate fallback, not an oversight: this
  project was built in a sandboxed environment with outbound network
  access restricted to an allowlist (package registries only) --
  lobsterdata.com and every other external data source were confirmed
  unreachable (`EGRESS_BLOCKED`) before falling back to a better
  synthetic generator instead. **No code changes are needed to use
  real data** -- `lob.read_message_file(path)` already reads any
  LOBSTER-format CSV, real or synthetic, identically; swapping
  `data_path` in `run_comparison.py` / `run_multi_seed_comparison.py`
  to a real downloaded LOBSTER sample is the entire integration step.
- **The fill model is a standard approximation** (queue-position /
  volume-based), not exact L3 matching, AND -- as the multi-seed
  robustness check's second finding shows -- it never auto-executes a
  crossing/marketable order on insertion. That's fine for orders
  placed near the market, but it means a strategy that (correctly, by
  its own formula) prices a quote far enough from the market to cross
  the book will see that quote sit unfilled rather than executing
  immediately, which is what actually happened to `as_rolling` above.
  See `backtest/replay_backtest.py` for the detailed fill-model caveat.
- **The rolling-horizon sweep uses one fixed `horizon` per run, chosen
  by hand**, and the multi-seed check found it only fixes the
  half-spread term, not the separate skew-term blowup once inventory
  is nonzero -- see above.
- **No transaction costs or adverse selection modeling** in the
  backtest yet.

## What I'd build next

- Two independent fixes for the skew-term blowup found by the
  multi-seed check, worth trying separately since they address
  different halves of the problem: (1) cap the skew term with the same
  kind of bound the rolling horizon applies to the half-spread term
  (e.g. clamp `inventory * gamma * sigma^2 * (T-t)` itself, not just
  `T-t`), and (2) give the backtest's fill model the ability to
  auto-execute a resting order of ours that crosses the current best
  opposing price, which is what a real matching engine would do and
  is arguably the more honest fix, since the strategy's quote wasn't
  wrong, the backtest's execution model was incomplete
- Real LOBSTER sample data, the moment it's reachable from wherever
  this runs next -- see the honest-limitations note on why it isn't
  here now, and that no code changes are needed once it is
- Calibrate the horizon length itself (rather than hand-picking it),
  e.g. from how quickly the fitted sigma/kappa drift over the session
- Sweep fee_bps further (the 1.0 bps point above is one sample, not a
  curve) to find the actual breakeven fee where naive's edge flips
