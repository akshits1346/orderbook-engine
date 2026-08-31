"""
Hand-traced test of AvellanedaStoikov's quoting formula.

Parameters chosen for clean arithmetic, NOT calibrated to any real
market -- see the calibration caveat in avellaneda_stoikov.py. This
test exists to verify the FORMULA is implemented correctly (correct
skew direction, correct time-decay direction, correct numeric values
for a hand-computable case), not to claim these are realistic trading
parameters.

Config: gamma=0.01, kappa=0.3, sigma=200, tick_size=100, mid=1000000.

Case 1 -- inventory=0, time=0, session_end_time=10 (T-t=10):
  half_spread = (0.01 * 200^2 * 10)/2 + (1/0.01)*ln(1 + 0.01/0.3)
              = 2000/2 + 100*ln(1.03333) = 1000 + 3.279 = 2003.279
  reservation = mid (inventory=0, so no skew) = 1000000
  bid = round((1000000 - 2003.279)/100)*100 = 998000
  ask = round((1000000 + 2003.279)/100)*100 = 1002000

Case 2 -- inventory=1, time=0, same session (T-t=10):
  reservation = 1000000 - 1*0.01*200^2*10 = 1000000 - 4000 = 996000
  bid = round((996000 - 2003.279)/100)*100 = 994000
  ask = round((996000 + 2003.279)/100)*100 = 998000
  Both quotes shift DOWN by 4000 vs case 1 -- this is the whole point
  of the model: carrying long inventory should make you skew down to
  encourage selling back toward flat.

Case 3 -- inventory=0, time=5, same session (T-t now = 5):
  half_spread = (0.01*200^2*5)/2 + 100*ln(1.03333) = 500 + 3.279 = 1003.279
  bid = round((1000000 - 1003.279)/100)*100 = 999000
  ask = round((1000000 + 1003.279)/100)*100 = 1001000
  Spread is NARROWER than case 1 -- less time remaining means less
  variance risk to price into the spread, which is the model's other
  key behavior (spread widens early in a session, narrows near the end).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategies.avellaneda_stoikov import AvellanedaStoikov, AvellanedaStoikovConfig

failures = 0


def check(cond, msg):
    global failures
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {msg}")
    if not cond:
        failures += 1


def main():
    cfg = AvellanedaStoikovConfig(
        gamma=0.01, kappa=0.3, sigma=200,
        session_end_time=10, tick_size=100, quote_size=100,
    )
    strategy = AvellanedaStoikov(cfg)

    # Case 1: inventory=0, time=0
    strategy.inventory = 0
    bid1, ask1 = strategy.desired_quotes(1000000, time=0)
    check(bid1 == 998000, f"case 1 bid == 998000, got {bid1}")
    check(ask1 == 1002000, f"case 1 ask == 1002000, got {ask1}")

    # Case 2: inventory=1, time=0 -- quotes should shift DOWN vs case 1
    strategy.inventory = 1
    bid2, ask2 = strategy.desired_quotes(1000000, time=0)
    check(bid2 == 994000, f"case 2 bid == 994000, got {bid2}")
    check(ask2 == 998000, f"case 2 ask == 998000, got {ask2}")
    check(bid2 < bid1 and ask2 < ask1,
          "positive inventory skews BOTH quotes down vs zero inventory")

    # Case 3: inventory=0, time=5 -- spread should be NARROWER than case 1
    strategy.inventory = 0
    bid3, ask3 = strategy.desired_quotes(1000000, time=5)
    check(bid3 == 999000, f"case 3 bid == 999000, got {bid3}")
    check(ask3 == 1001000, f"case 3 ask == 1001000, got {ask3}")
    spread1 = ask1 - bid1
    spread3 = ask3 - bid3
    check(spread3 < spread1,
          f"spread narrows as session end approaches ({spread3} < {spread1})")

    # Case 4 -- rolling horizon: horizon=5 caps T-t at 5 even at time=0,
    # session_end_time=10 (so T-t would otherwise be 10). Quotes should
    # come out IDENTICAL to case 3 (inventory=0, effective T-t=5),
    # even though we're evaluating at time=0, not time=5.
    cfg_horizon = AvellanedaStoikovConfig(
        gamma=0.01, kappa=0.3, sigma=200,
        session_end_time=10, horizon=5, tick_size=100, quote_size=100,
    )
    strategy_horizon = AvellanedaStoikov(cfg_horizon)
    strategy_horizon.inventory = 0
    bid4, ask4 = strategy_horizon.desired_quotes(1000000, time=0)
    check(bid4 == bid3 and ask4 == ask3,
          f"horizon=5 at time=0 matches fixed-horizon T-t=5 case: "
          f"got ({bid4}, {ask4}), want ({bid3}, {ask3})")

    # Case 5 -- rolling horizon never WIDENS the spread vs the
    # fixed-session-end version: capping T-t can only shrink it.
    cfg_uncapped = AvellanedaStoikovConfig(
        gamma=0.01, kappa=0.3, sigma=200,
        session_end_time=10, tick_size=100, quote_size=100,
    )
    strategy_uncapped = AvellanedaStoikov(cfg_uncapped)
    bid_u, ask_u = strategy_uncapped.desired_quotes(1000000, time=0)
    spread_uncapped = ask_u - bid_u
    spread_horizon = ask4 - bid4
    check(spread_horizon <= spread_uncapped,
          f"horizon-capped spread ({spread_horizon}) <= uncapped spread "
          f"({spread_uncapped}) at the same (time, session_end_time)")

    # Case 6 -- once t is close enough to session_end_time that natural
    # T-t already falls below the horizon cap, horizon has no effect
    # (min(small, horizon) == small) -- rolling horizon only ever
    # tightens, matching fixed-horizon behavior in the near-close regime.
    strategy_horizon.inventory = 0
    bid6, ask6 = strategy_horizon.desired_quotes(1000000, time=9)
    bid6_fixed, ask6_fixed = strategy_uncapped.desired_quotes(1000000, time=9)
    check(bid6 == bid6_fixed and ask6 == ask6_fixed,
          "horizon has no effect once natural T-t is already below the cap")

    print()
    if failures == 0:
        print("All Avellaneda-Stoikov formula checks passed.")
        return 0
    else:
        print(f"{failures} check(s) FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
