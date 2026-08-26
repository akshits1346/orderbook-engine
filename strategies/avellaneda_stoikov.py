"""
Avellaneda-Stoikov optimal market making -- the model this whole project
exists to test against the naive baseline.

Core idea: instead of quoting symmetrically around mid regardless of
inventory, skew your RESERVATION PRICE away from mid based on how much
inventory risk you're currently carrying. Long inventory -> skew down
(more eager to sell, less eager to buy more, nudging you back toward
flat). Short inventory -> skew up, symmetric logic.

    reservation_price = mid - inventory * gamma * sigma^2 * (T - t)
    optimal_half_spread = (gamma * sigma^2 * (T - t)) / 2
                          + (1 / gamma) * ln(1 + gamma / kappa)

    gamma  : risk aversion. Higher = more aggressive inventory skewing
             and a wider spread.
    sigma  : volatility of the underlying, in the SAME units as price
             (LOBSTER's dollars * 10000 convention), per second.
    kappa  : order arrival intensity decay parameter -- how fast fill
             probability drops off as your quote moves away from the
             best price. Larger kappa = liquidity/fills dry up faster
             as you move away from the touch.
    T - t  : time remaining in the trading session, in seconds.

CALIBRATION CAVEAT -- READ THIS BEFORE TRUSTING ANY RESULT FROM THIS
STRATEGY: gamma is a risk preference you choose. sigma and kappa are
NOT free choices -- they're supposed to be estimated from real market
data (sigma from realized price volatility, kappa from how fill rates
actually decay with quote distance in the data you're replaying).
Using arbitrary sigma/kappa values turns any "AS beats naive" or
"naive beats AS" result into a comparison of two arbitrary parameter
choices, not a real finding about the model. This file uses whatever
config is passed to it; calibrating sigma/kappa from data is a
separate, explicit step (see backtest/calibrate.py) that must be done
before the comparison means anything.
"""
import math
from dataclasses import dataclass


@dataclass
class AvellanedaStoikovConfig:
    gamma: float               # risk aversion -- no sensible default; must be chosen deliberately
    kappa: float               # order arrival decay -- calibrate from data, don't guess
    sigma: float                # volatility, LOBSTER price units per second -- calibrate from data
    session_end_time: float     # absolute time (LOBSTER 'seconds since midnight' units) the session ends
    tick_size: int = 100
    quote_size: int = 100
    max_inventory: int = 500
    requote_threshold_ticks: int = 1


class AvellanedaStoikov:
    def __init__(self, config: AvellanedaStoikovConfig):
        self.cfg = config
        self.inventory = 0
        self.cash = 0
        self.active_bid_id = None
        self.active_ask_id = None
        self.active_bid_price = None
        self.active_ask_price = None
        self.fills = []

    def _time_remaining(self, time: float) -> float:
        # Floor at a tiny positive value rather than 0 or negative --
        # the formulas divide by nothing here, but a negative T-t would
        # produce a nonsensical negative spread, which is worth guarding
        # against explicitly rather than letting it happen silently.
        return max(self.cfg.session_end_time - time, 1e-6)

    def _reservation_price(self, mid_price: int, time: float) -> float:
        T_minus_t = self._time_remaining(time)
        return mid_price - self.inventory * self.cfg.gamma * (self.cfg.sigma ** 2) * T_minus_t

    def _half_spread(self, time: float) -> float:
        T_minus_t = self._time_remaining(time)
        gamma, sigma, kappa = self.cfg.gamma, self.cfg.sigma, self.cfg.kappa
        inventory_term = (gamma * (sigma ** 2) * T_minus_t) / 2.0
        liquidity_term = (1.0 / gamma) * math.log(1 + gamma / kappa)
        return inventory_term + liquidity_term

    def desired_quotes(self, mid_price: int, time: float):
        reservation = self._reservation_price(mid_price, time)
        half_spread = self._half_spread(time)

        tick = self.cfg.tick_size
        bid = int(round((reservation - half_spread) / tick) * tick)
        ask = int(round((reservation + half_spread) / tick) * tick)
        return bid, ask

    def wants_requote(self, mid_price: int, time: float) -> bool:
        if self.active_bid_price is None or self.active_ask_price is None:
            return True
        desired_bid, desired_ask = self.desired_quotes(mid_price, time)
        threshold = self.cfg.requote_threshold_ticks * self.cfg.tick_size
        return (abs(desired_bid - self.active_bid_price) > threshold or
                abs(desired_ask - self.active_ask_price) > threshold)

    def can_buy(self) -> bool:
        return self.inventory + self.cfg.quote_size <= self.cfg.max_inventory

    def can_sell(self) -> bool:
        return self.inventory - self.cfg.quote_size >= -self.cfg.max_inventory

    def on_fill(self, side: str, price: int, size: int, time: float):
        if side == "buy":
            self.inventory += size
            self.cash -= price * size
        else:
            self.inventory -= size
            self.cash += price * size
        self.fills.append({"time": time, "side": side, "price": price, "size": size})

    def mark_to_market(self, mid_price: int) -> float:
        return self.cash + self.inventory * mid_price
