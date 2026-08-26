"""
Naive symmetric market maker: quotes a fixed half-spread around the
current mid-price, requotes when the market has moved enough, and stops
adding to a side once inventory would breach a hard cap.

This exists specifically as a BASELINE. The point of building
Avellaneda-Stoikov is to see whether skewing quotes by inventory
actually beats this simple, inventory-blind approach on the same
replayed data -- if it doesn't, that's a real, honest finding worth
reporting, not a reason to hide the naive strategy's results.

NOTE ON THE `time` ARGUMENT: this strategy doesn't use it at all --
it's accepted purely so NaiveMarketMaker and AvellanedaStoikov share
an identical interface and ReplayBacktest can run either one without
caring which. Avellaneda-Stoikov genuinely needs the current time (to
compute time-remaining-in-session); naive doesn't, but takes the same
argument anyway so the two are interchangeable.
"""
from dataclasses import dataclass


@dataclass
class NaiveMarketMakerConfig:
    half_spread_ticks: int = 1
    tick_size: int = 100          # LOBSTER price units per tick (100 units = $0.01 at the *10000 convention)
    quote_size: int = 100         # shares quoted per side
    max_inventory: int = 500      # hard risk limit -- stop quoting a side that would breach this
    requote_threshold_ticks: int = 1  # requote once desired price moves more than this many ticks from current quote


class NaiveMarketMaker:
    def __init__(self, config: NaiveMarketMakerConfig):
        self.cfg = config
        self.inventory = 0
        self.cash = 0
        self.active_bid_id = None
        self.active_ask_id = None
        self.active_bid_price = None
        self.active_ask_price = None
        self.fills = []  # each: {"time", "side", "price", "size"}

    def desired_quotes(self, mid_price: int, time: float = None):
        """What this strategy WANTS to quote right now, given mid price.
        Ignores inventory (that's the whole point of calling it naive)
        and ignores time (see module docstring)."""
        half = self.cfg.half_spread_ticks * self.cfg.tick_size
        return mid_price - half, mid_price + half

    def wants_requote(self, mid_price: int, time: float = None) -> bool:
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
