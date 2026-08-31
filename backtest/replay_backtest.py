"""
Event-by-event backtest: replays LOBSTER-format market data through the
real C++ order book one message at a time, letting a strategy quote
against it and tracking realistic fills.

READ THIS BEFORE CHANGING ANYTHING -- the fill model is the single most
important design decision in this file, and getting it wrong would make
every result downstream meaningless.

Our own resting orders are inserted into the SAME OrderBook instance the
historical data replays into (using negative order ids, which never
collide with LOBSTER's positive ids). That gives us a real queue
position for free via size_ahead_of(). But historical execution
messages only ever reference historical order ids -- they can never
directly "hit" our order, because the historical data was recorded
before we existed.

Fill model used here (a standard simplification in LOB backtesting,
sometimes called a queue-position / volume-based fill model): for each
active quote, track cumulative traded volume at that exact price level
since the quote was placed. Once that cumulative volume reaches the
size that was ahead of us in queue at placement time, treat our order
as filled for its full quoted size.

Known limitation, stated honestly: this assumes strict FIFO consumption
of the queue ahead of us, and doesn't distinguish "500 shares traded
through this level" from "500 shares of cancellations freed up the
level" -- LOBSTER's data lets you tell these apart (a cancel is a
different message type than an execution) and this model already does
use only execution messages, not cancels, to accumulate volume -- but
it still assumes every unit of traded volume ahead of us actually
depletes queue position, which is the standard (and correct) assumption
for execution messages specifically.

INTERFACE NOTE: strategies passed in here must implement
desired_quotes(mid_price, time), wants_requote(mid_price, time),
can_buy(), can_sell(), on_fill(side, price, size, time), and
mark_to_market(mid_price). Both NaiveMarketMaker and AvellanedaStoikov
implement this same interface, which is what lets this one harness run
either strategy interchangeably for a head-to-head comparison.

TRANSACTION COSTS: optional, via the fee_bps constructor argument (0.0
by default -- every existing test and comparison is fee-free unless it
explicitly opts in). See the ReplayBacktest.__init__ docstring and
tests/test_transaction_costs.py.
"""
import lob_engine as lob


class ReplayBacktest:
    def __init__(self, events, strategy, fee_bps: float = 0.0):
        """
        fee_bps: transaction cost per fill, in basis points of notional
        (price * size), deducted from the strategy's cash immediately
        after on_fill(). 1 bp = 0.01%; a typical maker fee/rebate for a
        liquidity-providing limit order is on this order of magnitude
        (single-digit bps, sometimes negative -- a rebate -- for adding
        liquidity, which this parameter can represent too by passing a
        negative value). Applied here, at the backtest level, rather
        than inside each strategy's on_fill: transaction cost is a
        property of the EXECUTION VENUE, not the strategy's quoting
        logic, so every strategy gets it identically without needing to
        know about fees itself. See tests/test_transaction_costs.py.
        """
        self.events = events
        self.strategy = strategy
        self.book = lob.OrderBook()
        self.fee_bps = fee_bps
        self.total_fees_paid = 0.0

        self._next_own_id = -1  # counts down: -1, -2, -3, ... never collides with LOBSTER's positive ids
        self._bid_volume_since_quote = 0
        self._ask_volume_since_quote = 0
        self._bid_size_ahead_at_quote = 0
        self._ask_size_ahead_at_quote = 0

        self.mtm_history = []  # list of (time, mark_to_market_value)

    def _next_id(self):
        self._next_own_id -= 1
        return self._next_own_id

    def _charge_fee(self, price, size):
        if self.fee_bps == 0.0:
            return
        fee = abs(price) * size * (self.fee_bps / 10000.0)
        self.strategy.cash -= fee
        self.total_fees_paid += fee

    def _mid(self):
        bid = self.book.best_bid()
        ask = self.book.best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) // 2

    def _cancel_own_quotes(self):
        if self.strategy.active_bid_id is not None:
            self.book.delete_order(self.strategy.active_bid_id)
            self.strategy.active_bid_id = None
            self.strategy.active_bid_price = None
        if self.strategy.active_ask_id is not None:
            self.book.delete_order(self.strategy.active_ask_id)
            self.strategy.active_ask_id = None
            self.strategy.active_ask_price = None

    def _place_quotes(self, time, mid):
        bid_price, ask_price = self.strategy.desired_quotes(mid, time)

        if self.strategy.can_buy():
            bid_id = self._next_id()
            self.book.add_limit_order(bid_id, lob.Side.Buy, bid_price, self.strategy.cfg.quote_size, time)
            self.strategy.active_bid_id = bid_id
            self.strategy.active_bid_price = bid_price
            self._bid_size_ahead_at_quote = self.book.size_ahead_of(bid_id)
            self._bid_volume_since_quote = 0

        if self.strategy.can_sell():
            ask_id = self._next_id()
            self.book.add_limit_order(ask_id, lob.Side.Sell, ask_price, self.strategy.cfg.quote_size, time)
            self.strategy.active_ask_id = ask_id
            self.strategy.active_ask_price = ask_price
            self._ask_size_ahead_at_quote = self.book.size_ahead_of(ask_id)
            self._ask_volume_since_quote = 0

    def _check_fills(self, event):
        """Must run BEFORE the event is applied to the book, so 'size ahead at
        quote time' still refers to orders that haven't been depleted by
        THIS event yet."""
        is_execution = event.type in (lob.MessageType.VisibleExecution, lob.MessageType.HiddenExecution)
        if not is_execution:
            return

        if self.strategy.active_bid_price is not None and event.price == self.strategy.active_bid_price:
            self._bid_volume_since_quote += event.size
            if self._bid_volume_since_quote >= self._bid_size_ahead_at_quote:
                fill_size = self.strategy.cfg.quote_size
                self.book.delete_order(self.strategy.active_bid_id)
                self.strategy.on_fill("buy", self.strategy.active_bid_price, fill_size, event.time)
                self._charge_fee(self.strategy.active_bid_price, fill_size)
                self.strategy.active_bid_id = None
                self.strategy.active_bid_price = None

        if self.strategy.active_ask_price is not None and event.price == self.strategy.active_ask_price:
            self._ask_volume_since_quote += event.size
            if self._ask_volume_since_quote >= self._ask_size_ahead_at_quote:
                fill_size = self.strategy.cfg.quote_size
                self.book.delete_order(self.strategy.active_ask_id)
                self.strategy.on_fill("sell", self.strategy.active_ask_price, fill_size, event.time)
                self._charge_fee(self.strategy.active_ask_price, fill_size)
                self.strategy.active_ask_id = None
                self.strategy.active_ask_price = None

    def run(self):
        for event in self.events:
            self._check_fills(event)
            lob.apply_message(self.book, event)

            mid = self._mid()
            if mid is None:
                continue

            if self.strategy.wants_requote(mid, event.time):
                self._cancel_own_quotes()
                self._place_quotes(event.time, mid)

            self.mtm_history.append((event.time, self.strategy.mark_to_market(mid)))

        return self.mtm_history
