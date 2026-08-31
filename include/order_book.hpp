#pragma once
// Core limit order book with price-time priority.
//
// Design notes (read this before the .cpp):
// - Bids are stored highest-price-first, asks lowest-price-first, so
//   "best" is always begin() on the appropriate map.
// - Each PriceLevel holds a deque of Orders in arrival order. That deque
//   IS the time-priority queue: front() is always the order that would
//   be filled first at that price.
// - A separate index_ (order id -> which side/price it lives at) exists
//   so cancel/execute/modify by order id are O(log n) instead of O(n)
//   scans through every level.
// - IMPORTANT price-time-priority rule this engine enforces: a cancel
//   that reduces size in place KEEPS the order's queue position. A
//   modify that increases size or changes price is treated as a new
//   order at the back of the queue (loses time priority). This matches
//   real exchange matching rules and is exactly the kind of detail
//   interviewers ask about, so make sure you can explain why.
//
// - COMPLEXITY, and a bug that was here and got fixed: cancelPartial /
//   deleteOrder / executeOrder all go id -> its order in O(log P) via
//   index_ (P = number of distinct price levels), then must mutate or
//   remove THAT SPECIFIC order within its level's queue. The first
//   version of this file stored orders in a std::deque and looked the
//   id up with a LINEAR SCAN through the deque -- meaning cancel/
//   execute were actually O(log P + k), k = orders resting at that
//   exact price level, not O(log P) as originally documented here.
//   That's wrong for any operation that isn't add: a std::deque
//   iterator is invalidated by insertion/erasure anywhere in the
//   deque except at the very ends, so there was no way to cache a
//   stable position to jump to directly.
//   Fix: PriceLevel now uses std::list, whose iterators stay valid
//   under insertion/erasure ANYWHERE ELSE in the list (only the
//   erased element's own iterator is invalidated). index_ stores that
//   iterator directly (via OrderLocation::it), so cancelPartial /
//   deleteOrder / executeOrder go straight to the order with no scan:
//   true O(log P) (the map lookup for eraseLevelIfEmpty), not O(log P
//   + k). See bench/order_book_bench.cpp for the measurement that
//   caught this (latency growing with orders-per-level, not staying
//   flat) and confirms the fix.

#include <cstdint>
#include <list>
#include <map>
#include <optional>
#include <unordered_map>
#include <vector>

namespace lob {

enum class Side { Buy, Sell };

struct Order {
    int64_t id;
    Side side;
    int64_t price;   // LOBSTER convention: dollars * 10000, so $100.00 == 1000000
    int64_t size;    // shares remaining (not original size)
    double timestamp; // seconds since midnight, as LOBSTER encodes it
};

struct PriceLevel {
    std::list<Order> orders; // front = earliest = highest time priority
                             // (std::list, not std::deque: erasing/mutating
                             // by a cached iterator must be O(1) and must not
                             // invalidate OTHER orders' cached iterators --
                             // see the COMPLEXITY note above)

    int64_t totalSize() const {
        int64_t s = 0;
        for (const auto& o : orders) s += o.size;
        return s;
    }
};

struct BookLevel {
    int64_t price;
    int64_t size;
};

struct OrderLocation {
    Side side;
    int64_t price;
    std::list<Order>::iterator it; // direct handle to this order within its
                                    // level's queue -- O(1) access, no scan
};

class OrderBook {
public:
    // type=1 events in LOBSTER: a brand new resting limit order.
    void addLimitOrder(int64_t id, Side side, int64_t price, int64_t size, double timestamp);

    // type=2 events: partial cancellation. Size shrinks, queue position
    // is PRESERVED (this is the exchange-standard rule).
    void cancelPartial(int64_t id, int64_t cancelSize);

    // type=3 events: total deletion. Order leaves the book entirely.
    void deleteOrder(int64_t id);

    // type=4/5 events: (visible/hidden) execution against a resting
    // order. Reduces that order's size; if it hits zero, removes it.
    void executeOrder(int64_t id, int64_t execSize);

    // Top n levels, best first, for comparing against a reference
    // reconstruction (e.g. LOBSTER's own published orderbook file).
    std::vector<BookLevel> bidLevels(size_t n) const;
    std::vector<BookLevel> askLevels(size_t n) const;

    std::optional<int64_t> bestBid() const;
    std::optional<int64_t> bestAsk() const;

    // How many resting orders sit ahead of this one in its own queue,
    // by total size. Used later (week 3) to judge whether a simulated
    // order of ours would realistically have been filled yet.
    int64_t sizeAheadOf(int64_t id) const;

    size_t orderCount() const { return index_.size(); }

private:
    // Bids: highest price first. Asks: lowest price first (map's
    // natural order already gives us this for asks).
    std::map<int64_t, PriceLevel, std::greater<int64_t>> bids_;
    std::map<int64_t, PriceLevel> asks_;
    std::unordered_map<int64_t, OrderLocation> index_;

    PriceLevel* levelForBuy(int64_t price);
    PriceLevel* levelForSell(int64_t price);
    void eraseLevelIfEmpty(Side side, int64_t price);
};

} // namespace lob
