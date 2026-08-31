#include "order_book.hpp"
#include <stdexcept>

namespace lob {

PriceLevel* OrderBook::levelForBuy(int64_t price) {
    return &bids_[price]; // map::operator[] default-constructs if absent
}

PriceLevel* OrderBook::levelForSell(int64_t price) {
    return &asks_[price];
}

void OrderBook::eraseLevelIfEmpty(Side side, int64_t price) {
    if (side == Side::Buy) {
        auto it = bids_.find(price);
        if (it != bids_.end() && it->second.orders.empty()) bids_.erase(it);
    } else {
        auto it = asks_.find(price);
        if (it != asks_.end() && it->second.orders.empty()) asks_.erase(it);
    }
}

void OrderBook::addLimitOrder(int64_t id, Side side, int64_t price, int64_t size, double timestamp) {
    Order o{id, side, price, size, timestamp};
    PriceLevel* level = (side == Side::Buy) ? levelForBuy(price) : levelForSell(price);
    level->orders.push_back(o); // back of list = joins the end of the queue
    // push_back on a std::list never invalidates any OTHER element's
    // iterators, so this new iterator (and every previously-cached one
    // in index_) stays valid for as long as this exact order remains.
    auto it = std::prev(level->orders.end());
    index_[id] = OrderLocation{side, price, it};
}

void OrderBook::cancelPartial(int64_t id, int64_t cancelSize) {
    auto it = index_.find(id);
    if (it == index_.end()) return; // unknown order id, ignore (could log)

    // Direct O(1) access via the cached iterator -- no scan through the
    // level's queue needed to find this order.
    it->second.it->size -= cancelSize;
    if (it->second.it->size <= 0) {
        // fully cancelled via repeated partials; remove it
        deleteOrder(id);
    }
}

void OrderBook::deleteOrder(int64_t id) {
    auto it = index_.find(id);
    if (it == index_.end()) return;

    OrderLocation loc = it->second;
    PriceLevel* level = (loc.side == Side::Buy) ? levelForBuy(loc.price) : levelForSell(loc.price);

    // std::list::erase by iterator is O(1) and, critically, does NOT
    // invalidate any other element's iterator -- every other order's
    // cached OrderLocation::it in index_ stays valid after this.
    level->orders.erase(loc.it);
    index_.erase(it);
    eraseLevelIfEmpty(loc.side, loc.price);
}

void OrderBook::executeOrder(int64_t id, int64_t execSize) {
    auto it = index_.find(id);
    if (it == index_.end()) return;

    it->second.it->size -= execSize;
    if (it->second.it->size <= 0) {
        deleteOrder(id);
    }
}

std::vector<BookLevel> OrderBook::bidLevels(size_t n) const {
    std::vector<BookLevel> out;
    for (const auto& [price, level] : bids_) {
        if (out.size() >= n) break;
        out.push_back({price, level.totalSize()});
    }
    return out;
}

std::vector<BookLevel> OrderBook::askLevels(size_t n) const {
    std::vector<BookLevel> out;
    for (const auto& [price, level] : asks_) {
        if (out.size() >= n) break;
        out.push_back({price, level.totalSize()});
    }
    return out;
}

std::optional<int64_t> OrderBook::bestBid() const {
    if (bids_.empty()) return std::nullopt;
    return bids_.begin()->first;
}

std::optional<int64_t> OrderBook::bestAsk() const {
    if (asks_.empty()) return std::nullopt;
    return asks_.begin()->first;
}

int64_t OrderBook::sizeAheadOf(int64_t id) const {
    auto it = index_.find(id);
    if (it == index_.end()) throw std::runtime_error("sizeAheadOf: unknown order id");

    // The cached iterator tells us exactly which order this is directly
    // -- no scan needed to FIND it. Summing everyone strictly ahead of
    // it in the queue is still O(queue position), but that's inherent
    // to the question being asked ("how much size is ahead of me"), not
    // an avoidable scan-to-find-the-order cost like the one fixed above.
    OrderLocation loc = it->second;
    const PriceLevel* level = (loc.side == Side::Buy) ? &bids_.at(loc.price) : &asks_.at(loc.price);

    int64_t ahead = 0;
    for (auto o = level->orders.begin(); o != loc.it; ++o) {
        ahead += o->size;
    }
    return ahead;
}

} // namespace lob
