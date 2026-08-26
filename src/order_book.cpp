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
    level->orders.push_back(o); // back of deque = joins the end of the queue
    index_[id] = OrderLocation{side, price};
}

void OrderBook::cancelPartial(int64_t id, int64_t cancelSize) {
    auto it = index_.find(id);
    if (it == index_.end()) return; // unknown order id, ignore (could log)

    OrderLocation loc = it->second;
    PriceLevel* level = (loc.side == Side::Buy) ? levelForBuy(loc.price) : levelForSell(loc.price);

    for (auto& o : level->orders) {
        if (o.id == id) {
            o.size -= cancelSize;
            if (o.size <= 0) {
                // fully cancelled via repeated partials; remove it
                deleteOrder(id);
            }
            return;
        }
    }
}

void OrderBook::deleteOrder(int64_t id) {
    auto it = index_.find(id);
    if (it == index_.end()) return;

    OrderLocation loc = it->second;
    PriceLevel* level = (loc.side == Side::Buy) ? levelForBuy(loc.price) : levelForSell(loc.price);

    for (auto d = level->orders.begin(); d != level->orders.end(); ++d) {
        if (d->id == id) {
            level->orders.erase(d);
            break;
        }
    }
    index_.erase(it);
    eraseLevelIfEmpty(loc.side, loc.price);
}

void OrderBook::executeOrder(int64_t id, int64_t execSize) {
    auto it = index_.find(id);
    if (it == index_.end()) return;

    OrderLocation loc = it->second;
    PriceLevel* level = (loc.side == Side::Buy) ? levelForBuy(loc.price) : levelForSell(loc.price);

    for (auto& o : level->orders) {
        if (o.id == id) {
            o.size -= execSize;
            if (o.size <= 0) {
                deleteOrder(id);
            }
            return;
        }
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

    OrderLocation loc = it->second;
    const PriceLevel* level = nullptr;
    if (loc.side == Side::Buy) {
        auto bIt = bids_.find(loc.price);
        if (bIt != bids_.end()) level = &bIt->second;
    } else {
        auto aIt = asks_.find(loc.price);
        if (aIt != asks_.end()) level = &aIt->second;
    }
    if (!level) throw std::runtime_error("sizeAheadOf: level missing for indexed order");

    int64_t ahead = 0;
    for (const auto& o : level->orders) {
        if (o.id == id) return ahead;
        ahead += o.size;
    }
    throw std::runtime_error("sizeAheadOf: order id in index but not in level deque (bug)");
}

} // namespace lob
