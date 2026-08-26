#pragma once
// Applies a stream of parsed LOBSTER MessageEvents to an OrderBook.
//
// This is intentionally a separate translation layer from OrderBook
// itself: OrderBook knows nothing about LOBSTER's type codes, and the
// parser knows nothing about order books. Keeping them decoupled means
// either one can be reused (e.g. OrderBook against a different
// exchange's message format) without touching the other.

#include "lobster_parser.hpp"
#include "order_book.hpp"

namespace lob {

// Applies one event to the book. Cross trades (type 6) and trading
// halts (type 7) are intentionally no-ops here: a cross trade doesn't
// mutate the resting book the way a normal execution does (LOBSTER
// documents it as reporting an auction match, not a continuous-trading
// event), and a halt has no size/price semantics at all. Skipping them
// explicitly (rather than silently falling through a switch) is a
// deliberate choice -- it's the kind of thing worth being able to
// explain rather than have quietly happen.
void applyMessage(OrderBook& book, const MessageEvent& ev);

// Convenience: replay an entire vector of events in order.
void replayAll(OrderBook& book, const std::vector<MessageEvent>& events);

} // namespace lob
