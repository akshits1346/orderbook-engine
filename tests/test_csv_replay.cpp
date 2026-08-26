// Exercises parser + replay layer together against data/sample_message.csv.
// Expected values below were hand-traced from that file line by line --
// see the comment block for the trace. If you add more rows to the CSV
// later, re-trace by hand before trusting new assertions.
//
// Trace of data/sample_message.csv:
//  1. new  id=10001 buy  300 @ 1000000        bid[1000000] = {10001:300}
//  2. new  id=10002 buy  200 @  999900        bid[999900]  = {10002:200}
//  3. new  id=10003 sell 500 @ 1000200        ask[1000200] = {10003:500}
//  4. new  id=10004 sell 400 @ 1000300        ask[1000300] = {10004:400}
//  5. partial cancel id=10001 by 100           10001: 300->200, keeps position
//  6. new  id=10005 buy  250 @ 1000000        joins behind 10001
//  7. exec id=10001 for 200                    10001 -> 0, removed
//                                              bid[1000000] = {10005:250}
//  8. delete id=10002                          bid[999900] level now empty, erased
//  9. new  id=10006 sell 600 @ 1000200        joins behind 10003
// 10. exec id=10003 for 500                    10003 -> 0, removed
//                                              ask[1000200] = {10006:600}
//
// Final expected state:
//   bids: only {1000000: 250}  (id=10005)
//   asks: {1000200: 600} (id=10006), {1000300: 400} (id=10004)
//   best bid = 1000000, best ask = 1000200
//   3 live orders total (10004, 10005, 10006)

#include "lobster_parser.hpp"
#include "order_book.hpp"
#include "replay.hpp"
#include <iostream>

using namespace lob;

static int failures = 0;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAIL: " << msg << " (line " << __LINE__ << ")\n"; \
            ++failures; \
        } else { \
            std::cout << "PASS: " << msg << "\n"; \
        } \
    } while (0)

int main() {
    auto events = readMessageFile("data/sample_message.csv");
    CHECK(events.size() == 10, "parsed exactly 10 rows from sample_message.csv");

    OrderBook book;
    replayAll(book, events);

    CHECK(book.bestBid().has_value() && book.bestBid().value() == 1000000,
          "best bid is 1000000 after full replay");
    CHECK(book.bestAsk().has_value() && book.bestAsk().value() == 1000200,
          "best ask is 1000200 after full replay");

    auto bids = book.bidLevels(5);
    CHECK(bids.size() == 1, "exactly one bid level remains");
    if (!bids.empty()) {
        CHECK(bids[0].price == 1000000, "remaining bid level price is 1000000");
        CHECK(bids[0].size == 250, "remaining bid level size is 250 (id=10005 only)");
    }

    auto asks = book.askLevels(5);
    CHECK(asks.size() == 2, "exactly two ask levels remain");
    if (asks.size() == 2) {
        CHECK(asks[0].price == 1000200 && asks[0].size == 600,
              "best ask level is 1000200 x 600 (id=10006)");
        CHECK(asks[1].price == 1000300 && asks[1].size == 400,
              "second ask level is 1000300 x 400 (id=10004, untouched)");
    }

    CHECK(book.orderCount() == 3, "3 live orders remain (10004, 10005, 10006)");

    if (failures == 0) {
        std::cout << "\nAll CSV replay checks passed.\n";
        return 0;
    } else {
        std::cout << "\n" << failures << " check(s) FAILED.\n";
        return 1;
    }
}
