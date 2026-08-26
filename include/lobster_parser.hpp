#pragma once
// Parser for LOBSTER's message file format.
//
// LOBSTER message files are headerless CSVs with 6 columns per row:
//   Time, Type, OrderID, Size, Price, Direction
//
// Time      : seconds since midnight (decimal, e.g. 34200.123456789)
// Type      : 1 = new limit order submission
//             2 = partial cancellation (order shrinks, keeps queue position)
//             3 = total deletion (order removed entirely)
//             4 = execution of a visible limit order (at the resting price)
//             5 = execution of a hidden limit order
//             6 = cross trade (opening/closing auction match)
//             7 = trading halt indicator
// OrderID   : exchange-assigned id of the order being acted on
// Size      : number of shares involved in this event
// Price     : dollars * 10000 (LOBSTER's fixed-point convention)
// Direction : -1 = sell, 1 = buy (the direction of the ORDER referenced,
//             not of whoever is initiating the event)
//
// This parser only reads the file into structured rows. Turning those
// rows into order book mutations is a separate step (see replay.hpp) --
// keeping parsing and replay separate means you can unit test each half
// independently, which is exactly what the tests below do.

#include <cstdint>
#include <string>
#include <vector>

namespace lob {

enum class MessageType {
    NewLimitOrder = 1,
    PartialCancel = 2,
    TotalDelete = 3,
    VisibleExecution = 4,
    HiddenExecution = 5,
    CrossTrade = 6,
    TradingHalt = 7
};

struct MessageEvent {
    double time;
    MessageType type;
    int64_t orderId;
    int64_t size;
    int64_t price;
    int direction; // +1 buy, -1 sell
};

// Throws std::runtime_error on a malformed row (wrong column count,
// unparseable number, or a type code outside 1-7) rather than silently
// skipping it -- for reconstructing a book from real market data, a
// row you can't parse is a correctness bug you want to know about
// immediately, not a warning to scroll past.
std::vector<MessageEvent> readMessageFile(const std::string& path);

// Same parsing logic applied to an in-memory string instead of a file,
// so tests can exercise the parser without needing a file on disk.
std::vector<MessageEvent> parseMessageLines(const std::string& csvContent);

} // namespace lob
