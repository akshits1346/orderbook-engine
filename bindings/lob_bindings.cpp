// pybind11 bindings for the order book engine.
//
// This module intentionally exposes a narrow, strategy-facing surface --
// not every internal detail of OrderBook. A Python strategy needs to:
// place/cancel/execute orders (or more realistically, feed it replayed
// LOBSTER events), read the current top-of-book, and query queue
// position for its own resting orders. It does NOT need direct access
// to the internal std::map/deque structures, so those stay in C++.
//
// Why bind at all instead of reimplementing the book in Python: the
// matching engine is performance- and correctness-critical (it's the
// thing being tested against LOBSTER reference reconstruction), so it
// stays in C++ where it's already validated. The strategy layer is
// where you'll iterate fastest (trying different quoting logic,
// plotting results), so that's what goes in Python. This split --
// C++ for the hot/correctness-critical path, Python for research and
// iteration -- mirrors how real trading systems are typically built.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "order_book.hpp"
#include "lobster_parser.hpp"
#include "replay.hpp"

namespace py = pybind11;
using namespace lob;

PYBIND11_MODULE(lob_engine, m) {
    m.doc() = "Price-time-priority limit order book engine (C++ core, Python-bound)";

    py::enum_<Side>(m, "Side")
        .value("Buy", Side::Buy)
        .value("Sell", Side::Sell);

    py::class_<BookLevel>(m, "BookLevel")
        .def_readonly("price", &BookLevel::price)
        .def_readonly("size", &BookLevel::size)
        .def("__repr__", [](const BookLevel& l) {
            return "<BookLevel price=" + std::to_string(l.price) +
                   " size=" + std::to_string(l.size) + ">";
        });

    py::class_<OrderBook>(m, "OrderBook")
        .def(py::init<>())
        .def("add_limit_order", &OrderBook::addLimitOrder,
             py::arg("id"), py::arg("side"), py::arg("price"),
             py::arg("size"), py::arg("timestamp"),
             "Add a new resting limit order (LOBSTER type 1 equivalent).")
        .def("cancel_partial", &OrderBook::cancelPartial,
             py::arg("id"), py::arg("cancel_size"),
             "Reduce an order's size in place, KEEPING queue position "
             "(LOBSTER type 2 equivalent).")
        .def("delete_order", &OrderBook::deleteOrder,
             py::arg("id"),
             "Remove an order entirely (LOBSTER type 3 equivalent).")
        .def("execute_order", &OrderBook::executeOrder,
             py::arg("id"), py::arg("exec_size"),
             "Execute (fill) part or all of a resting order "
             "(LOBSTER type 4/5 equivalent).")
        .def("bid_levels", &OrderBook::bidLevels, py::arg("n"),
             "Top n bid levels, highest price first.")
        .def("ask_levels", &OrderBook::askLevels, py::arg("n"),
             "Top n ask levels, lowest price first.")
        .def("best_bid", &OrderBook::bestBid,
             "Best bid price, or None if no bids.")
        .def("best_ask", &OrderBook::bestAsk,
             "Best ask price, or None if no asks.")
        .def("size_ahead_of", &OrderBook::sizeAheadOf, py::arg("id"),
             "Total size of orders ahead of the given order id in its "
             "own price level's time-priority queue. Raises if the id "
             "isn't currently resting in the book.")
        .def("order_count", &OrderBook::orderCount,
             "Number of currently resting (live) orders.");

    // --- LOBSTER replay, exposed so Python can drive a full historical
    // replay without re-implementing message parsing itself ---

    py::enum_<MessageType>(m, "MessageType")
        .value("NewLimitOrder", MessageType::NewLimitOrder)
        .value("PartialCancel", MessageType::PartialCancel)
        .value("TotalDelete", MessageType::TotalDelete)
        .value("VisibleExecution", MessageType::VisibleExecution)
        .value("HiddenExecution", MessageType::HiddenExecution)
        .value("CrossTrade", MessageType::CrossTrade)
        .value("TradingHalt", MessageType::TradingHalt);

    py::class_<MessageEvent>(m, "MessageEvent")
        .def_readonly("time", &MessageEvent::time)
        .def_readonly("type", &MessageEvent::type)
        .def_readonly("order_id", &MessageEvent::orderId)
        .def_readonly("size", &MessageEvent::size)
        .def_readonly("price", &MessageEvent::price)
        .def_readonly("direction", &MessageEvent::direction);

    m.def("read_message_file", &readMessageFile, py::arg("path"),
          "Parse a LOBSTER-format message CSV into a list of MessageEvent.");

    m.def("apply_message", &applyMessage, py::arg("book"), py::arg("event"),
          "Apply one parsed MessageEvent to an OrderBook.");

    m.def("replay_all", &replayAll, py::arg("book"), py::arg("events"),
          "Apply an entire list of MessageEvents to an OrderBook, in order.");
}
