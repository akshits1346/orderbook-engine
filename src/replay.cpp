#include "replay.hpp"

namespace lob {

void applyMessage(OrderBook& book, const MessageEvent& ev) {
    Side side = (ev.direction == 1) ? Side::Buy : Side::Sell;

    switch (ev.type) {
        case MessageType::NewLimitOrder:
            book.addLimitOrder(ev.orderId, side, ev.price, ev.size, ev.time);
            break;

        case MessageType::PartialCancel:
            book.cancelPartial(ev.orderId, ev.size);
            break;

        case MessageType::TotalDelete:
            book.deleteOrder(ev.orderId);
            break;

        case MessageType::VisibleExecution:
        case MessageType::HiddenExecution:
            book.executeOrder(ev.orderId, ev.size);
            break;

        case MessageType::CrossTrade:
        case MessageType::TradingHalt:
            // Deliberate no-op -- see header comment.
            break;
    }
}

void replayAll(OrderBook& book, const std::vector<MessageEvent>& events) {
    for (const auto& ev : events) {
        applyMessage(book, ev);
    }
}

} // namespace lob
