#include "lobster_parser.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace lob {

namespace {

MessageType toMessageType(int raw, size_t lineNo) {
    switch (raw) {
        case 1: return MessageType::NewLimitOrder;
        case 2: return MessageType::PartialCancel;
        case 3: return MessageType::TotalDelete;
        case 4: return MessageType::VisibleExecution;
        case 5: return MessageType::HiddenExecution;
        case 6: return MessageType::CrossTrade;
        case 7: return MessageType::TradingHalt;
        default:
            throw std::runtime_error(
                "lobster_parser: unknown message type '" + std::to_string(raw) +
                "' at line " + std::to_string(lineNo) +
                " (LOBSTER type codes are 1-7)");
    }
}

std::vector<MessageEvent> parseStream(std::istream& in) {
    std::vector<MessageEvent> events;
    std::string line;
    size_t lineNo = 0;

    while (std::getline(in, line)) {
        ++lineNo;
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string field;
        std::vector<std::string> cols;
        while (std::getline(ss, field, ',')) cols.push_back(field);

        if (cols.size() != 6) {
            throw std::runtime_error(
                "lobster_parser: expected 6 columns, got " +
                std::to_string(cols.size()) + " at line " + std::to_string(lineNo) +
                " (\"" + line + "\")");
        }

        MessageEvent ev{};
        try {
            ev.time = std::stod(cols[0]);
            int rawType = std::stoi(cols[1]);
            ev.type = toMessageType(rawType, lineNo);
            ev.orderId = std::stoll(cols[2]);
            ev.size = std::stoll(cols[3]);
            ev.price = std::stoll(cols[4]);
            ev.direction = std::stoi(cols[5]);
        } catch (const std::invalid_argument&) {
            throw std::runtime_error(
                "lobster_parser: unparseable numeric field at line " +
                std::to_string(lineNo) + " (\"" + line + "\")");
        }

        if (ev.direction != 1 && ev.direction != -1) {
            throw std::runtime_error(
                "lobster_parser: direction must be +1 or -1, got " +
                std::to_string(ev.direction) + " at line " + std::to_string(lineNo));
        }

        events.push_back(ev);
    }
    return events;
}

} // namespace

std::vector<MessageEvent> readMessageFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("lobster_parser: could not open file: " + path);
    }
    return parseStream(file);
}

std::vector<MessageEvent> parseMessageLines(const std::string& csvContent) {
    std::stringstream ss(csvContent);
    return parseStream(ss);
}

} // namespace lob
