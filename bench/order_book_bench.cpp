// Latency/throughput benchmark for the core OrderBook, and specifically
// the empirical test that CAUGHT the O(1)-cancel bug documented in
// order_book.hpp: does per-operation latency stay flat as the number of
// orders resting at a single price level (k) grows, or does it grow
// with k (meaning something's doing a scan rather than a direct index)?
//
// METHODOLOGY, stated honestly: this measures wall-clock time via
// std::chrono::steady_clock in a shared/virtualized container, not a
// pinned-core, isolated benchmarking rig. Absolute nanosecond numbers
// here will be noisier than a real HFT benchmarking setup and should
// NOT be read as "this is what production latency would be." What IS
// meaningful, and is what this benchmark is actually for, is the
// SCALING BEHAVIOR: whether per-op cost grows with k or stays flat.
// That's a property of the algorithm, not the machine, and it's
// visible even through wall-clock noise once k spans several orders
// of magnitude.
//
// Build: g++ -std=c++17 -O2 -Iinclude -o order_book_bench
//            bench/order_book_bench.cpp src/order_book.cpp
// (or: make bench)

#include "order_book.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

using Clock = std::chrono::steady_clock;

namespace {

double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    size_t idx = std::min(v.size() - 1, static_cast<size_t>(p * v.size()));
    return v[idx];
}

struct Stats {
    double mean_ns;
    double p50_ns;
    double p99_ns;
};

Stats summarize(const std::vector<double>& latencies_ns) {
    double total = 0.0;
    for (double x : latencies_ns) total += x;
    return Stats{
        total / latencies_ns.size(),
        percentile(latencies_ns, 0.50),
        percentile(latencies_ns, 0.99),
    };
}

// All n orders land at DISTINCT price levels (P == n). Tests how
// addLimitOrder scales with the number of price levels in the book --
// should grow slowly (O(log P), the map lookup/insert), not linearly.
Stats bench_add_scaling_with_levels(int n) {
    lob::OrderBook book;
    std::vector<double> latencies_ns;
    latencies_ns.reserve(n);
    for (int i = 0; i < n; i++) {
        auto t0 = Clock::now();
        book.addLimitOrder(i + 1, lob::Side::Buy, 1000000 + static_cast<int64_t>(i) * 100, 100, 0.0);
        auto t1 = Clock::now();
        latencies_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
    return summarize(latencies_ns);
}

// All n orders land at the SAME single price level (P == 1). Tests how
// addLimitOrder scales with how many orders are already resting at
// that one level -- should be flat (O(1): list push_back doesn't care
// how long the list already is).
Stats bench_add_scaling_with_level_depth(int n) {
    lob::OrderBook book;
    std::vector<double> latencies_ns;
    latencies_ns.reserve(n);
    for (int i = 0; i < n; i++) {
        auto t0 = Clock::now();
        book.addLimitOrder(i + 1, lob::Side::Buy, 1000000, 100, 0.0);
        auto t1 = Clock::now();
        latencies_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
    return summarize(latencies_ns);
}

// The key benchmark: k orders all resting at ONE price level, deleted
// in REVERSE arrival order (last-added first). Reverse order is
// deliberate -- it's the worst case for a "scan from the front of the
// queue" implementation (each delete has to walk nearly the whole
// remaining queue to find the target), which is exactly the bug that
// was here before OrderLocation started caching a direct iterator (see
// order_book.hpp). If cancel/delete is truly O(1) per operation
// regardless of k, mean/p50/p99 here should stay flat as k grows. If
// it's actually O(k) (a scan), they should grow roughly linearly with k.
Stats bench_cancel_scaling_with_level_depth(int k) {
    lob::OrderBook book;
    for (int i = 1; i <= k; i++) {
        book.addLimitOrder(i, lob::Side::Buy, 1000000, 100, 0.0);
    }
    std::vector<double> latencies_ns;
    latencies_ns.reserve(k);
    for (int i = k; i >= 1; i--) {
        auto t0 = Clock::now();
        book.deleteOrder(i);
        auto t1 = Clock::now();
        latencies_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
    return summarize(latencies_ns);
}

void print_row(const char* label, int n, const Stats& s) {
    std::printf("%-10s %8d %14.1f %14.1f %14.1f\n", label, n, s.mean_ns, s.p50_ns, s.p99_ns);
}

}  // namespace

int main() {
    std::printf("%-10s %8s %14s %14s %14s\n", "op", "n", "mean(ns)", "p50(ns)", "p99(ns)");
    std::printf("--------------------------------------------------------------\n");

    std::printf("\n-- add, growing number of DISTINCT price levels (expect slow growth, O(log P)) --\n");
    for (int n : {1000, 10000, 100000}) print_row("add/levels", n, bench_add_scaling_with_levels(n));

    std::printf("\n-- add, growing orders at ONE price level (expect flat, O(1)) --\n");
    for (int n : {1000, 10000, 100000}) print_row("add/depth", n, bench_add_scaling_with_level_depth(n));

    std::printf("\n-- cancel, growing orders at ONE price level, worst-case reverse-order deletion --\n");
    std::printf("-- (expect flat, O(1), after the OrderLocation-caches-iterator fix; would grow --\n");
    std::printf("-- roughly linearly with k on the old scan-based implementation) --\n");
    for (int k : {10, 100, 1000, 10000, 50000}) print_row("cancel", k, bench_cancel_scaling_with_level_depth(k));

    return 0;
}
