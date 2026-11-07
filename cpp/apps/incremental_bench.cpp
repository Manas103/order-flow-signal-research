// Reads the flat binary export (scripts/export_messages_flat.py), runs
// every row through IncrementalFeatureEngine, writes the 43 output
// columns to a binary file in the same row order (for
// scripts/compare_incremental_to_sql.py to diff against the real DuckDB
// SQL pass), and times the per-event update cost after a warm-up pass.
//
// Usage: incremental_bench <input.bin> <output.bin>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "incremental_features.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }

    std::FILE* in = std::fopen(argv[1], "rb");
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }
    std::fseek(in, 0, SEEK_END);
    const long file_size = std::ftell(in);
    std::fseek(in, 0, SEEK_SET);
    const size_t n = static_cast<size_t>(file_size) / sizeof(RawEvent);
    std::vector<RawEvent> events(n);
    const size_t read = std::fread(events.data(), sizeof(RawEvent), n, in);
    std::fclose(in);
    if (read != n) {
        std::fprintf(stderr, "short read: wanted %zu rows, got %zu\n", n, read);
        return 1;
    }
    std::printf("loaded %zu events (%ld bytes)\n", n, file_size);

    // Per-event latency, not amortized: run the whole stream once as a
    // warm-up (populates CPU caches, pages in the whole input and output
    // buffer), then time a second, identical pass and report the median
    // and p99 of the per-event wall time measured with a fresh
    // steady_clock read around every single update call.
    {
        incfeat::IncrementalFeatureEngine warm_engine;
        for (const auto& ev : events) warm_engine.update(ev);
    }

    incfeat::IncrementalFeatureEngine engine;
    std::vector<incfeat::FeatureRow> out(n);
    std::vector<double> latencies_ns(n);
    for (size_t i = 0; i < n; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        out[i] = engine.update(events[i]);
        const auto t1 = std::chrono::steady_clock::now();
        latencies_ns[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
    }

    std::FILE* outf = std::fopen(argv[2], "wb");
    if (!outf) {
        std::fprintf(stderr, "cannot open %s for writing\n", argv[2]);
        return 1;
    }
    for (const auto& row : out) {
        std::fwrite(row.data(), sizeof(double), row.size(), outf);
    }
    std::fclose(outf);

    std::vector<double> sorted_latencies = latencies_ns;
    std::sort(sorted_latencies.begin(), sorted_latencies.end());
    const double median_ns = sorted_latencies[sorted_latencies.size() / 2];
    const double p99_ns = sorted_latencies[static_cast<size_t>(sorted_latencies.size() * 0.99)];
    double mean_ns = 0.0;
    for (double v : latencies_ns) mean_ns += v;
    mean_ns /= static_cast<double>(latencies_ns.size());

    std::printf("events processed: %zu\n", n);
    std::printf("median latency: %.1f ns (%.3f us)\n", median_ns, median_ns / 1000.0);
    std::printf("p99 latency:    %.1f ns (%.3f us)\n", p99_ns, p99_ns / 1000.0);
    std::printf("mean latency:   %.1f ns (%.3f us)\n", mean_ns, mean_ns / 1000.0);
    std::printf("wrote %zu feature rows (%zu doubles each) to %s\n", out.size(),
                static_cast<size_t>(incfeat::kNumFeatures), argv[2]);
    return 0;
}
