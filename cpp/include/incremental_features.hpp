// Recomputes obsr/features.py's 43 feature columns one event at a time,
// online, instead of in one DuckDB SQL window-function pass over the whole
// partition. Every windowed aggregate (the per-type message counts, the
// realized-vol sum/sumsq, the trade-flow-imbalance sum, the gap mean) is
// maintained as a true O(1)-amortized sliding sum: a fixed-capacity ring
// buffer holds exactly the window's current contents, and each new event
// adds its own value and subtracts whatever value just fell out of the
// window, the same "removable cumulative aggregate" technique a streaming
// window-function implementation uses internally, rather than re-summing
// the window from scratch on every event. See README "Extension: online
// incremental feature path" for why this choice, and for which of the 43
// columns are bit-identical to the SQL pass (the pure per-row and
// LAG-based columns, trivially; the integer-valued windowed sums, exactly,
// because a sum of double-precision-exact integers is exact regardless of
// addition order; the two real-valued windowed aggregates, close but not
// bit-identical, because `ts` is not an exact binary fraction and this
// engine's add/remove summation order is not provably the same as
// DuckDB's internal one).
#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "raw_event.hpp"

namespace incfeat {

inline constexpr int kDepth = 5;
inline constexpr int kCountWindows[] = {20, 50, 100};
inline constexpr int kVolWindows[] = {20, 50, 100};
inline constexpr int kFlowWindows[] = {10, 50, 100};
inline constexpr int kGapWindows[] = {20, 100};
inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Index of every column in FEATURE_COLUMNS (obsr/features.py), in the
// exact order that file defines them, so this engine's output vector and
// the DuckDB SQL pass's output DataFrame line up column for column.
enum ColIdx {
    kSpread = 0,
    kMicroprice,
    kBidDepthTotal,
    kAskDepthTotal,
    kDepthImbalance,
    kOfiL1, kOfiL2, kOfiL3, kOfiL4, kOfiL5,
    kOfiAgg5,
    kQiL1, kQiL2, kQiL3, kQiL4, kQiL5,
    kQiAgg5,
    kBidSz1,
    kAskSz1,
    // msg_count_{add,cancel,execute,delete}_w{20,50,100}, in that nested order
    kMsgCountBase,
    kRealizedVolW20 = kMsgCountBase + 12,
    kRealizedVolW50,
    kRealizedVolW100,
    kTradeFlowW10,
    kTradeFlowW50,
    kTradeFlowW100,
    kGapMeanW20,
    kGapMeanW100,
    kLagRet1,
    kLagRet2,
    kLagRet3,
    kCumNetFlow,
    kNumFeatures
};

static_assert(kNumFeatures == 43, "obsr/features.py defines exactly 43 feature columns");

using FeatureRow = std::array<double, kNumFeatures>;

// A fixed-capacity ring buffer that maintains a running sum (and,
// optionally, a running sum of squares) over the last `capacity` pushed
// rows, i.e. a SQL `ROWS BETWEEN w PRECEDING AND CURRENT ROW` frame, where
// `capacity == w + 1`. A NaN push is a row whose value is NULL: it occupies
// a slot in the frame (so later evictions stay aligned with SQL's row
// frame) but is excluded from the sum and the count, exactly as SQL
// aggregates skip NULL.
class SlidingMoment {
public:
    void reset(int window) {
        capacity_ = static_cast<size_t>(window) + 1;
        buf_.assign(capacity_, 0.0);
        count_ = 0;
        rows_ = 0;
        next_ = 0;
        sum_ = 0.0;
        sumsq_ = 0.0;
    }

    void push(double v) {
        if (rows_ == capacity_) {
            const double evicted = buf_[next_];
            if (!std::isnan(evicted)) {
                sum_ -= evicted;
                sumsq_ -= evicted * evicted;
                --count_;
            }
        } else {
            ++rows_;
        }
        buf_[next_] = v;
        if (!std::isnan(v)) {
            sum_ += v;
            sumsq_ += v * v;
            ++count_;
        }
        next_ = (next_ + 1) % capacity_;
    }

    size_t n() const { return count_; }
    double sum() const { return sum_; }

    double mean() const { return count_ > 0 ? sum_ / static_cast<double>(count_) : kNaN; }

    // Sample standard deviation, DuckDB STDDEV_SAMP's convention: NULL
    // (here NaN) below 2 observations.
    double stddev_samp() const {
        if (count_ < 2) return kNaN;
        const double n = static_cast<double>(count_);
        const double variance = (sumsq_ - sum_ * sum_ / n) / (n - 1.0);
        return std::sqrt(std::max(variance, 0.0));
    }

private:
    std::vector<double> buf_;
    size_t capacity_ = 0;
    size_t count_ = 0;  // non-NaN values in the frame
    size_t rows_ = 0;   // rows in the frame, NaN or not
    size_t next_ = 0;
    double sum_ = 0.0;
    double sumsq_ = 0.0;
};

// Per-partition ((ticker, date) session) state. Reset whenever the input
// stream's group_id changes, exactly matching `PARTITION BY ticker, date`.
struct PartitionState {
    long local_idx = -1;  // event count within this partition, 0-based

    // LAG(mid, j) for j = 1, 2, 3 and LAG(bid_sz_k)/LAG(ask_sz_k) for the
    // OFI terms: a small fixed-depth history, not a window sum.
    double prev_mid[3] = {kNaN, kNaN, kNaN};
    double prev_bid_sz[kDepth] = {kNaN, kNaN, kNaN, kNaN, kNaN};
    double prev_ask_sz[kDepth] = {kNaN, kNaN, kNaN, kNaN, kNaN};
    double prev_ts = kNaN;

    double cum_net_flow = 0.0;

    SlidingMoment msg_count[4][3];  // [msg_type][window index into kCountWindows]
    SlidingMoment vol[3];           // mid_diff, one per kVolWindows
    SlidingMoment flow[3];          // signed_exec_qty, one per kFlowWindows
    SlidingMoment gap[2];           // time_gap, one per kGapWindows

    void reset_windows() {
        for (auto& row : msg_count)
            for (int w = 0; w < 3; ++w) row[w].reset(kCountWindows[w]);
        for (int w = 0; w < 3; ++w) vol[w].reset(kVolWindows[w]);
        for (int w = 0; w < 3; ++w) flow[w].reset(kFlowWindows[w]);
        for (int w = 0; w < 2; ++w) gap[w].reset(kGapWindows[w]);
    }
};

class IncrementalFeatureEngine {
public:
    IncrementalFeatureEngine() { state_.reset_windows(); }

    FeatureRow update(const RawEvent& ev) {
        if (ev.group_id != current_group_) {
            current_group_ = ev.group_id;
            state_ = PartitionState{};
            state_.reset_windows();
        }
        ++state_.local_idx;

        FeatureRow out{};
        out.fill(kNaN);

        const double bid_px1 = ev.bid_px[0], ask_px1 = ev.ask_px[0];
        const double bid_sz1 = ev.bid_sz[0], ask_sz1 = ev.ask_sz[0];
        // One-sided book (either touch empty): no mid, spread or microprice,
        // matching the SQL guard in obsr/features.py.
        const bool one_sided = bid_sz1 == 0.0 || ask_sz1 == 0.0;
        const double spread = one_sided ? kNaN : ask_px1 - bid_px1;
        const double denom1 = bid_sz1 + ask_sz1;
        const double microprice =
            one_sided ? kNaN : (bid_px1 * ask_sz1 + ask_px1 * bid_sz1) / denom1;

        double bid_depth_total = 0.0, ask_depth_total = 0.0;
        for (int k = 0; k < kDepth; ++k) {
            bid_depth_total += ev.bid_sz[k];
            ask_depth_total += ev.ask_sz[k];
        }
        const double depth_den = bid_depth_total + ask_depth_total;
        const double depth_imbalance =
            depth_den == 0.0 ? kNaN : (bid_depth_total - ask_depth_total) / depth_den;

        out[kSpread] = spread;
        out[kMicroprice] = microprice;
        out[kBidDepthTotal] = bid_depth_total;
        out[kAskDepthTotal] = ask_depth_total;
        out[kDepthImbalance] = depth_imbalance;
        out[kBidSz1] = bid_sz1;
        out[kAskSz1] = ask_sz1;

        // ofi_agg5 / qi_agg5 are plain SQL `+` expressions over the five
        // per-level terms, not NULL-skipping aggregates: a single NULL
        // level (e.g. an empty 5th book level making qi_l5 NULL) must make
        // the whole sum NULL. IEEE 754 addition propagates NaN the same
        // way SQL's NULL propagates through `+`, so a plain running sum
        // here already matches that semantics with no extra guard needed
        // (see tests/test_incremental.cpp for the regression this covers).
        double ofi_agg5 = 0.0, qi_agg5 = 0.0;
        for (int k = 0; k < kDepth; ++k) {
            const double ofi_k = (ev.bid_sz[k] - state_.prev_bid_sz[k]) -
                                  (ev.ask_sz[k] - state_.prev_ask_sz[k]);
            const double qi_den = ev.bid_sz[k] + ev.ask_sz[k];
            const double qi_k = qi_den == 0.0 ? kNaN : (ev.bid_sz[k] - ev.ask_sz[k]) / qi_den;
            out[kOfiL1 + k] = std::isnan(state_.prev_bid_sz[k]) ? kNaN : ofi_k;
            out[kQiL1 + k] = qi_k;
            ofi_agg5 += out[kOfiL1 + k];
            qi_agg5 += qi_k;
        }
        out[kOfiAgg5] = ofi_agg5;
        out[kQiAgg5] = qi_agg5 / 5.0;

        const double mid = one_sided ? kNaN : (bid_px1 + ask_px1) / 2.0;
        const double mid_diff = std::isnan(state_.prev_mid[0]) ? kNaN : mid - state_.prev_mid[0];
        for (int j = 0; j < 3; ++j) {
            out[kLagRet1 + j] = std::isnan(state_.prev_mid[j]) ? kNaN : mid - state_.prev_mid[j];
        }

        const double time_gap = std::isnan(state_.prev_ts) ? kNaN : ev.ts - state_.prev_ts;

        state_.cum_net_flow += ev.size_delta;
        out[kCumNetFlow] = state_.cum_net_flow;

        const double signed_exec_qty =
            ev.msg_type_id == kExecute ? (ev.side_id == 0 ? ev.qty : -ev.qty) : 0.0;

        for (int t = 0; t < 4; ++t) {
            const double indicator = (ev.msg_type_id == t) ? 1.0 : 0.0;
            for (int w = 0; w < 3; ++w) {
                state_.msg_count[t][w].push(indicator);
                out[kMsgCountBase + t * 3 + w] = state_.msg_count[t][w].sum();
            }
        }
        for (int w = 0; w < 3; ++w) {
            // mid_diff is NULL on the first row of a partition and on any row
            // where this or the previous mid is NULL (one-sided book). Push it
            // anyway: the frame is ROWS-based, so a NULL row still occupies a
            // slot, and SlidingMoment leaves it out of the moments.
            state_.vol[w].push(mid_diff);
            out[kRealizedVolW20 + w] = state_.vol[w].stddev_samp();
        }
        for (int w = 0; w < 3; ++w) {
            state_.flow[w].push(signed_exec_qty);
            out[kTradeFlowW10 + w] = state_.flow[w].sum();
        }
        for (int w = 0; w < 2; ++w) {
            state_.gap[w].push(time_gap);
            out[kGapMeanW20 + w] = state_.gap[w].mean();
        }

        for (int k = 0; k < kDepth; ++k) {
            state_.prev_bid_sz[k] = ev.bid_sz[k];
            state_.prev_ask_sz[k] = ev.ask_sz[k];
        }
        state_.prev_mid[2] = state_.prev_mid[1];
        state_.prev_mid[1] = state_.prev_mid[0];
        state_.prev_mid[0] = mid;
        state_.prev_ts = ev.ts;

        return out;
    }

private:
    int current_group_ = -1;
    PartitionState state_;
};

}  // namespace incfeat
