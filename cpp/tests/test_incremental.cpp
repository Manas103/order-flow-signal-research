// Hand-rolled test suite (no framework dependency, matching this
// portfolio's convention for a controlled C++ test binary). Two kinds of
// check: (1) a tiny, by-hand-computed partition small enough to verify
// every one of the 43 columns against arithmetic worked out on paper, and
// (2) partition-reset / NULL-propagation regressions for cases the
// bit-identical comparison against the real SQL pass (see
// scripts/compare_incremental_to_sql.py) would otherwise have to catch
// the hard way, on 6.4M rows.
#include <cmath>
#include <cstdio>
#include <string>

#include "incremental_features.hpp"

static int g_pass = 0;
static int g_fail = 0;

static void check(bool cond, const std::string& name) {
    if (cond) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("[FAIL] %s\n", name.c_str());
    }
}

static void check_close(double a, double b, double tol, const std::string& name) {
    const bool both_nan = std::isnan(a) && std::isnan(b);
    const bool ok = both_nan || std::fabs(a - b) <= tol;
    check(ok, name + " (" + std::to_string(a) + " vs " + std::to_string(b) + ")");
}

static RawEvent make_event(int64_t seq, int32_t group, int32_t msg_type, int32_t side, double ts,
                            double qty, double size_delta, double bid_px, double bid_sz,
                            double ask_px, double ask_sz) {
    RawEvent ev{};
    ev.event_seq = seq;
    ev.group_id = group;
    ev.msg_type_id = msg_type;
    ev.side_id = side;
    ev.ts = ts;
    ev.qty = qty;
    ev.size_delta = size_delta;
    for (int k = 0; k < 5; ++k) {
        ev.bid_px[k] = bid_px - k;
        ev.bid_sz[k] = (k == 0) ? bid_sz : 0.0;
        ev.ask_px[k] = ask_px + k;
        ev.ask_sz[k] = (k == 0) ? ask_sz : 0.0;
    }
    return ev;
}

// Three events, one partition, only level-1 depth populated (levels 2-5
// are zero-size so they do not perturb the hand worked-out numbers below).
static void hand_worked_small_partition() {
    using namespace incfeat;
    IncrementalFeatureEngine eng;

    // Row 0: ADD, bid 99@10, ask 101@10. No LAG yet: ofi/qi/lag_ret/vol/gap
    // must all be NaN; counts/flow/cum_net_flow are defined from row 0.
    RawEvent e0 = make_event(0, 1, kAdd, 0, 100.0, 0.0, 10.0, 99.0, 10.0, 101.0, 10.0);
    FeatureRow f0 = eng.update(e0);
    check(f0[kSpread] == 2.0, "row0 spread");
    check(f0[kMicroprice] == 100.0, "row0 microprice (balanced book)");
    check(f0[kBidDepthTotal] == 10.0, "row0 bid depth");
    check(f0[kCumNetFlow] == 10.0, "row0 cum_net_flow == first size_delta");
    check(std::isnan(f0[kOfiL1]), "row0 ofi_l1 is NaN (no LAG yet)");
    check(std::isnan(f0[kQiL1]) == false, "row0 qi_l1 is defined (no LAG needed)");
    check(f0[kQiL1] == 0.0, "row0 qi_l1 == 0 (balanced book)");
    check(std::isnan(f0[kLagRet1]), "row0 lag_ret_1 is NaN");
    check(std::isnan(f0[kRealizedVolW20]), "row0 realized_vol is NaN (mid_diff undefined)");
    check(f0[kMsgCountBase + 0 * 3 + 0] == 1.0, "row0 msg_count_add_w20 == 1");
    check(f0[kTradeFlowW10] == 0.0, "row0 trade_flow (not an EXECUTE) == 0");

    // Row 1: EXECUTE, side B (aggressor bought), qty 3, bid moves to
    // 100@7 (size dropped by the 3 filled), ask stays 101@10.
    RawEvent e1 = make_event(1, 1, kExecute, 0, 100.5, 3.0, -3.0, 100.0, 7.0, 101.0, 10.0);
    FeatureRow f1 = eng.update(e1);
    check(f1[kSpread] == 1.0, "row1 spread");
    check_close(f1[kMicroprice], (100.0 * 10.0 + 101.0 * 7.0) / 17.0, 1e-12, "row1 microprice");
    check(f1[kCumNetFlow] == 10.0 + -3.0, "row1 cum_net_flow accumulates");
    // ofi_l1 = (bid_sz1 - LAG(bid_sz1)) - (ask_sz1 - LAG(ask_sz1)) = (7-10)-(10-10) = -3
    check(f1[kOfiL1] == -3.0, "row1 ofi_l1 == -3");
    check_close(f1[kQiL1], (7.0 - 10.0) / 17.0, 1e-12, "row1 qi_l1");
    // mid0 = 100.0, mid1 = (100+101)/2 = 100.5 -> lag_ret_1 = 0.5
    check_close(f1[kLagRet1], 0.5, 1e-12, "row1 lag_ret_1");
    check(std::isnan(f1[kRealizedVolW20]), "row1 realized_vol still NaN (needs 2 mid_diffs)");
    check(f1[kTradeFlowW10] == 3.0, "row1 trade_flow: buy-side execute adds +qty");
    check(f1[kGapMeanW20] == 0.5, "row1 gap_mean == ts1 - ts0");

    // Row 2: EXECUTE, side S (aggressor sold), qty 2, ask moves to
    // 101@6, bid stays 100@7.
    RawEvent e2 = make_event(2, 1, kExecute, 1, 101.25, 2.0, -2.0, 100.0, 7.0, 101.0, 6.0);
    FeatureRow f2 = eng.update(e2);
    check(f2[kTradeFlowW10] == 3.0 + -2.0, "row2 trade_flow sums signed qty");
    // mid1=100.5, mid2=100.5 -> mid_diff for row2 = 0; row1's mid_diff (vs
    // row0) = 0.5. Two mid_diffs now exist (0.5, 0.0): stddev_samp of
    // {0.5, 0.0} = sqrt(((0.5-0.25)^2 + (0.0-0.25)^2) / 1) = sqrt(0.125).
    check_close(f2[kRealizedVolW20], std::sqrt(0.125), 1e-12, "row2 realized_vol_w20 (n=2)");
    check(f2[kMsgCountBase + kExecute * 3 + 0] == 2.0, "row2 msg_count_execute_w20 == 2");

    // A new group_id must reset all state: row 3 in a different partition
    // immediately after row 2 must see NaN for every LAG/window feature,
    // exactly as if it were the first row ever processed.
    RawEvent e3 = make_event(3, 2, kAdd, 0, 50.0, 0.0, 5.0, 50.0, 5.0, 52.0, 5.0);
    FeatureRow f3 = eng.update(e3);
    check(std::isnan(f3[kOfiL1]), "new partition resets ofi_l1 to NaN");
    check(std::isnan(f3[kRealizedVolW20]), "new partition resets realized_vol to NaN");
    check(f3[kCumNetFlow] == 5.0, "new partition resets cum_net_flow");
    check(f3[kMsgCountBase + 0 * 3 + 0] == 1.0, "new partition resets msg_count to 1");
}

// A sliding window must actually evict: push window+2 identical-type
// events and confirm the count caps at window+1, not window+2.
static void sliding_window_eviction() {
    using namespace incfeat;
    IncrementalFeatureEngine eng;
    FeatureRow last{};
    for (int i = 0; i < 25; ++i) {  // window 20 -> capacity 21
        RawEvent ev = make_event(i, 9, kAdd, 0, static_cast<double>(i), 0.0, 1.0, 100.0, 1.0,
                                  101.0, 1.0);
        last = eng.update(ev);
    }
    check(last[kMsgCountBase + 0 * 3 + 0] == 21.0,
          "msg_count_add_w20 caps at window+1 (21) after 25 ADDs");
}

// A NULL level (empty 5th book level) must make qi_agg5 NaN, not silently
// drop that one term and sum the other four (see incremental_features.hpp
// for why this needs an explicit comment: the first draft of this engine
// got this wrong).
static void qi_agg5_null_propagation() {
    using namespace incfeat;
    IncrementalFeatureEngine eng;
    RawEvent ev{};
    ev.event_seq = 0;
    ev.group_id = 11;
    ev.msg_type_id = kAdd;
    for (int k = 0; k < 5; ++k) {
        ev.bid_px[k] = 100.0 - k;
        ev.bid_sz[k] = 5.0;
        ev.ask_px[k] = 101.0 + k;
        ev.ask_sz[k] = 5.0;
    }
    ev.bid_sz[4] = 0.0;
    ev.ask_sz[4] = 0.0;  // level 5 totally empty on both sides -> qi_l5 NULL; levels 1-4 fine
    FeatureRow f = eng.update(ev);
    check(!std::isnan(f[kQiL1]) && !std::isnan(f[kQiL4]), "qi_l1..l4 are defined (levels 1-4 populated)");
    check(std::isnan(f[kQiL5]), "qi_l5 is NaN when level 5 is empty on both sides");
    check(std::isnan(f[kQiAgg5]), "qi_agg5 propagates NaN rather than skipping the one NULL level");
}

// Regression for the one-sided-book bug: an empty touch side used to give a
// mid of half the real price. Now mid, spread and microprice are NaN there,
// the NaN mid_diff rows occupy their slots in the ROWS frame (so the window
// still evicts the right row later), and the moments skip them like SQL NULL.
static void one_sided_book_is_null_and_keeps_frame_alignment() {
    using namespace incfeat;
    IncrementalFeatureEngine eng;
    // row 0: empty ask (seeding), row 1..: two-sided, mid moves 100 -> 101 -> 103
    FeatureRow f0 = eng.update(make_event(0, 0, 0, 0, 0.0, 1, 1, 99.0, 10.0, 0.0, 0.0));
    check(std::isnan(f0[kSpread]) && std::isnan(f0[kMicroprice]), "one-sided row: spread and microprice NaN");
    FeatureRow f1 = eng.update(make_event(1, 0, 0, 1, 1.0, 1, 1, 99.0, 10.0, 101.0, 10.0));
    check(std::isnan(f1[kLagRet1]), "first two-sided row: lag_ret_1 NaN (previous mid is NULL)");
    check(std::isnan(f1[kRealizedVolW20]), "vol NaN with no defined mid_diff yet");
    FeatureRow f2 = eng.update(make_event(2, 0, 0, 1, 2.0, 1, 1, 100.0, 10.0, 102.0, 10.0));
    check_close(f2[kLagRet1], 1.0, 1e-12, "lag_ret_1 = 101 - 100");
    check(std::isnan(f2[kLagRet2]), "lag_ret_2 reaches the one-sided row: NaN");
    FeatureRow f3 = eng.update(make_event(3, 0, 0, 1, 3.0, 1, 1, 102.0, 10.0, 104.0, 10.0));
    check_close(f3[kRealizedVolW20], std::sqrt(0.5), 1e-12, "vol over mid_diff {1, 2} skips the two NULL rows");

    // Frame alignment: a window of 2 rows (w=1). NaN must evict as a row.
    SlidingMoment m;
    m.reset(1);
    m.push(kNaN);
    m.push(4.0);
    check(m.n() == 1 && m.sum() == 4.0, "NaN occupies a slot but not the sum");
    m.push(6.0);  // evicts the NaN row, frame = {4, 6}
    check(m.n() == 2 && m.sum() == 10.0, "evicting a NaN row leaves the sum alone");
    m.push(kNaN);  // evicts 4, frame = {6, NULL}
    check(m.n() == 1 && m.sum() == 6.0, "pushing NaN evicts the oldest real value");
}

int main() {
    hand_worked_small_partition();
    sliding_window_eviction();
    qi_agg5_null_propagation();
    one_sided_book_is_null_and_keeps_frame_alignment();
    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
