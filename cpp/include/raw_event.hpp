// Binary record layout shared, byte-for-byte, with scripts/export_messages_flat.py's
// NumPy structured dtype (align=False, so no field padding on either side):
// sorted by (ticker, date, event_seq), with ticker/date collapsed into one
// integer `group_id` so this engine never has to parse a string. Partition
// boundaries (a new (ticker, date) session) are exactly the points where
// `group_id` changes between consecutive rows, matching the SQL feature
// pass's `PARTITION BY ticker, date`.
#pragma once
#include <cstdint>

#pragma pack(push, 1)
struct RawEvent {
    int64_t event_seq;
    int32_t group_id;
    int32_t msg_type_id;  // 0=ADD, 1=CANCEL, 2=EXECUTE, 3=DELETE
    int32_t side_id;      // 0=B, 1=S, -1=none
    double ts;
    double qty;
    double size_delta;
    double bid_px[5];
    double bid_sz[5];
    double ask_px[5];
    double ask_sz[5];
};
#pragma pack(pop)

static_assert(sizeof(RawEvent) == 8 + 4 + 4 + 4 + 8 + 8 + 8 + 40 + 40 + 40 + 40,
              "RawEvent must stay byte-identical to the exporter's packed NumPy dtype");

enum MsgType { kAdd = 0, kCancel = 1, kExecute = 2, kDelete = 3 };
