#!/usr/bin/env bash
# Regenerate the data and every committed result into $OUT (default repro_out/),
# so a rerun never overwrites docs/ by accident. Compare, then copy deliberately.
#
#   scripts/reproduce.sh            data, tests, base study, walk-forward, queue, volatility, quoting
#   CPP=1 scripts/reproduce.sh      also the C++ incremental engine and its diff against SQL (Linux)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
OUT=${OUT:-repro_out}
PY=${PYTHON:-python3}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

[[ -d data/messages ]] || $PY -m scripts.generate_data          # ~1 min, 220 MB of Parquet under data/
[[ -d data/queue_positions ]] || $PY -m scripts.compute_queue_positions
$PY -m pytest tests -q

mkdir -p "$OUT/docs"
cd "$OUT"   # the study scripts write docs/... relative to the working directory
D="--data-dir $ROOT/data"
$PY -m scripts.run_research $D | tee docs/benchmark_output.txt
$PY -m scripts.run_walk_forward $D | tee docs/walk_forward_output.txt
$PY -m scripts.run_queue_study $D | tee docs/queue_study_output.txt
$PY -m scripts.run_volatility_study $D | tee docs/volatility_benchmark_output.txt
$PY -m scripts.run_quoting_loop_study $D --quantile 0.20 | tee docs/quoting_loop_output.txt

if [[ "${CPP:-0}" == 1 ]]; then
  cmake -S "$ROOT/cpp" -B "$ROOT/cpp/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$ROOT/cpp/build" -j"$(nproc)"
  "$ROOT/cpp/build/test_incremental" | tee docs/incremental_test_output.txt
  tmp=$(mktemp -d)
  $PY -m scripts.export_messages_flat $D --out "$tmp/messages.bin"
  "$ROOT/cpp/build/incremental_bench" "$tmp/messages.bin" "$tmp/incremental_out.bin" | tee docs/incremental_bench_output.txt
  $PY -m scripts.compare_incremental_to_sql $D --incremental-out "$tmp/incremental_out.bin" | tee docs/incremental_vs_sql_output.txt
  rm -rf "$tmp"
fi
echo "done: outputs in $OUT/"
