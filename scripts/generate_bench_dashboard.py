#!/usr/bin/env python3
"""Generate the benchmark dashboard summary JSON from data.js.

Reads docs/dev/bench/data.js (written by github-action-benchmark),
extracts the latest run, computes human-readable metrics, and writes
docs/dev/bench/summary.json for the dashboard page.

Also usable locally:
    python scripts/generate_bench_dashboard.py [--data path/to/data.js]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Benchmark name → (human label, threshold description, unit conversion)
BENCH_META: dict[str, dict] = {
    "test_record_frame_overhead": {
        "label": "Record Frame",
        "group": "core",
        "threshold": "<5ms",
        "description": "Overhead to serialize and buffer one frame",
        "convert": lambda ns: f"{ns / 1e6:.3f}ms",
        "unit": "ms",
        "raw_convert": lambda ns: ns / 1e6,
    },
    "test_record_realistic_payload": {
        "label": "Record Realistic (~2KB)",
        "group": "core",
        "threshold": "<10ms",
        "description": "Record a typical agent state with messages, plan, context",
        "convert": lambda ns: f"{ns / 1e6:.3f}ms",
        "unit": "ms",
        "raw_convert": lambda ns: ns / 1e6,
    },
    "test_record_large_payload": {
        "label": "Record Large (~5KB)",
        "group": "core",
        "threshold": "<15ms",
        "description": "Record a large state with nested data",
        "convert": lambda ns: f"{ns / 1e6:.3f}ms",
        "unit": "ms",
        "raw_convert": lambda ns: ns / 1e6,
    },
    "test_write_throughput": {
        "label": "Write 10K Frames",
        "group": "core",
        "threshold": ">1000 fps",
        "description": "Sustained write throughput to disk",
        "convert": lambda ns: f"{10000 / (ns / 1e9):.0f} fps",
        "unit": "fps",
        "raw_convert": lambda ns: 10000 / (ns / 1e9),
    },
    "test_load_10k_session": {
        "label": "Load 10K Session",
        "group": "core",
        "threshold": "<500ms",
        "description": "Parse 10,000 frames from JSONL",
        "convert": lambda ns: f"{ns / 1e6:.1f}ms",
        "unit": "ms",
        "raw_convert": lambda ns: ns / 1e6,
    },
    "test_load_realistic_session": {
        "label": "Load 200 Realistic",
        "group": "core",
        "threshold": "<100ms",
        "description": "Parse 200 realistic frames",
        "convert": lambda ns: f"{ns / 1e6:.1f}ms",
        "unit": "ms",
        "raw_convert": lambda ns: ns / 1e6,
    },
    "test_goto_frame": {
        "label": "goto_frame()",
        "group": "time_travel",
        "threshold": "<1ms",
        "description": "Random-access time-travel to any frame",
        "convert": lambda ns: f"{ns / 1e3:.2f}µs",
        "unit": "µs",
        "raw_convert": lambda ns: ns / 1e3,
    },
    "test_compare_frames": {
        "label": "compare_frames()",
        "group": "time_travel",
        "threshold": "<5ms",
        "description": "Diff two frames to find state changes",
        "convert": lambda ns: f"{ns / 1e3:.1f}µs",
        "unit": "µs",
        "raw_convert": lambda ns: ns / 1e3,
    },
    "test_fork_session": {
        "label": "fork()",
        "group": "time_travel",
        "threshold": "<10ms",
        "description": "Fork a session from a given frame",
        "convert": lambda ns: f"{ns / 1e3:.1f}µs",
        "unit": "µs",
        "raw_convert": lambda ns: ns / 1e3,
    },
    "test_get_errors": {
        "label": "get_errors()",
        "group": "time_travel",
        "threshold": "<5ms",
        "description": "Scan 1K frames for error frames",
        "convert": lambda ns: f"{ns / 1e3:.1f}µs",
        "unit": "µs",
        "raw_convert": lambda ns: ns / 1e3,
    },
    "test_ghost_save": {
        "label": "Ghost Save",
        "group": "ghost_replay",
        "threshold": "<50ms",
        "description": "Save a 50-frame golden run to cache",
        "convert": lambda ns: f"{ns / 1e6:.1f}ms",
        "unit": "ms",
        "raw_convert": lambda ns: ns / 1e6,
    },
    "test_ghost_replay": {
        "label": "Ghost Replay",
        "group": "ghost_replay",
        "threshold": "<20ms",
        "description": "Replay 50 frames from cache (zero LLM calls)",
        "convert": lambda ns: f"{ns / 1e6:.1f}ms",
        "unit": "ms",
        "raw_convert": lambda ns: ns / 1e6,
    },
    "test_ghost_fingerprint_lookup": {
        "label": "Cache Lookup",
        "group": "ghost_replay",
        "threshold": "<1ms",
        "description": "Check if a golden run exists (100 cached tasks)",
        "convert": lambda ns: f"{ns / 1e3:.2f}µs",
        "unit": "µs",
        "raw_convert": lambda ns: ns / 1e3,
    },
}


def parse_data_js(path: Path) -> dict:
    """Parse the window.BENCHMARK_DATA = {...} file."""
    text = path.read_text()
    json_str = text.replace("window.BENCHMARK_DATA = ", "", 1).rstrip().rstrip(";")
    return json.loads(json_str)


def extract_latest(data: dict) -> list[dict]:
    """Get benchmarks from the most recent run."""
    entries = list(data.get("entries", {}).values())
    if not entries or not entries[0]:
        return []
    latest = entries[0][-1]
    return latest.get("benches", [])


def build_summary(benches: list[dict]) -> dict:
    """Build a summary dict with human-readable metrics."""
    groups: dict[str, list[dict]] = {
        "core": [],
        "time_travel": [],
        "ghost_replay": [],
    }

    for bench in benches:
        short_name = bench["name"].rsplit("::", 1)[-1]
        meta = BENCH_META.get(short_name)
        if not meta:
            continue

        # bench["value"] is in iter/sec for pytest-benchmark
        # We need to convert: 1/value gives seconds per op, * 1e9 gives ns
        ops_per_sec = bench["value"]
        ns_per_op = (1.0 / ops_per_sec) * 1e9 if ops_per_sec > 0 else 0

        entry = {
            "name": short_name,
            "label": meta["label"],
            "threshold": meta["threshold"],
            "description": meta["description"],
            "value_display": meta["convert"](ns_per_op),
            "value_raw": round(meta["raw_convert"](ns_per_op), 4),
            "unit": meta["unit"],
            "ops_per_sec": round(ops_per_sec, 2),
        }
        groups[meta["group"]].append(entry)

    return {
        "generated_by": "scripts/generate_bench_dashboard.py",
        "groups": groups,
        "total_benchmarks": sum(len(v) for v in groups.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark dashboard data")
    parser.add_argument(
        "--data",
        default="docs/dev/bench/data.js",
        help="Path to data.js from github-action-benchmark",
    )
    parser.add_argument(
        "--output",
        default="docs/dev/bench/summary.json",
        help="Output path for summary JSON",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"⚠ {data_path} not found — skipping dashboard generation")
        sys.exit(0)

    data = parse_data_js(data_path)
    benches = extract_latest(data)

    if not benches:
        print("⚠ No benchmark data found in data.js")
        sys.exit(0)

    summary = build_summary(benches)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"✅ Dashboard summary written to {out_path} ({summary['total_benchmarks']} benchmarks)")


if __name__ == "__main__":
    main()
