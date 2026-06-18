#!/usr/bin/env python3
"""Aggregate PianoMime eval_metrics logs across multiple run directories.

This script scans all logs/eval_metrics_* directories, collects metrics from
metrics/results.csv when available, and falls back to parsing task .log files.

Outputs, by default under logs/summary_all_runs/:
  - all_results.csv: every parsed row from every run
  - latest_by_song_label.csv: latest successful row for each (song, label)
  - summary_by_label_latest.csv: mean/std/min/max over latest_by_song_label
  - summary_by_label_all.csv: mean/std/min/max over all_results
  - coverage_by_label_latest.csv: how many songs each label covers
  - pivot_f1_latest.csv: rows=song, columns=label, values=f1
  - pivot_runtime_latest.csv: rows=song, columns=label, values=runtime_sec

Usage:
  python summarize_metrics_all.py
  python summarize_metrics_all.py --log-root logs --out-dir logs/summary_all_runs
  python summarize_metrics_all.py --dedupe best_f1
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

FLOAT_FIELDS = [
    "precision",
    "recall",
    "f1",
    "runtime_sec",
    "runtime_hl_sec",
    "runtime_ll_sec",
    "wall_runtime_sec",
]

BASE_FIELDS = [
    "run_dir",
    "source",
    "source_mtime",
    "timestamp",
    "song",
    "label",
    "policy",
    "precision",
    "recall",
    "f1",
    "runtime_sec",
    "runtime_hl_sec",
    "runtime_ll_sec",
    "wall_runtime_sec",
    "high_level_status",
    "low_level_status",
    "use_midi",
    "enable_ik",
    "flow_steps",
    "flow_solver",
    "flow_clip_mode",
    "flow_hl_steps",
    "flow_ll_steps",
    "flow_hl_solver",
    "flow_ll_solver",
    "flow_hl_clip_mode",
    "flow_ll_clip_mode",
    "ddim_steps",
    "ddim_hl_steps",
    "ddim_ll_steps",
    "ddpm_hl_iters",
    "ddpm_ll_iters",
    "ae_ckpt",
    "high_level_ckpt",
    "low_level_ckpt",
    "dataset_hl",
    "dataset_ll",
]


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in {"none", "nan", "null"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fmt_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return f"{value:.10g}"


def source_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def normalize_csv_row(row: Dict[str, Any], source: Path, run_dir: Path) -> Optional[Dict[str, Any]]:
    """Normalize rows from metrics/results.csv or older summary CSVs."""
    song = (row.get("song") or "").strip()
    label = (row.get("label") or "").strip()
    if not song or not label:
        return None

    precision = parse_float(row.get("precision"))
    recall = parse_float(row.get("recall"))
    f1 = parse_float(row.get("f1"))
    if precision is None or recall is None or f1 is None:
        return None

    runtime = (
        parse_float(row.get("runtime_total_sec"))
        or parse_float(row.get("runtime_sec"))
        or parse_float(row.get("time"))
        or parse_float(row.get("total_time"))
    )
    runtime_hl = parse_float(row.get("runtime_hl_sec"))
    runtime_ll = parse_float(row.get("runtime_ll_sec"))
    wall_runtime = parse_float(row.get("wall_runtime_sec"))

    out: Dict[str, Any] = {k: "" for k in BASE_FIELDS}
    out.update({
        "run_dir": str(run_dir),
        "source": str(source),
        "source_mtime": source_mtime(source),
        "timestamp": (row.get("timestamp") or "").strip(),
        "song": song,
        "label": label,
        "policy": (row.get("policy") or "").strip(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "runtime_sec": runtime,
        "runtime_hl_sec": runtime_hl,
        "runtime_ll_sec": runtime_ll,
        "wall_runtime_sec": wall_runtime,
    })

    # Preserve useful optional metadata if present.
    for key in BASE_FIELDS:
        if key in out and out[key] != "":
            continue
        if key in row and row[key] is not None:
            out[key] = str(row[key]).strip()

    return out


def read_csv_rows(csv_path: Path, run_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                normalized = normalize_csv_row(row, csv_path, run_dir)
                if normalized is not None:
                    rows.append(normalized)
    except Exception as exc:
        print(f"[warn] Failed to read CSV {csv_path}: {exc}", file=sys.stderr)
    return rows


def parse_value_from_log(text: str, label: str) -> Optional[float]:
    # Matches lines such as "precision: 0.123" or "runtime_sec:      12.345".
    pattern = rf"(?mi)^\s*{re.escape(label)}\s*:?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
    m = re.search(pattern, text)
    if m:
        return parse_float(m.group(1))
    return None


def parse_string_from_log(text: str, label: str) -> str:
    pattern = rf"(?mi)^\s*{re.escape(label)}\s*:?\s*(.*?)\s*$"
    m = re.search(pattern, text)
    return m.group(1).strip() if m else ""


def parse_log_file(log_path: Path, run_dir: Path) -> Optional[Dict[str, Any]]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        print(f"[warn] Failed to read log {log_path}: {exc}", file=sys.stderr)
        return None

    # Prefer the explicit Result block from eval_metrics.py.
    song = parse_string_from_log(text, "song")
    label = parse_string_from_log(text, "label")
    policy = parse_string_from_log(text, "policy")

    # Fallback to worker START line: START <song> <label> enable_ik=...
    if not song or not label:
        m = re.search(r"START\s+(\S+)\s+(\S+)", text)
        if m:
            song = song or m.group(1)
            label = label or m.group(2)

    precision = parse_value_from_log(text, "precision")
    recall = parse_value_from_log(text, "recall")
    f1 = parse_value_from_log(text, "f1")
    runtime = parse_value_from_log(text, "runtime_sec")
    runtime_hl = parse_value_from_log(text, "HL runtime_sec")
    runtime_ll = parse_value_from_log(text, "LL runtime_sec")
    wall_runtime = parse_value_from_log(text, "wall_runtime_sec")

    if not song or not label or precision is None or recall is None or f1 is None:
        return None

    out: Dict[str, Any] = {k: "" for k in BASE_FIELDS}
    out.update({
        "run_dir": str(run_dir),
        "source": str(log_path),
        "source_mtime": source_mtime(log_path),
        "timestamp": datetime.fromtimestamp(source_mtime(log_path)).isoformat(timespec="seconds"),
        "song": song,
        "label": label,
        "policy": policy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "runtime_sec": runtime,
        "runtime_hl_sec": runtime_hl,
        "runtime_ll_sec": runtime_ll,
        "wall_runtime_sec": wall_runtime,
        "high_level_status": parse_string_from_log(text, "HL status"),
        "low_level_status": parse_string_from_log(text, "LL status"),
        "use_midi": parse_string_from_log(text, "use_midi"),
        "enable_ik": parse_string_from_log(text, "enable_ik"),
    })
    return out


def collect_rows(log_root: Path, parse_logs: bool = True) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_sources = set()

    run_dirs = sorted(
        [p for p in log_root.glob("eval_metrics_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )

    for run_dir in run_dirs:
        csv_candidates = [
            run_dir / "metrics" / "results.csv",
            run_dir / "results.csv",
        ]
        found_csv_rows = 0
        for csv_path in csv_candidates:
            if csv_path.exists():
                csv_rows = read_csv_rows(csv_path, run_dir)
                rows.extend(csv_rows)
                found_csv_rows += len(csv_rows)
                seen_sources.add(str(csv_path.resolve()))

        # If a run has no usable results.csv, fall back to parsing top-level task logs.
        # You can force log parsing with --parse-logs-always.
        if parse_logs and found_csv_rows == 0:
            for log_path in sorted(run_dir.glob("*.log")):
                parsed = parse_log_file(log_path, run_dir)
                if parsed is not None:
                    rows.append(parsed)

    return rows


def sort_key(row: Dict[str, Any]) -> Tuple[float, str]:
    mtime = parse_float(row.get("source_mtime")) or 0.0
    timestamp = str(row.get("timestamp") or "")
    return (mtime, timestamp)


def dedupe_rows(rows: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    if mode == "none":
        return rows

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("song", "")), str(row.get("label", "")))].append(row)

    chosen: List[Dict[str, Any]] = []
    for key, group in grouped.items():
        if mode == "latest":
            chosen.append(max(group, key=sort_key))
        elif mode == "best_f1":
            chosen.append(max(group, key=lambda r: parse_float(r.get("f1")) or -1.0))
        else:
            raise ValueError(f"Unknown dedupe mode: {mode}")
    return sorted(chosen, key=lambda r: (str(r.get("label", "")), str(r.get("song", ""))))


def numeric_values(rows: Iterable[Dict[str, Any]], field: str) -> List[float]:
    vals = []
    for row in rows:
        v = parse_float(row.get(field))
        if v is not None:
            vals.append(v)
    return vals


def mean(vals: List[float]) -> Optional[float]:
    return sum(vals) / len(vals) if vals else None


def std(vals: List[float]) -> Optional[float]:
    if len(vals) <= 1:
        return 0.0 if vals else None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))


def summarize_by_label(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("label", ""))].append(row)

    summary: List[Dict[str, Any]] = []
    for label, group in sorted(grouped.items()):
        songs = sorted({str(r.get("song", "")) for r in group})
        out: Dict[str, Any] = {
            "label": label,
            "count": len(group),
            "num_songs": len(songs),
            "songs": ";".join(songs),
        }
        for field in ["precision", "recall", "f1", "runtime_sec", "runtime_hl_sec", "runtime_ll_sec", "wall_runtime_sec"]:
            vals = numeric_values(group, field)
            out[f"{field}_mean"] = mean(vals)
            out[f"{field}_std"] = std(vals)
            out[f"{field}_min"] = min(vals) if vals else None
            out[f"{field}_max"] = max(vals) if vals else None
        summary.append(out)
    return summary


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_pivot(path: Path, rows: List[Dict[str, Any]], value_field: str) -> None:
    songs = sorted({str(r.get("song", "")) for r in rows})
    labels = sorted({str(r.get("label", "")) for r in rows})
    lookup = {(str(r.get("song", "")), str(r.get("label", ""))): r for r in rows}

    out_rows: List[Dict[str, Any]] = []
    for song in songs:
        row: Dict[str, Any] = {"song": song}
        for label in labels:
            v = parse_float(lookup.get((song, label), {}).get(value_field)) if (song, label) in lookup else None
            row[label] = fmt_float(v)
        out_rows.append(row)
    write_csv(path, out_rows, ["song"] + labels)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate PianoMime metrics across multiple logs/eval_metrics_* runs.")
    parser.add_argument("--log-root", default="logs", help="Root logs directory.")
    parser.add_argument("--out-dir", default="logs/summary_all_runs", help="Output directory for summary CSVs.")
    parser.add_argument("--dedupe", choices=["latest", "best_f1", "none"], default="latest",
                        help="How to choose one row per (song,label) for the main summary.")
    parser.add_argument("--no-parse-logs", action="store_true",
                        help="Only use metrics/results.csv files. Do not parse raw .log files when CSVs are absent.")
    args = parser.parse_args()

    log_root = Path(args.log_root)
    out_dir = Path(args.out_dir)
    rows = collect_rows(log_root, parse_logs=not args.no_parse_logs)

    if not rows:
        print(f"[error] No metric rows found under {log_root}", file=sys.stderr)
        return 1

    all_rows = sorted(rows, key=lambda r: (str(r.get("label", "")), str(r.get("song", "")), sort_key(r)))
    deduped = dedupe_rows(all_rows, args.dedupe)

    # Convert float values to compact strings only at write time via csv writer default str().
    write_csv(out_dir / "all_results.csv", all_rows, BASE_FIELDS)
    write_csv(out_dir / f"{args.dedupe}_by_song_label.csv", deduped, BASE_FIELDS)

    summary_all = summarize_by_label(all_rows)
    summary_deduped = summarize_by_label(deduped)
    write_csv(out_dir / "summary_by_label_all.csv", summary_all)
    write_csv(out_dir / f"summary_by_label_{args.dedupe}.csv", summary_deduped)

    coverage_rows = []
    grouped = defaultdict(list)
    for row in deduped:
        grouped[str(row.get("label", ""))].append(row)
    for label, group in sorted(grouped.items()):
        coverage_rows.append({
            "label": label,
            "count": len(group),
            "songs": ";".join(sorted(str(r.get("song", "")) for r in group)),
        })
    write_csv(out_dir / f"coverage_by_label_{args.dedupe}.csv", coverage_rows)

    write_pivot(out_dir / f"pivot_f1_{args.dedupe}.csv", deduped, "f1")
    write_pivot(out_dir / f"pivot_runtime_{args.dedupe}.csv", deduped, "runtime_sec")

    print(f"[ok] Parsed rows: {len(all_rows)}")
    print(f"[ok] Deduped rows ({args.dedupe}): {len(deduped)}")
    print(f"[ok] Wrote outputs to: {out_dir}")
    print(f"[ok] Main summary: {out_dir / ('summary_by_label_' + args.dedupe + '.csv')}")
    print(f"[ok] Latest/best table: {out_dir / (args.dedupe + '_by_song_label.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
