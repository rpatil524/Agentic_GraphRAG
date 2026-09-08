"""
compute_stats.py
================
Compute a compact summary of evaluation outputs across Tier 1-4.

Usage:
    python evaluation/compute_stats.py

The script reads available JSON files in EVAL_OUTPUT_DIR (or
./evaluation/results by default)
and writes one consolidated file: evaluation_summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from _bootstrap import output_dir


def read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    out_dir = output_dir()

    files = {
        "tier1": out_dir / "tier1_sample_results.json",
        "tier2_auto": out_dir / "tier2_trajectory_results.json",
        "tier2_human": out_dir / "tier2_trajectory_human_results.json",
        "tier3_auto": out_dir / "tier3_ragas_results.json",
        "tier3_human": out_dir / "tier3_regas_human_dataset_results.json",
        "tier4": out_dir / "tier4_conversational_summary.json",
    }

    summary = {"available_files": {}, "metrics": {}}

    for key, path in files.items():
        data = read_json(path)
        summary["available_files"][key] = path.exists()
        if not data:
            continue

        if key == "tier1":
            summary["metrics"]["tier1"] = data.get("metrics", {})
        elif key in {"tier2_auto", "tier2_human"}:
            summary["metrics"][key] = data.get("metrics", {})
        elif key == "tier3_auto":
            summary["metrics"]["tier3_auto"] = data.get("summary", {})
        elif key == "tier3_human":
            summary["metrics"]["tier3_human"] = data.get("summary", {})
        elif key == "tier4":
            summary["metrics"]["tier4"] = data

    output_path = out_dir / "evaluation_summary.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved consolidated evaluation summary to: {output_path}")


if __name__ == "__main__":
    main()
