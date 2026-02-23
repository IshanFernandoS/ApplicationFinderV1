#!/usr/bin/env python3
"""rank_hec_candidates.py

Implements:
Step 1) Application-finder query: filter a given application_tag and rank materials by score.
Step 2) Final decision score: combine several tag scores into a single weighted score per material.

Input:
  outputs_csv/applications_long_master.csv
(from batch_hec_app_finder_fixed.py)

Outputs (by default, into outputs_csv/):
  - ranked_<tag>.csv (one per requested tag)
  - materials_final_rank.csv (one row per material with weighted final score)

No external deps (uses only Python stdlib).

Examples:
  # Rank by oxidation/ablation
  python rank_hec_candidates.py --in_long outputs_csv/applications_long_master.csv --tag "Oxidation / ablation resistance"

  # Create final ranking (default weights) + also rank default tags
  python rank_hec_candidates.py --in_long outputs_csv/applications_long_master.csv --final

  # Custom weights (tag names must match exactly as in the CSV)
  python rank_hec_candidates.py --in_long outputs_csv/applications_long_master.csv --final \
    --weights "Ultra-high temperature ceramics (UHTC) / thermal protection=0.35;Oxidation / ablation resistance=0.35;Phase stability / solid solution / rock-salt structure=0.20;Sintering / processing (SPS / hot pressing)=0.10"
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Make Windows terminal output safe
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


DEFAULT_TAGS = [
    "Ultra-high temperature ceramics (UHTC) / thermal protection",
    "Oxidation / ablation resistance",
    "Phase stability / solid solution / rock-salt structure",
    "Sintering / processing (SPS / hot pressing)",
]

DEFAULT_WEIGHTS = {
    "Ultra-high temperature ceramics (UHTC) / thermal protection": 0.35,
    "Oxidation / ablation resistance": 0.35,
    "Phase stability / solid solution / rock-salt structure": 0.20,
    "Sintering / processing (SPS / hot pressing)": 0.10,
}


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:80] if len(s) > 80 else s


def to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    # utf-8-sig is Excel-friendly and reads BOM safely
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def parse_weights(weights_str: str) -> Dict[str, float]:
    """Parse: 'tag=0.3;tag2=0.7'"""
    out: Dict[str, float] = {}
    if not weights_str.strip():
        return out
    parts = [p.strip() for p in weights_str.split(";") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        out[k] = float(v)
    return out


def rank_by_tag(rows: List[Dict[str, str]], tag: str) -> List[Dict[str, Any]]:
    filtered = [r for r in rows if (r.get("application_tag", "") or "").strip() == tag]
    ranked = sorted(filtered, key=lambda r: to_float(r.get("score", 0.0)), reverse=True)

    out: List[Dict[str, Any]] = []
    for r in ranked:
        out.append(
            {
                "material_canonical": r.get("material_canonical", ""),
                "score": to_float(r.get("score", 0.0)),
                "elements": r.get("elements", ""),
                "top1_title": r.get("top1_title", ""),
                "top1_year": r.get("top1_year", ""),
                "top1_doi": r.get("top1_doi", ""),
                "top2_title": r.get("top2_title", ""),
                "top2_year": r.get("top2_year", ""),
                "top2_doi": r.get("top2_doi", ""),
                "top3_title": r.get("top3_title", ""),
                "top3_year": r.get("top3_year", ""),
                "top3_doi": r.get("top3_doi", ""),
            }
        )
    return out


def build_material_tag_index(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, Dict[str, str]]]:
    """index[material][tag] = row"""
    index: Dict[str, Dict[str, Dict[str, str]]] = {}
    for r in rows:
        m = (r.get("material_canonical", "") or "").strip()
        t = (r.get("application_tag", "") or "").strip()
        if not m or not t:
            continue
        index.setdefault(m, {})[t] = r
    return index


def compute_final_rank(
    index: Dict[str, Dict[str, Dict[str, str]]],
    weights: Dict[str, float],
) -> List[Dict[str, Any]]:
    materials = sorted(index.keys())
    out: List[Dict[str, Any]] = []

    tags_used = list(weights.keys())
    weight_sum = sum(weights.values()) if weights else 1.0

    for m in materials:
        row_out: Dict[str, Any] = {"material_canonical": m}
        final = 0.0

        any_row = next(iter(index[m].values()), None)
        row_out["elements"] = (any_row.get("elements", "") if any_row else "")

        for t in tags_used:
            r = index[m].get(t)
            s = to_float(r.get("score")) if r else 0.0
            w = weights[t]
            final += w * s

            row_out[f"{slugify(t)}_score"] = round(s, 4)
            row_out[f"{slugify(t)}_doi"] = (r.get("top1_doi", "") if r else "")
            row_out[f"{slugify(t)}_title"] = (r.get("top1_title", "") if r else "")

        row_out["final_score"] = round(final / weight_sum, 4)
        out.append(row_out)

    out.sort(key=lambda r: float(r.get("final_score", 0.0)), reverse=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Step1/Step2 for HEC application finder (rank + weighted final score).")
    ap.add_argument("--in_long", required=True, help="Path to applications_long_master.csv")
    ap.add_argument("--out_dir", default="outputs_csv", help="Where to write ranked CSVs (default: outputs_csv)")
    ap.add_argument("--tag", action="append", default=[], help="Tag to rank by (repeatable). If omitted, uses defaults.")
    ap.add_argument("--final", action="store_true", help="Also compute materials_final_rank.csv (weighted).")
    ap.add_argument("--weights", default="", help="Custom weights: 'tag=0.3;tag2=0.7'")
    args = ap.parse_args()

    in_path = Path(args.in_long).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Not found: {in_path}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(in_path)

    # Step 1: per-tag ranking CSVs
    tags = args.tag if args.tag else list(DEFAULT_TAGS)
    for t in tags:
        ranked = rank_by_tag(rows, t)
        out_path = out_dir / f"ranked_{slugify(t)}.csv"
        write_csv(out_path, ranked)
        print(f"Wrote: {out_path}")

    # Step 2: final weighted score
    if args.final:
        weights = DEFAULT_WEIGHTS.copy()
        custom = parse_weights(args.weights)
        if custom:
            weights = custom

        index = build_material_tag_index(rows)
        final_rows = compute_final_rank(index, weights)
        out_path = out_dir / "materials_final_rank.csv"
        write_csv(out_path, final_rows)
        print(f"Wrote: {out_path}")
        print("Weights used:")
        for k, v in weights.items():
            print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
