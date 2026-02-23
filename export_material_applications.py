#!/usr/bin/env python3
"""
Export HEC application-finder JSON -> CSV.

Creates:
  1) applications_long.csv  (one row per material-tag)
  2) materials_summary.csv  (one row per material with top tags)

Usage:
  python export_hec_applications.py hec_result_filtered.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List



def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def export_csvs(json_path: Path) -> None:
    raw = json_path.read_bytes()

    # Handle common BOMs (UTF-16/UTF-8-SIG) + fallback
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16")
    elif raw.startswith(b"\xef\xbb\xbf"):
        text = raw.decode("utf-8-sig")
    else:
        text = raw.decode("utf-8")

    data = json.loads(text)


    material_input = data.get("input", "")
    resolved = data.get("resolved", {}) or {}
    canonical = resolved.get("canonical", material_input)
    elements = resolved.get("elements", [])
    context = data.get("context", "")
    use_gpt = bool(data.get("use_gpt_queries", False))

    retrieval_stats = data.get("retrieval_stats", {}) or {}
    unique_papers = retrieval_stats.get("unique_papers", "")
    openalex_queries = retrieval_stats.get("openalex_queries", "")

    all_tags = data.get("all_tags", []) or data.get("top_applications", []) or []

    # --------
    # 1) Long table: one row per (material, tag)
    # --------
    long_rows: List[Dict[str, Any]] = []
    for tag_obj in all_tags:
        tag = tag_obj.get("tag", "")
        score = tag_obj.get("score", 0.0)

        top_papers = tag_obj.get("top_papers", []) or []
        top_titles = [p.get("title", "") for p in top_papers[:3]]
        top_dois = [p.get("doi", "") for p in top_papers[:3]]
        top_years = [p.get("year", "") for p in top_papers[:3]]

        long_rows.append(
            {
                "material_input": material_input,
                "material_canonical": canonical,
                "elements": " ".join(elements) if isinstance(elements, list) else str(elements),
                "context": context,
                "use_gpt_queries": use_gpt,
                "openalex_queries": openalex_queries,
                "unique_papers_after_filter": unique_papers,
                "application_tag": tag,
                "score": score,
                "top1_title": top_titles[0] if len(top_titles) > 0 else "",
                "top1_year": top_years[0] if len(top_years) > 0 else "",
                "top1_doi": top_dois[0] if len(top_dois) > 0 else "",
                "top2_title": top_titles[1] if len(top_titles) > 1 else "",
                "top2_year": top_years[1] if len(top_years) > 1 else "",
                "top2_doi": top_dois[1] if len(top_dois) > 1 else "",
                "top3_title": top_titles[2] if len(top_titles) > 2 else "",
                "top3_year": top_years[2] if len(top_years) > 2 else "",
                "top3_doi": top_dois[2] if len(top_dois) > 2 else "",
            }
        )

    long_out = json_path.with_name("applications_long.csv")
    # utf-8-sig helps Excel display UTF-8 properly on Windows
    with long_out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()) if long_rows else [])
        w.writeheader()
        w.writerows(long_rows)

    # --------
    # 2) Summary: one row per material (top N tags)
    # --------
    # sort by score desc
    tags_sorted = sorted(all_tags, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    topN = 6
    summary = {
        "material_input": material_input,
        "material_canonical": canonical,
        "elements": " ".join(elements) if isinstance(elements, list) else str(elements),
        "context": context,
        "use_gpt_queries": use_gpt,
        "openalex_queries": openalex_queries,
        "unique_papers_after_filter": unique_papers,
    }
    for i in range(topN):
        t = tags_sorted[i] if i < len(tags_sorted) else {}
        summary[f"tag_{i+1}"] = t.get("tag", "")
        summary[f"score_{i+1}"] = t.get("score", "")
        tp = (t.get("top_papers", []) or [])[:1]
        summary[f"evidence_title_{i+1}"] = tp[0].get("title", "") if tp else ""
        summary[f"evidence_doi_{i+1}"] = tp[0].get("doi", "") if tp else ""

    summary_out = json_path.with_name("materials_summary.csv")
    with summary_out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    print(f"Wrote: {long_out}")
    print(f"Wrote: {summary_out}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python export_hec_applications.py <hec_result_filtered.json>")
        raise SystemExit(2)

    json_path = Path(sys.argv[1]).expanduser().resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    export_csvs(json_path)


if __name__ == "__main__":
    main()