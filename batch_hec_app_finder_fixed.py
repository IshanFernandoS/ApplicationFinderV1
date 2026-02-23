#!/usr/bin/env python3
"""batch_hec_app_finder.py

Batch-run hec_app_finder.py for up to N compositions (default: 10),
save one JSON per material, and export two master CSVs:

1) outputs_csv/applications_long_master.csv   (one row per material-tag)
2) outputs_csv/materials_summary_master.csv   (one row per material)

REQUIREMENTS:
- Put this file in the SAME folder as hec_app_finder.py
- Install deps: pip install requests python-dotenv
- Ensure .env contains OPENALEX_API_KEY (and OPENAI_API_KEY if using --use_gpt_queries)

USAGE:
  python batch_hec_app_finder.py materials.txt --max_n 10 --use_gpt_queries --max_queries 18 --papers_per_query 20
"""


from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv()

# Avoid Windows UnicodeEncodeError when printing
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def read_text_auto(path: Path) -> str:
    """Read text with BOM detection (UTF-16/UTF-8-SIG/UTF-8)."""
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    return raw.decode("utf-8")


def load_hec_module(script_dir: Path):
    """Dynamically import hec_app_finder.py from the same folder.

    Important: we must register the module in sys.modules BEFORE exec_module(),
    otherwise dataclasses/type resolution can crash on some Python versions.
    """
    import importlib.util

    hec_path = script_dir / "hec_app_finder.py"
    if not hec_path.exists():
        raise FileNotFoundError(f"Expected hec_app_finder.py next to this script: {hec_path}")

    name = "hec_app_finder"
    spec = importlib.util.spec_from_file_location(name, hec_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load hec_app_finder.py module spec")

    mod = importlib.util.module_from_spec(spec)

    # ✅ Key fix: register before executing so dataclasses can resolve module globals
    sys.modules[name] = mod

    spec.loader.exec_module(mod)  # type: ignore
    return mod


def safe_filename(name: str) -> str:
    """Make a Windows-safe filename."""
    s = name.strip().replace(" ", "_")
    s = re.sub(r"[\\/:*?\"<>|]", "_", s)
    return s[:140] if len(s) > 140 else s


def extract_rows_from_result(result: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    material_input = result.get("input", "")
    resolved = result.get("resolved", {}) or {}
    canonical = resolved.get("canonical", material_input)
    elements = resolved.get("elements", [])
    context = result.get("context", "")
    use_gpt = bool(result.get("use_gpt_queries", False))

    stats = result.get("retrieval_stats", {}) or {}
    openalex_queries = stats.get("openalex_queries", "")
    unique_papers = stats.get("unique_papers", "")

    tags = result.get("all_tags", []) or result.get("top_applications", []) or []

    long_rows: List[Dict[str, Any]] = []
    for t in tags:
        tag = t.get("tag", "")
        score = t.get("score", "")

        top_papers = t.get("top_papers", []) or []

        def pick(i: int, field: str) -> str:
            if i < len(top_papers):
                v = top_papers[i].get(field, "")
                return "" if v is None else str(v)
            return ""

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
                "top1_title": pick(0, "title"),
                "top1_year": pick(0, "year"),
                "top1_doi": pick(0, "doi"),
                "top2_title": pick(1, "title"),
                "top2_year": pick(1, "year"),
                "top2_doi": pick(1, "doi"),
                "top3_title": pick(2, "title"),
                "top3_year": pick(2, "year"),
                "top3_doi": pick(2, "doi"),
            }
        )

    tags_sorted = sorted(tags, key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
    topN = 6
    summary: Dict[str, Any] = {
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

    return long_rows, summary


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch run HEC application finder (up to N compositions)")
    parser.add_argument("materials_file", help="Text file with one composition per line (materials.txt)")
    parser.add_argument("--max_n", type=int, default=10, help="Run only first N materials (default: 10)")
    parser.add_argument("--use_gpt_queries", action="store_true", help="Use OpenAI to expand queries")
    parser.add_argument("--max_queries", type=int, default=18, help="Max OpenAlex queries per material")
    parser.add_argument("--papers_per_query", type=int, default=20, help="OpenAlex per-page results per query")
    parser.add_argument("--top_k", type=int, default=6, help="How many tags to keep in top_applications")
    parser.add_argument("--cache", type=str, default=os.environ.get("CACHE_DB", "appfinder_cache.sqlite3"))
    parser.add_argument("--out_json_dir", type=str, default="outputs_json")
    parser.add_argument("--out_csv_dir", type=str, default="outputs_csv")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    mod = load_hec_module(script_dir)

    materials_path = Path(args.materials_file).expanduser().resolve()
    text = read_text_auto(materials_path)

    lines: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)

    if not lines:
        raise RuntimeError(f"No materials found in {materials_path}")

    materials = lines[: max(1, args.max_n)]

    out_json_dir = script_dir / args.out_json_dir
    out_csv_dir = script_dir / args.out_csv_dir
    out_json_dir.mkdir(parents=True, exist_ok=True)
    out_csv_dir.mkdir(parents=True, exist_ok=True)

    all_long_rows: List[Dict[str, Any]] = []
    all_summary_rows: List[Dict[str, Any]] = []

    print(f"Running {len(materials)} materials...\n")

    for idx, mat in enumerate(materials, 1):
        try:
            result = mod.run_app_finder(
                material=mat,
                context="",
                use_gpt_queries=bool(args.use_gpt_queries),
                max_queries=int(args.max_queries),
                papers_per_query=int(args.papers_per_query),
                top_k=int(args.top_k),
                cache_path=str(script_dir / args.cache),
            )
        except Exception as e:
            result = {"input": mat, "error": str(e)}

        canonical = (result.get("resolved", {}) or {}).get("canonical") or mat
        fname = safe_filename(str(canonical)) + ".json"
        json_path = out_json_dir / fname

        json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{idx}/{len(materials)}] Saved JSON: {json_path.name}")

        if "error" not in result:
            long_rows, summary_row = extract_rows_from_result(result)
            all_long_rows.extend(long_rows)
            all_summary_rows.append(summary_row)

    long_csv = out_csv_dir / "applications_long_master.csv"
    summary_csv = out_csv_dir / "materials_summary_master.csv"
    write_csv(long_csv, all_long_rows)
    write_csv(summary_csv, all_summary_rows)

    print("\nDone.")
    print(f"Master CSV (long): {long_csv}")
    print(f"Master CSV (summary): {summary_csv}")
    print(f"Per-material JSONs: {out_json_dir}")


if __name__ == "__main__":
    main()
