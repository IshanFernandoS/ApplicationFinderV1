#!/usr/bin/env python3
"""
HEC Application Finder (MVP)
- Canonicalizes high-entropy carbide composition strings
- Builds literature queries (rule-based OR GPT-assisted via OpenAI Responses API)
- Retrieves papers from OpenAlex
- Scores HEC-relevant application tags
- Caches API results in SQLite

Dependencies:
  pip install requests python-dotenv

Environment (.env supported):
  # OpenAlex (recommended; may be required depending on OpenAlex policy)
  OPENALEX_API_KEY=...

  # OpenAI (only needed if --use_gpt_queries)
  OPENAI_API_KEY=...
  OPENAI_MODEL=gpt-4o-mini    # default; supports json_schema Structured Outputs

  # Optional
  CACHE_DB=appfinder_cache.sqlite3

Run examples:
  python hec_app_finder.py "(Ta0.2Nb0.2Ti0.2Sc0.2Hf0.2)C" --top_k 6
  python hec_app_finder.py "(Ta0.2Nb0.2Ti0.2Sc0.2Hf0.2)C" --use_gpt_queries --max_queries 20
  python hec_app_finder.py "(TaNbTiScHf)C" --context "rock salt solid solution" --json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()


# ----------------------------
# SQLite cache
# ----------------------------

class CacheDB:
    def __init__(self, path: str) -> None:
        self.path = path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    k TEXT PRIMARY KEY,
                    v TEXT NOT NULL,
                    ts INTEGER NOT NULL
                )
                """
            )
            conn.commit()

    def get(self, key: str, max_age_seconds: int) -> Optional[Dict[str, Any]]:
        now = int(time.time())
        with sqlite3.connect(self.path) as conn:
            row = conn.execute("SELECT v, ts FROM cache WHERE k = ?", (key,)).fetchone()

        if not row:
            return None

        v_str, ts = row
        if now - int(ts) > max_age_seconds:
            return None

        try:
            return json.loads(v_str)
        except json.JSONDecodeError:
            return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        now = int(time.time())
        v_str = json.dumps(value, ensure_ascii=False)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (k, v, ts) VALUES (?, ?, ?)",
                (key, v_str, now),
            )
            conn.commit()


# ----------------------------
# HTTP helpers
# ----------------------------

def http_get_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 25,
    retries: int = 2,
) -> Dict[str, Any]:
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code} for {r.url}: {r.text[:400]}")
            return r.json()
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(0.6 * (attempt + 1))
            else:
                raise
    raise RuntimeError(f"Request failed: {last_exc}")


def http_post_json(
    url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 60,
    retries: int = 1,
) -> Dict[str, Any]:
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code} for {r.url}: {r.text[:600]}")
            return r.json()
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(0.8 * (attempt + 1))
            else:
                raise
    raise RuntimeError(f"Request failed: {last_exc}")


# ----------------------------
# HEC parsing / canonicalization
# ----------------------------

ELEM_RE = re.compile(r"[A-Z][a-z]?")
HEC_PAREN_RE = re.compile(r"^\s*\((.*?)\)\s*([A-Z][a-z]?)\s*$")  # (cationstuff)Anion, e.g. (...)C

import re

def is_relevant_hec_carbide_paper(
    work: Dict[str, Any],
    canonical: str,
) -> bool:
    """
    Keep papers that are likely about high-entropy CARBIDE CERAMICS (not HEA coatings).

    Strategy:
    - Positive anchors: (a) explicit 'high-entropy carbide', (b) canonical '(TaNbTiScHf)C',
      (c) carbide + ceramic/UHTC/rock salt/solid solution signals.
    - Negative: papers that look like HEA-only (high-entropy alloy) without carbide evidence.
    """
    title = (work.get("title") or "").lower()
    abstract = inverted_index_to_text(work.get("abstract_inverted_index") or {}).lower()
    text = f"{title} {abstract}"

    canon = (canonical or "").lower()

    # Must mention carbide somewhere (avoid HEA-only cladding papers)
    has_carbide = "carbide" in text

    # Strong anchors
    has_hec_phrase = ("high-entropy carbide" in text) or ("high entropy carbide" in text)
    has_canonical = canon and (canon in text)

    # Supporting "ceramic" / structure / UHTC signals
    ceramic_signals = any(
        kw in text
        for kw in [
            "ceramic", "uhtc", "ultra-high temperature", "rock salt", "nacl",
            "solid solution", "single-phase", "entropy-stabilized"
        ]
    )

    # HEA-only negative pattern: "high-entropy alloy" with no carbide
    hea_only = ("high-entropy alloy" in text or "high entropy alloy" in text) and not has_carbide

    if hea_only:
        return False

    # Accept rules:
    # 1) canonical mention
    if has_canonical:
        return True

    # 2) explicit "high-entropy carbide" AND carbide present
    if has_hec_phrase and has_carbide:
        return True

    # 3) carbide + ceramic/structure signals (captures HEC ceramics papers without exact phrase)
    if has_carbide and ceramic_signals:
        return True

    return False

def parse_cations_from_parentheses(content: str) -> List[str]:
    """
    Extract element symbols from a parentheses block like:
      Ta0.2Nb0.2Ti0.2Sc0.2Hf0.2
      TaNbTiScHf
    """
    elems = ELEM_RE.findall(content)
    # preserve order, unique
    seen = set()
    out = []
    for e in elems:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out

def canonicalize_hec(query: str) -> Optional[Dict[str, Any]]:
    """
    If the query looks like a HEC in parentheses: (....)C, return:
      canonical: (TaNbTiScHf)C
      cation_block: TaNbTiScHf
      elements: ["Ta","Nb","Ti","Sc","Hf","C"]
      anion: "C"
    Else return None.
    """
    q = query.strip()

    m = HEC_PAREN_RE.match(q)
    if not m:
        return None

    inside = m.group(1)
    anion = m.group(2)
    cations = parse_cations_from_parentheses(inside)

    if not cations:
        return None

    cation_block = "".join(cations)
    canonical = f"({cation_block}){anion}"

    return {
        "is_hec": True,
        "input": q,
        "canonical": canonical,
        "cation_block": cation_block,
        "anion": anion,
        "cations": cations,
        "elements": cations + [anion],
    }

import sys

# Force UTF-8 output on Windows terminals (prevents UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ----------------------------
# OpenAlex client
# ----------------------------

class OpenAlexClient:
    API = "https://api.openalex.org"

    def __init__(self, api_key: Optional[str], cache: CacheDB) -> None:
        self.api_key = api_key
        self.cache = cache

    def search_works(self, query: str, per_page: int, cache_ttl_days: int = 7) -> List[Dict[str, Any]]:
        q = " ".join(query.split())
        key = f"openalex:works:{q}:{per_page}"
        cached = self.cache.get(key, max_age_seconds=cache_ttl_days * 86400)
        if cached is not None:
            return cached["works"]

        url = f"{self.API}/works"
        params: Dict[str, Any] = {"search": q, "per-page": per_page}
        headers: Dict[str, str] = {"User-Agent": "hec-app-finder-mvp/1.0"}

        if self.api_key:
            params["api_key"] = self.api_key

        data = http_get_json(url, params=params, headers=headers)

        works: List[Dict[str, Any]] = []
        for w in (data.get("results", []) or []):
            primary_loc = w.get("primary_location") or {}
            source = (primary_loc.get("source") or {})
            works.append(
                {
                    "id": w.get("id"),
                    "doi": w.get("doi"),
                    "title": w.get("display_name"),
                    "year": w.get("publication_year"),
                    "venue": source.get("display_name"),
                    "cited_by_count": w.get("cited_by_count") or 0,
                    "abstract_inverted_index": w.get("abstract_inverted_index"),
                }
            )

        self.cache.set(key, {"works": works})
        return works


def inverted_index_to_text(inv: Optional[Dict[str, List[int]]], max_words: int = 50) -> str:
    """
    If OpenAlex provides abstract_inverted_index, reconstruct a short snippet.
    """
    if not inv:
        return ""
    pos_to_word: Dict[int, str] = {}
    for word, positions in inv.items():
        for p in positions:
            pos_to_word[int(p)] = word
    if not pos_to_word:
        return ""
    words = [pos_to_word[i] for i in sorted(pos_to_word.keys())]
    return " ".join(words[:max_words]).strip()


def dedupe_works(works: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for w in works:
        key = w.get("doi") or w.get("id") or w.get("title")
        if not key:
            continue
        k = str(key).lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(w)
    return out


# ----------------------------
# HEC application tags (scoring)
# ----------------------------

@dataclass(frozen=True)
class AppTag:
    name: str
    keywords: Tuple[str, ...]


HEC_TAGS: List[AppTag] = [
    AppTag(
        name="Ultra-high temperature ceramics (UHTC) / thermal protection",
        keywords=("ultra-high temperature", "uhtc", "thermal protection", "hypersonic", "re-entry", "aerospace"),
    ),
    AppTag(
        name="Oxidation / ablation resistance",
        keywords=("oxidation", "oxide scale", "ablation", "arc jet", "thermal shock", "corrosion"),
    ),
    AppTag(
        name="Hard coatings / wear / cutting tools",
        keywords=("hardness", "wear", "coating", "cutting tool", "tribology", "scratch", "fracture toughness"),
    ),
    AppTag(
        name="Sintering / processing (SPS / hot pressing)",
        keywords=("spark plasma sintering", "sps", "hot pressing", "densification", "sintering", "microstructure"),
    ),
    AppTag(
        name="Phase stability / solid solution / rock-salt structure",
        keywords=("solid solution", "single-phase", "rock salt", "nacl", "phase stability", "entropy-stabilized"),
    ),
    AppTag(
        name="Thermal / electrical transport (conductivity)",
        keywords=("thermal conductivity", "electrical conductivity", "transport", "seebeck", "thermoelectric"),
    ),
    AppTag(
        name="Extreme environments / radiation / nuclear (rare but possible)",
        keywords=("radiation", "nuclear", "irradiation", "defect", "swelling", "fuel"),
    ),
]


def compute_hit_score(text: str, keywords: Tuple[str, ...]) -> Tuple[float, List[str]]:
    t = (text or "").lower()
    hits = [kw for kw in keywords if kw.lower() in t]
    # saturating score: 0, 0.5, 0.75, 0.875...
    score = 1.0 - (0.5 ** len(hits)) if hits else 0.0
    return score, hits[:10]


def recency_boost(year: Optional[int], now_year: int) -> float:
    if not year:
        return 0.0
    age = max(0, now_year - int(year))
    if age <= 2:
        return 0.25
    if age <= 5:
        return 0.15
    if age <= 10:
        return 0.08
    return 0.0


def rank_papers_for_tag(works: List[Dict[str, Any]], tag: AppTag, now_year: int) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for w in works:
        title = w.get("title") or ""
        abstract = inverted_index_to_text(w.get("abstract_inverted_index"))
        combined = f"{title} {abstract}"

        hit_s, hits = compute_hit_score(combined, tag.keywords)
        if hit_s <= 0:
            continue

        boost = recency_boost(w.get("year"), now_year)
        cite = int(w.get("cited_by_count") or 0)
        cite_boost = min(0.15, (cite / 200.0) * 0.15)  # cap
        score = min(1.0, 0.7 * hit_s + boost + cite_boost)

        ranked.append(
            {
                **w,
                "match_score": round(score, 3),
                "keyword_hits": hits,
                "abstract_snippet": abstract[:320],
            }
        )

    ranked.sort(key=lambda x: x["match_score"], reverse=True)
    return ranked


def aggregate_tag_score(ranked: List[Dict[str, Any]]) -> float:
    top = ranked[:5]
    if not top:
        return 0.0
    score = 0.0
    weight = 1.0
    for p in top:
        score += weight * float(p["match_score"])
        weight *= 0.65
    score = score / 2.4
    return float(max(0.0, min(1.0, score)))


# ----------------------------
# Query building (rule-based)
# ----------------------------

def build_hec_queries_rulebased(canonical: str, cations: List[str], anion: str, context: str = "") -> List[str]:
    elem_list = " ".join(cations + [anion])
    base = [
        f"{canonical} high-entropy carbide",
        f"{canonical} entropy-stabilized carbide",
        f"{canonical} rock salt solid solution",
        f"high-entropy carbide {elem_list}",
        f"equimolar {canonical} carbide",
    ]
    scenarios = [
        f"{canonical} ultra-high temperature ceramic UHTC",
        f"{canonical} thermal protection hypersonic",
        f"{canonical} oxidation resistance ablation",
        f"{canonical} hardness wear coating",
        f"{canonical} cutting tool coating",
        f"{canonical} spark plasma sintering SPS densification",
        f"{canonical} phase stability single-phase solid solution",
        f"{canonical} microstructure mechanical properties",
    ]
    queries = base + scenarios
    if context.strip():
        queries = [f"{q} {context.strip()}".strip() for q in queries]
    return queries


# ----------------------------
# Query building (GPT-assisted via OpenAI Responses API Structured Outputs)
# ----------------------------

HEC_QUERY_PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "canonical": {"type": "string"},
        "aliases": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
        "themes": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "theme": {"type": "string"},
                    "queries": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
                },
                "required": ["theme", "queries"],
                "additionalProperties": False,
            },
        },
        "all_queries": {"type": "array", "items": {"type": "string"}, "maxItems": 30},
    },
    "required": ["canonical", "aliases", "themes", "all_queries"],
    "additionalProperties": False,
}

def openai_generate_hec_query_plan(
    cache: CacheDB,
    canonical: str,
    cations: List[str],
    anion: str,
    context: str,
    cache_ttl_days: int = 30,
) -> Dict[str, Any]:
    """
    Calls OpenAI Responses API to generate a structured query plan (JSON Schema).
    The response content is expected as JSON text (then we parse it).

    Requires OPENAI_API_KEY in environment.
    """
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for --use_gpt_queries. Put it in .env.")

    model = (os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip()
    elem_list = " ".join(cations + [anion])
    ctx = context.strip()

    cache_key = f"openai:hec_query_plan:{model}:{canonical}:{ctx}"
    cached = cache.get(cache_key, max_age_seconds=cache_ttl_days * 86400)
    if cached is not None:
        return cached["plan"]

    sys_msg = (
        "You are an expert in materials science literature search. "
        "Generate realistic search queries for the given high-entropy carbide (HEC). "
        "Do not invent extra elements or dopants."
    )

    user_msg = f"""
HEC composition:
- canonical: {canonical}
- elements: {elem_list}

Optional context (append to relevant queries if helpful): {ctx if ctx else "(none)"}

Rules for queries:
- Each query must be 6–16 words.
- Each query MUST include at least one anchor:
  - "high-entropy carbide" OR "{canonical}" OR the exact element list "{elem_list}".
- Cover multiple themes: UHTC/thermal protection, oxidation/ablation, coatings/wear, processing (SPS/hot pressing),
  phase stability/rock-salt solid solution, and mechanical properties.
- Output MUST match the JSON schema (no extra keys).
""".strip()

    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "hec_query_plan",
                "schema": HEC_QUERY_PLAN_SCHEMA,
                "strict": True,
            }
        },
    }

    data = http_post_json(
        "https://api.openai.com/v1/responses",
        payload=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=90,
        retries=1,
    )

    # Extract output_text
    out_text: Optional[str] = None
    for item in (data.get("output") or []):
        if item.get("type") == "message":
            content = item.get("content") or []
            for c in content:
                if c.get("type") == "output_text":
                    out_text = c.get("text")
                    break
        if out_text:
            break

    if not out_text:
        raise RuntimeError(f"OpenAI response had no output_text. Raw keys: {list(data.keys())}")

    try:
        plan = json.loads(out_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse OpenAI JSON output: {e}\nRaw:\n{out_text[:800]}")

    cache.set(cache_key, {"plan": plan})
    return plan


def filter_and_limit_queries(
    canonical: str,
    cations: List[str],
    anion: str,
    queries: List[str],
    max_queries: int,
    min_words: int = 4,
    max_words: int =20,
) -> List[str]:
    elem_list = " ".join(cations + [anion]).lower()
    anchors = ["high-entropy carbide", canonical.lower(), elem_list]

    cleaned: List[str] = []
    for q in queries:
        q2 = " ".join((q or "").split())
        if not q2:
            continue
        wcount = len(q2.split())
        if wcount < min_words or wcount > max_words:
            continue

        low = q2.lower()
        # must contain at least one anchor
        if not any(a in low for a in anchors):
            continue

        cleaned.append(q2)

    # dedupe preserve order
    seen = set()
    out: List[str] = []
    for q in cleaned:
        k = q.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(q)
        if len(out) >= max_queries:
            break
    return out


# ----------------------------
# Main pipeline
# ----------------------------

def build_queries_for_input(
    cache: CacheDB,
    raw_query: str,
    context: str,
    use_gpt_queries: bool,
    max_queries: int,
) -> Dict[str, Any]:
    hec = canonicalize_hec(raw_query)

    if not hec:
        # fallback: treat as generic string
        q = " ".join(raw_query.split())
        queries = [q] if not context.strip() else [f"{q} {context.strip()}"]
        return {
            "resolved": {"is_hec": False, "input": raw_query, "canonical": q},
            "queries": queries[:max_queries],
            "query_plan": None,
        }

    canonical = hec["canonical"]
    cations = hec["cations"]
    anion = hec["anion"]

    if use_gpt_queries:
        plan = openai_generate_hec_query_plan(cache, canonical, cations, anion, context)
        raw_queries = plan.get("all_queries") or []
        queries = filter_and_limit_queries(canonical, cations, anion, raw_queries, max_queries=max_queries)
        # if GPT returns too few usable queries, fall back to rule-based
        if len(queries) < max(8, max_queries // 3):
            fallback = build_hec_queries_rulebased(canonical, cations, anion, context=context)
            queries = filter_and_limit_queries(canonical, cations, anion, fallback, max_queries=max_queries)
        return {"resolved": hec, "queries": queries, "query_plan": plan}

    # rule-based
    raw_queries = build_hec_queries_rulebased(canonical, cations, anion, context=context)
    queries = filter_and_limit_queries(canonical, cations, anion, raw_queries, max_queries=max_queries)
    return {"resolved": hec, "queries": queries, "query_plan": None}


def run_app_finder(
    material: str,
    context: str,
    use_gpt_queries: bool,
    max_queries: int,
    papers_per_query: int,
    top_k: int,
    cache_path: str,
) -> Dict[str, Any]:
    cache = CacheDB(cache_path)

    # Build queries (HEC-aware)
    qb = build_queries_for_input(cache, material, context, use_gpt_queries, max_queries)
    resolved = qb["resolved"]
    queries: List[str] = qb["queries"]
    plan = qb["query_plan"]

    # OpenAlex retrieval
    openalex_key = (os.environ.get("OPENALEX_API_KEY") or "").strip() or None
    oa = OpenAlexClient(openalex_key, cache)

    all_works: List[Dict[str, Any]] = []
    for q in queries:
        works = oa.search_works(q, per_page=papers_per_query)
        all_works.extend(works)

    all_works = dedupe_works(all_works)

    # --- HEC carbide relevance filter (paper gating) ---
    canonical = resolved.get("canonical") if isinstance(resolved, dict) else ""
    if canonical:
        all_works = [w for w in all_works if is_relevant_hec_carbide_paper(w, canonical)]


    # Score tags (HEC-specific if HEC, else still use HEC tags as default in this MVP)
    now_year = time.gmtime().tm_year
    tag_results: List[Dict[str, Any]] = []

    for tag in HEC_TAGS:
        ranked = rank_papers_for_tag(all_works, tag, now_year=now_year)
        score = aggregate_tag_score(ranked)
        tag_results.append(
            {
                "tag": tag.name,
                "score": round(score, 3),
                "top_papers": [
                    {
                        "title": p.get("title"),
                        "year": p.get("year"),
                        "venue": p.get("venue"),
                        "doi": p.get("doi"),
                        "match_score": p.get("match_score"),
                        "keyword_hits": p.get("keyword_hits"),
                        "abstract_snippet": p.get("abstract_snippet"),
                    }
                    for p in ranked[:6]
                ],
            }
        )

    tag_results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "input": material,
        "resolved": resolved,
        "context": context,
        "use_gpt_queries": use_gpt_queries,
        "queries_used": queries,
        "query_plan": plan,
        "retrieval_stats": {
            "openalex_queries": len(queries),
            "papers_per_query": papers_per_query,
            "unique_papers": len(all_works),
            "cache_db": cache_path,
        },
        "top_applications": tag_results[:top_k],
        "all_tags": tag_results,
    }


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="HEC Application Finder (OpenAlex + optional GPT query expansion)")
    parser.add_argument("material", help='e.g. "(Ta0.2Nb0.2Ti0.2Sc0.2Hf0.2)C"')
    parser.add_argument("--context", type=str, default="", help="Extra words appended to queries (optional).")
    parser.add_argument("--use_gpt_queries", action="store_true", help="Use OpenAI to propose query variants.")
    parser.add_argument("--max_queries", type=int, default=20, help="Max number of OpenAlex queries to run.")
    parser.add_argument("--papers_per_query", type=int, default=8, help="OpenAlex per-page results per query.")
    parser.add_argument("--top_k", type=int, default=6, help="How many application tags to show.")
    parser.add_argument("--cache", type=str, default=os.environ.get("CACHE_DB", "appfinder_cache.sqlite3"))
    parser.add_argument("--json", action="store_true", help="Print full JSON output.")
    args = parser.parse_args()

    out = run_app_finder(
        material=args.material,
        context=args.context,
        use_gpt_queries=args.use_gpt_queries,
        max_queries=args.max_queries,
        papers_per_query=args.papers_per_query,
        top_k=args.top_k,
        cache_path=args.cache,
    )

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    print("\n=== HEC Application Finder ===")
    print(f"Input: {out['input']}")
    if out["resolved"].get("is_hec"):
        print(f"Canonical: {out['resolved']['canonical']}")
        print(f"Elements: {' '.join(out['resolved']['elements'])}")
    print(f"Context: {out['context'] or '(none)'}")
    print(f"GPT queries: {out['use_gpt_queries']}")
    print(f"Queries used: {len(out['queries_used'])} | Unique papers: {out['retrieval_stats']['unique_papers']}")
    print("\nSample queries:")
    for q in out["queries_used"][:8]:
        print(f"  - {q}")

    print("\nTop application directions (evidence-backed):")
    for i, app in enumerate(out["top_applications"], 1):
        print(f"\n{i}. {app['tag']}  (score={app['score']})")
        for p in app["top_papers"][:3]:
            title = p.get("title") or "(no title)"
            year = p.get("year") or "?"
            venue = p.get("venue") or ""
            hits = ", ".join(p.get("keyword_hits") or [])
            print(f"   - {title} ({year}){(' — ' + venue) if venue else ''}")
            if hits:
                print(f"     hits: {hits}")

    print("\nTip: rerun later—SQLite cache prevents repeated API calls.\n")


if __name__ == "__main__":
    main()