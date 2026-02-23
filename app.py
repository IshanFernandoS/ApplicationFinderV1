import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# =========================
# Page config + style
# =========================
st.set_page_config(page_title="HEC Application Finder", page_icon="🧪", layout="wide")

st.title("🧪 HEC Application Finder")
st.caption("Generate candidates → retrieve literature → discover application tags → score & explain")

PROJECT_DIR = Path(__file__).resolve().parent
OUT_JSON_DIR = PROJECT_DIR / "outputs_json"
OUT_CSV_DIR = PROJECT_DIR / "outputs_csv"
OUT_JSON_DIR.mkdir(exist_ok=True)
OUT_CSV_DIR.mkdir(exist_ok=True)

# Soft UI styling (no extra deps)
st.markdown(
    """
<style>
.stButton>button { border-radius: 12px; padding: 0.5rem 0.9rem; }
.card {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 18px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.03);
}
.small { opacity: 0.8; font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# Import your pipeline
# =========================
try:
    import hec_app_finder as hec  # expects hec_app_finder.py in same folder
except Exception as e:
    st.error(
        "Could not import hec_app_finder.py. Put this app file in the SAME folder as hec_app_finder.py.\n\n"
        f"Import error: {e}"
    )
    st.stop()

# =========================
# Constants + helpers
# =========================
HEC_RE = re.compile(r"^\([A-Z][a-z]?\d*\.?\d*(?:[A-Z][a-z]?\d*\.?\d*)+\)[A-Z][a-z]?$")
ELEM_RE = re.compile(r"[A-Z][a-z]?")

OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY") or "").strip()
OPENAI_MODEL = (os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip()

def doi_link(doi: str) -> str:
    if not isinstance(doi, str) or not doi.strip():
        return ""
    d = doi.strip()
    if d.startswith("http://") or d.startswith("https://"):
        return d
    return f"https://doi.org/{d}"

def sanitize_filename(s: str) -> str:
    s = (s or "").strip()
    s = s.replace(" ", "_")
    s = re.sub(r"[\\/:*?\"<>|]", "_", s)
    return s[:140] if len(s) > 140 else s

def as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        s = str(x).strip()
        return float(s) if s else default
    except Exception:
        return default

def paper_text(p: Dict[str, Any]) -> str:
    t = (p.get("title") or "")
    a = (p.get("abstract_snippet") or "")
    return f"{t} {a}".lower()

def compute_hit_score(text: str, keywords: List[str]) -> Tuple[float, List[str]]:
    t = (text or "").lower()
    hits = []
    for kw in keywords:
        k = (kw or "").strip().lower()
        if not k:
            continue
        if k in t:
            hits.append(kw)
    score = 1.0 - (0.5 ** len(hits)) if hits else 0.0
    return score, hits[:10]

def recency_boost(year: Any, now_year: int) -> float:
    try:
        y = int(year)
    except Exception:
        return 0.0
    age = max(0, now_year - y)
    if age <= 2:
        return 0.25
    if age <= 5:
        return 0.15
    if age <= 10:
        return 0.08
    return 0.0

def aggregate_tag_score(paper_scores: List[float]) -> float:
    top = sorted(paper_scores, reverse=True)[:5]
    if not top:
        return 0.0
    score = 0.0
    w = 1.0
    for s in top:
        score += w * s
        w *= 0.65
    score /= 2.4
    return max(0.0, min(1.0, score))

def collect_material_papers(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    papers: List[Dict[str, Any]] = []
    seen = set()

    if isinstance(result.get("papers"), list):
        for p in result["papers"]:
            key = (p.get("doi") or p.get("title") or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            papers.append(p)
        return papers

    tags = result.get("all_tags") or result.get("top_applications") or []
    for t in tags:
        for p in (t.get("top_papers") or []):
            key = (p.get("doi") or p.get("title") or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            papers.append(p)
    return papers

def write_json_utf8(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

# =========================
# GPT: candidate generation
# =========================
CAND_SCHEMA = {
    "type": "object",
    "properties": {
        "family_interpretation": {"type": "string"},
        "materials": {
            "type": "array",
            "minItems": 1,
            "maxItems": 200,
            "items": {
                "type": "object",
                "properties": {
                    "composition": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["composition", "notes"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["family_interpretation", "materials"],
    "additionalProperties": False,
}

def canonical_cation_key(hec_str: str) -> str:
    """
    For dedupe: treat different ordering as the same composition.
    Extract element symbols from inside parentheses and sort them.
    """
    s = hec_str.strip()
    if not s.startswith("(") or ")" not in s:
        return s.lower()
    inside = s[s.find("(") + 1 : s.find(")")]
    cations = ELEM_RE.findall(inside)
    uniq = sorted(set(cations))
    # anion = after ')'
    anion = s[s.find(")") + 1 :].strip()
    return ("+".join(uniq) + "|" + anion).lower()

def openai_generate_candidates(
    family: str,
    element_pool: List[str],
    k: int,
    anion: str,
    n: int,
) -> Tuple[str, List[str]]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing in .env (needed for candidate generation).")

    frac = 1.0 / float(k)
    frac_str = f"{frac:.3g}"  # e.g., 0.2 for k=5

    user_msg = f"""
Generate candidate compositions for high-entropy carbides (rock-salt / NaCl structure).

Family description:
{family}

Hard constraints:
- Use ONLY these cations (no extra elements): {", ".join(element_pool)}
- Exactly k={k} distinct cations per composition.
- Anion must be: {anion}
- Equimolar fractions: each cation fraction = {frac_str}
- Format EXACTLY like: (Ta0.2Nb0.2Ti0.2Sc0.2Hf0.2){anion}
- Produce at least {n} UNIQUE compositions (order-only duplicates count as duplicates).
- Avoid duplicates.

Return JSON strictly matching the schema.
""".strip()

    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": "You are a materials scientist. Output only schema-valid JSON."},
            {"role": "user", "content": user_msg},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "hec_candidates",
                "schema": CAND_SCHEMA,
                "strict": True,
            }
        },
    }

    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=90,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI HTTP {r.status_code}: {r.text[:800]}")

    data = r.json()

    out_text = None
    for item in (data.get("output") or []):
        if item.get("type") == "message":
            for c in (item.get("content") or []):
                if c.get("type") == "output_text":
                    out_text = c.get("text")
                    break
        if out_text:
            break

    if not out_text:
        raise RuntimeError("No output_text from OpenAI.")

    obj = json.loads(out_text)
    interp = obj.get("family_interpretation", "")

    raw = [m.get("composition", "").strip() for m in (obj.get("materials") or [])]
    out: List[str] = []
    seen = set()

    for c in raw:
        if not c or not HEC_RE.match(c):
            continue
        key = canonical_cation_key(c)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= n:
            break

    return interp, out

# =========================
# GPT: dynamic tag discovery
# =========================
TAG_SCHEMA = {
    "type": "object",
    "properties": {
        "tag_set_name": {"type": "string"},
        "rationale": {"type": "string"},
        "tags": {
            "type": "array",
            "minItems": 4,
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "tag": {"type": "string"},
                    "description": {"type": "string"},
                    "keywords": {"type": "array", "minItems": 4, "maxItems": 16, "items": {"type": "string"}},
                },
                "required": ["tag", "description", "keywords"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["tag_set_name", "rationale", "tags"],
    "additionalProperties": False,
}

def openai_discover_tags(family: str, corpus: List[Dict[str, Any]], max_tags: int = 8) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing in .env (needed for dynamic tags).")

    lines = []
    for i, p in enumerate(corpus[:180], 1):
        title = (p.get("title") or "").strip()
        snippet = (p.get("abstract_snippet") or "").strip()
        year = p.get("year") or ""
        if not title:
            continue
        if snippet:
            snippet = snippet[:220]
        lines.append(f"{i}. {title} ({year}) — {snippet}")
    corpus_text = "\n".join(lines)

    user_msg = f"""
You will propose an application-tag taxonomy for high-entropy carbides based on a small literature corpus.

Family description:
{family}

Corpus (titles + short snippets):
{corpus_text}

Requirements:
- Return 6 to {max_tags} application tags that appear in or are strongly implied by this corpus.
- Tags should be short, clear, and suitable as UI labels.
- For each tag, provide 6–14 keywords/phrases that can be used for matching in titles/abstract snippets.
- Avoid overly generic tags like "Materials" or "Ceramics". Prefer application/property directions IF supported by corpus.
- Do not invent unrelated domains not evidenced by the corpus.

Return JSON strictly matching the schema.
""".strip()

    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": "You are an expert materials scientist. Output only schema-valid JSON."},
            {"role": "user", "content": user_msg},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "dynamic_hec_tags",
                "schema": TAG_SCHEMA,
                "strict": True,
            }
        },
    }

    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=90,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI HTTP {r.status_code}: {r.text[:800]}")

    data = r.json()

    out_text = None
    for item in (data.get("output") or []):
        if item.get("type") == "message":
            for c in (item.get("content") or []):
                if c.get("type") == "output_text":
                    out_text = c.get("text")
                    break
        if out_text:
            break

    if not out_text:
        raise RuntimeError("No output_text from OpenAI.")

    obj = json.loads(out_text)
    obj["tags"] = obj.get("tags", [])[:max_tags]
    return obj

# =========================
# Dynamic scoring
# =========================
def score_material_dynamic(papers: List[Dict[str, Any]], tags: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    now_year = time.gmtime().tm_year
    out = []
    for tag in tags:
        tag_name = tag["tag"]
        keywords = tag.get("keywords") or []
        ranked_papers = []

        for p in papers:
            txt = paper_text(p)
            hit_s, hits = compute_hit_score(txt, keywords)
            if hit_s <= 0:
                continue
            boost = recency_boost(p.get("year"), now_year)
            cite = as_float(p.get("cited_by_count"), 0.0)
            cite_boost = min(0.15, (cite / 200.0) * 0.15)
            score = min(1.0, 0.7 * hit_s + boost + cite_boost)
            ranked_papers.append(
                {
                    "title": p.get("title"),
                    "year": p.get("year"),
                    "venue": p.get("venue"),
                    "doi": p.get("doi"),
                    "match_score": round(score, 3),
                    "keyword_hits": hits,
                    "abstract_snippet": (p.get("abstract_snippet") or "")[:300],
                }
            )

        ranked_papers.sort(key=lambda x: x["match_score"], reverse=True)
        tag_score = aggregate_tag_score([p["match_score"] for p in ranked_papers])

        out.append(
            {
                "tag": tag_name,
                "description": tag.get("description", ""),
                "score": round(tag_score, 3),
                "top_papers": ranked_papers[:6],
            }
        )

    out.sort(key=lambda x: x["score"], reverse=True)
    return out

def build_dynamic_long_table(material_results: Dict[str, Dict[str, Any]], dynamic_scores: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for canonical, res in material_results.items():
        resolved = res.get("resolved", {}) or {}
        elements = resolved.get("elements", [])
        elem_str = " ".join(elements) if isinstance(elements, list) else str(elements)
        for t in dynamic_scores.get(canonical, []):
            top = t.get("top_papers", []) or []
            def pick(i: int, field: str) -> str:
                if i < len(top):
                    v = top[i].get(field, "")
                    return "" if v is None else str(v)
                return ""
            rows.append(
                {
                    "material_canonical": canonical,
                    "elements": elem_str,
                    "application_tag": t.get("tag", ""),
                    "description": t.get("description", ""),
                    "score": as_float(t.get("score"), 0.0),
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
    df = pd.DataFrame(rows)
    if not df.empty:
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    return df

def compute_final_score_from_dynamic(df_long: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    pivot = df_long.pivot_table(
        index=["material_canonical", "elements"],
        columns="application_tag",
        values="score",
        aggfunc="max",
        fill_value=0.0,
    ).reset_index()
    wsum = sum(weights.values()) if weights else 1.0
    total = 0.0
    for tag, w in weights.items():
        if tag not in pivot.columns:
            pivot[tag] = 0.0
        total += w * pivot[tag].astype(float)
    pivot["final_score"] = (total / wsum).round(4)
    pivot.sort_values("final_score", ascending=False, inplace=True)
    return pivot

# =========================
# Session state
# =========================
if "materials" not in st.session_state:
    st.session_state.materials = []
if "family" not in st.session_state:
    st.session_state.family = "Equimolar refractory high-entropy carbides for UHTC"
if "results" not in st.session_state:
    st.session_state.results = {}
if "dynamic_tagset" not in st.session_state:
    st.session_state.dynamic_tagset = None
if "dynamic_scores" not in st.session_state:
    st.session_state.dynamic_scores = {}
if "dynamic_long" not in st.session_state:
    st.session_state.dynamic_long = pd.DataFrame()
if "dynamic_final" not in st.session_state:
    st.session_state.dynamic_final = pd.DataFrame()

# =========================
# Sidebar settings
# =========================
st.sidebar.header("Pipeline")
use_gpt_queries = st.sidebar.checkbox("GPT-assisted query building (OpenAlex)", value=True)
max_queries = st.sidebar.slider("Max OpenAlex queries per material", 6, 30, 18, 1)
papers_per_query = st.sidebar.slider("Papers per query", 5, 50, 20, 5)
top_k = st.sidebar.slider("Top-K fixed tags (internal)", 3, 10, 6, 1)

st.sidebar.divider()
st.sidebar.header("Dynamic tags")
max_dynamic_tags = st.sidebar.slider("Number of dynamic tags", 6, 12, 8, 1)

# =========================
# Tabs
# =========================
tabs = st.tabs(["1) Candidates", "2) Run", "3) Discover tags", "4) Application Finder"])

# -------------------------
# Tab 1
# -------------------------
with tabs[0]:
    st.subheader("1) Candidate list generation")

    st.session_state.family = st.text_area(
        "Materials family description (also used for tag discovery)",
        value=st.session_state.family,
        height=80,
    )

    with st.expander("🧠 Generate candidates with GPT", expanded=True):
        if not OPENAI_API_KEY:
            st.warning("OPENAI_API_KEY is missing in .env. Add it to enable candidate generation.")
        pool_str = st.text_input("Allowed cation pool (comma-separated)", value="Ta,Nb,Ti,Sc,Hf,Zr,V,Mo,W")
        colA, colB, colC = st.columns(3)
        with colA:
            k = st.number_input("k (number of cations)", min_value=3, max_value=10, value=5, step=1)
        with colB:
            anion = st.text_input("Anion", value="C")
        with colC:
            n = st.number_input("How many candidates", min_value=1, max_value=50, value=10, step=1)

        if st.button("Generate candidates", type="primary", disabled=not bool(OPENAI_API_KEY)):
            try:
                pool = [x.strip() for x in pool_str.split(",") if x.strip()]
                interp, comps = openai_generate_candidates(st.session_state.family, pool, int(k), anion.strip(), int(n))
                st.info(interp or "Generated candidates.")
                if comps:
                    st.session_state.materials = comps
                    st.success(f"Generated {len(comps)} candidates.")
                else:
                    st.warning("Generated zero valid compositions. Try widening the element pool.")
            except Exception as e:
                st.error(str(e))

    st.write("Edit compositions below (one per line):")
    if not st.session_state.materials:
        st.session_state.materials = ["(Ta0.2Nb0.2Ti0.2Sc0.2Hf0.2)C"]

    editable = st.text_area("materials.txt", value="\n".join(st.session_state.materials), height=220)
    st.session_state.materials = [x.strip() for x in editable.splitlines() if x.strip()]

    invalid = [m for m in st.session_state.materials if not HEC_RE.match(m)]
    if invalid:
        st.warning("Some lines do not match the expected HEC format. Best to fix them:")
        st.code("\n".join(invalid))

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download materials.txt",
            data=("\n".join(st.session_state.materials) + "\n").encode("utf-8"),
            file_name="materials.txt",
            mime="text/plain",
        )
    with col2:
        if st.button("🧹 Clear results (keep materials)"):
            st.session_state.results = {}
            st.session_state.dynamic_tagset = None
            st.session_state.dynamic_scores = {}
            st.session_state.dynamic_long = pd.DataFrame()
            st.session_state.dynamic_final = pd.DataFrame()
            st.success("Cleared app session results.")

# -------------------------
# Tab 2
# -------------------------
with tabs[1]:
    st.subheader("2) Run literature retrieval and filtering")

    mats = st.session_state.materials
    if not mats:
        st.warning("Add candidates in Tab 1 first.")
    else:
        st.write(f"Materials queued: **{len(mats)}**")

        if st.button("▶️ Run pipeline now", type="primary"):
            st.session_state.results = {}
            OUT_JSON_DIR.mkdir(exist_ok=True)
            progress = st.progress(0)
            status = st.empty()

            for i, mat in enumerate(mats, 1):
                status.write(f"Running {i}/{len(mats)}: `{mat}`")
                try:
                    res = hec.run_app_finder(
                        material=mat,
                        context="",
                        use_gpt_queries=bool(use_gpt_queries),
                        max_queries=int(max_queries),
                        papers_per_query=int(papers_per_query),
                        top_k=int(top_k),
                        cache_path=str(PROJECT_DIR / os.environ.get("CACHE_DB", "appfinder_cache.sqlite3")),
                    )
                except Exception as e:
                    res = {"input": mat, "error": str(e)}

                canonical = (res.get("resolved", {}) or {}).get("canonical") or mat
                st.session_state.results[canonical] = res
                write_json_utf8(OUT_JSON_DIR / f"{sanitize_filename(canonical)}.json", res)

                progress.progress(i / len(mats))
                time.sleep(0.03)

            status.success("Done. JSON saved to outputs_json/")
            st.info("Next: go to **Tab 3** to discover dynamic tags.")

# -------------------------
# Tab 3
# -------------------------
with tabs[2]:
    st.subheader("3) Discover dynamic application tags")

    if not st.session_state.results:
        st.warning("Run the pipeline in Tab 2 first.")
    else:
        all_papers = []
        seen = set()
        for _, res in st.session_state.results.items():
            if "error" in res:
                continue
            for p in collect_material_papers(res):
                key = (p.get("doi") or p.get("title") or "").strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                all_papers.append(p)

        st.write(f"Corpus size (unique papers for tag discovery): **{len(all_papers)}**")

        colA, colB = st.columns([1, 1])
        with colA:
            if st.button("🧠 Discover tags from corpus", type="primary"):
                try:
                    tag_json = openai_discover_tags(st.session_state.family, all_papers, max_tags=int(max_dynamic_tags))
                    st.session_state.dynamic_tagset = tag_json
                    write_json_utf8(OUT_CSV_DIR / "dynamic_tags.json", tag_json)
                    st.success("Dynamic tags created.")
                except Exception as e:
                    st.error(str(e))

        with colB:
            if st.session_state.dynamic_tagset:
                st.download_button(
                    "⬇️ Download tagset JSON",
                    data=json.dumps(st.session_state.dynamic_tagset, indent=2, ensure_ascii=False).encode("utf-8"),
                    file_name="dynamic_tags.json",
                    mime="application/json",
                )

        if st.session_state.dynamic_tagset:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write(f"**Tag set:** {st.session_state.dynamic_tagset.get('tag_set_name','')}")
            st.write(st.session_state.dynamic_tagset.get("rationale", ""))
            st.markdown("</div>", unsafe_allow_html=True)

            st.write("### Discovered tags")
            for t in st.session_state.dynamic_tagset.get("tags", []):
                st.markdown(f"**{t['tag']}** — {t.get('description','')}")
                st.caption("Keywords: " + ", ".join(t.get("keywords", [])[:12]))

            st.divider()

            if st.button("📌 Score all materials using these dynamic tags"):
                tags = st.session_state.dynamic_tagset.get("tags", [])
                dyn_scores = {}

                progress = st.progress(0)
                status = st.empty()

                items = [(k, v) for k, v in st.session_state.results.items() if "error" not in v]
                for i, (canonical, res) in enumerate(items, 1):
                    status.write(f"Scoring {i}/{len(items)}: `{canonical}`")
                    papers = collect_material_papers(res)
                    dyn_scores[canonical] = score_material_dynamic(papers, tags)
                    progress.progress(i / len(items))
                    time.sleep(0.02)

                st.session_state.dynamic_scores = dyn_scores

                df_long = build_dynamic_long_table(st.session_state.results, dyn_scores)
                st.session_state.dynamic_long = df_long
                (OUT_CSV_DIR / "applications_long_dynamic.csv").write_text(df_long.to_csv(index=False), encoding="utf-8-sig")

                st.success("Dynamic scoring complete. Go to **Tab 4**.")

# -------------------------
# Tab 4
# -------------------------
with tabs[3]:
    st.subheader("4) Application Finder")

    if st.session_state.dynamic_long is None or st.session_state.dynamic_long.empty:
        st.info("No dynamic results yet. In Tab 3: Discover tags → Score all materials.")
    else:
        df = st.session_state.dynamic_long.copy()
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)

        tags = sorted(df["application_tag"].astype(str).unique().tolist())
        materials = sorted(df["material_canonical"].astype(str).unique().tolist())

        c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
        with c1:
            view_mode = st.selectbox("View", ["By application", "By material"], index=0)
        with c2:
            min_s = st.slider("Min score", 0.0, 1.0, 0.6, 0.01)
        with c3:
            topN = st.slider("Show top N", 5, 60, 20, 5)
        with c4:
            show_evidence = st.checkbox("Show evidence", value=True)

        st.divider()

        if view_mode == "By application":
            tag = st.selectbox("Application tag", tags, index=0)
            df_t = df[(df["application_tag"] == tag) & (df["score"] >= min_s)].copy()
            df_t.sort_values("score", ascending=False, inplace=True)
            df_t = df_t.head(topN)

            st.markdown(f"### {tag}")
            desc = ""
            if "description" in df_t.columns and not df_t.empty:
                dlist = df_t["description"].dropna().astype(str)
                desc = dlist[dlist != ""].head(1).tolist()[0] if not dlist[dlist != ""].empty else ""
            if desc:
                st.caption(desc)

            for _, row in df_t.iterrows():
                material = row["material_canonical"]
                score = float(row["score"])
                elems = row.get("elements", "")

                st.markdown('<div class="card">', unsafe_allow_html=True)
                top = st.columns([2.8, 1.2])
                with top[0]:
                    st.markdown(f"**{material}**  \n<span class='small'>Elements: {elems}</span>", unsafe_allow_html=True)
                with top[1]:
                    st.metric("Score", f"{score:.3f}")
                st.progress(min(1.0, score))

                if show_evidence:
                    ecols = st.columns(3)
                    for i, col in enumerate(ecols, 1):
                        title = row.get(f"top{i}_title", "")
                        doi = row.get(f"top{i}_doi", "")
                        year = row.get(f"top{i}_year", "")
                        if title:
                            url = doi_link(str(doi))
                            md = f"- [{title}]({url}) ({year})" if url else f"- {title} ({year})"
                            col.markdown(md)

                st.markdown("</div>", unsafe_allow_html=True)
                st.write("")

            st.download_button(
                "⬇️ Download this view (CSV)",
                data=df_t.drop(columns=["description"], errors="ignore").to_csv(index=False).encode("utf-8"),
                file_name=f"ranked_{sanitize_filename(tag)}.csv",
                mime="text/csv",
            )

        else:
            material = st.selectbox("Material", materials, index=0)
            df_m = df[(df["material_canonical"] == material) & (df["score"] >= 0.0)].copy()
            df_m.sort_values("score", ascending=False, inplace=True)

            st.markdown(f"### {material}")
            elems = (df_m["elements"].head(1).tolist() or [""])[0]
            if elems:
                st.caption(f"Elements: {elems}")

            for _, row in df_m.head(topN).iterrows():
                tag = row["application_tag"]
                score = float(row["score"])
                desc = row.get("description", "")

                st.markdown('<div class="card">', unsafe_allow_html=True)
                head = st.columns([2.8, 1.2])
                with head[0]:
                    st.markdown(f"**{tag}**  \n<span class='small'>{desc}</span>", unsafe_allow_html=True)
                with head[1]:
                    st.metric("Score", f"{score:.3f}")
                st.progress(min(1.0, score))

                if show_evidence:
                    ecols = st.columns(3)
                    for i, col in enumerate(ecols, 1):
                        title = row.get(f"top{i}_title", "")
                        doi = row.get(f"top{i}_doi", "")
                        year = row.get(f"top{i}_year", "")
                        if title:
                            url = doi_link(str(doi))
                            md = f"- [{title}]({url}) ({year})" if url else f"- {title} ({year})"
                            col.markdown(md)

                st.markdown("</div>", unsafe_allow_html=True)
                st.write("")

        st.divider()
        st.subheader("Optional: one-number ranking (final score)")

        chosen = st.multiselect("Choose tags to combine (2–4)", tags, default=tags[:2])
        chosen = chosen[:4]

        weights: Dict[str, float] = {}
        if chosen:
            cols = st.columns(len(chosen))
            for i, t in enumerate(chosen):
                with cols[i]:
                    weights[t] = st.slider(f"Weight: {t}", 0.0, 1.0, 0.25, 0.05)

            if st.button("Compute final ranking"):
                df_final = compute_final_score_from_dynamic(df, weights)
                st.session_state.dynamic_final = df_final
                (OUT_CSV_DIR / "materials_final_rank_dynamic.csv").write_text(df_final.to_csv(index=False), encoding="utf-8-sig")

        if st.session_state.dynamic_final is not None and not st.session_state.dynamic_final.empty:
            showN = st.slider("Show top N materials (final score)", 5, 50, 20, 5)
            st.dataframe(
                st.session_state.dynamic_final[["material_canonical", "final_score", "elements"]].head(showN),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "⬇️ Download final ranking CSV",
                data=st.session_state.dynamic_final.to_csv(index=False).encode("utf-8"),
                file_name="materials_final_rank_dynamic.csv",
                mime="text/csv",
            )

        st.divider()
        st.subheader("Downloads")
        st.download_button(
            "Download dynamic long table CSV (material × tag)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="applications_long_dynamic.csv",
            mime="text/csv",
        )
        if st.session_state.dynamic_tagset:
            st.download_button(
                "Download dynamic tagset JSON",
                data=json.dumps(st.session_state.dynamic_tagset, indent=2, ensure_ascii=False).encode("utf-8"),
                file_name="dynamic_tags.json",
                mime="application/json",
            )
