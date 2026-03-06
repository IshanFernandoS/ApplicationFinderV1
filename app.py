import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import io
import csv
import tempfile
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components


# Optional similarity recommender (requires pymatgen+matminer+scikit-learn)
try:
    import similarity
except Exception:
    similarity = None
from dotenv import load_dotenv

load_dotenv()

# =========================
# Page config + style
# =========================
st.set_page_config(page_title="Universal Material Application Finder", page_icon="🧪", layout="wide")
st.markdown("<div style=\"height: 0.75rem\"></div>", unsafe_allow_html=True)
st.title("🧪 Application Finder V1")
st.caption("Goal: turn a material (or shortlist) into evidence-backed application directions (papers + databases).")

PROJECT_DIR = Path(__file__).resolve().parent
OUT_CSV_DIR = PROJECT_DIR / "outputs_csv"
OUT_CSV_DIR.mkdir(exist_ok=True)

SIM_LIB_FILE = OUT_CSV_DIR / "sim_library.pkl"

st.markdown(
    """
<style>
/* -------------------------
   General UI
------------------------- */

/* Button look (global, for non-periodic-table buttons) */
.stButton>button {
  border-radius: 10px;
  padding: 0.35rem 0.45rem;
  min-height: 2.1rem;
  width: 100%;
}

/* Card UI */
.card {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 18px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.03);
}
.small { opacity: 0.8; font-size: 0.9rem; }
.block-container { padding-top: 2.4rem; padding-bottom: 2.0rem; }

/* -------------------------
   Periodic table (scoped)
   Make it smaller + nicer without affecting the rest of the app
------------------------- */
.pt-wrap div[data-testid="stHorizontalBlock"] { gap: 0.06rem !important; }
.pt-wrap div[data-testid="column"] { padding: 0 !important; }

/* Periodic-table buttons only (override global button sizing) */
.pt-wrap .stButton>button {
  border-radius: 9px;
  padding: 0.12rem 0.18rem;
  min-height: 1.55rem;
  width: 100%;
  font-size: 0.78rem;
  line-height: 1.05;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.03);
  transition: transform 90ms ease, background 120ms ease, border-color 120ms ease;
}
.pt-wrap .stButton>button:hover {
  transform: translateY(-1px);
  border-color: rgba(255,255,255,0.22);
  background: rgba(255,255,255,0.06);
}
.pt-wrap .stButton>button:active {
  transform: translateY(0px) scale(0.98);
}

/* Optional: if group numbers are shown inside PT, make them tiny */
.pt-wrap div[data-testid="stCaptionContainer"] p {
  font-size: 0.60rem;
  opacity: 0.65;
  margin: 0 0 0.15rem 0;
  text-align: center;
}

/* Legend chips (optional) */
.pt-legend {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  font-size: 0.82rem;
  opacity: 0.9;
  margin: 0.25rem 0 0.5rem 0;
}
.pt-legend span {
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  padding: 4px 8px;
  border-radius: 999px;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# Import hec_app_finder helpers (cache + OpenAlex client)
# =========================
try:
    import hec_app_finder as hec  # expects hec_app_finder.py in same folder
except Exception as e:
    st.error(
        "Could not import hec_app_finder.py. Put this file in the SAME folder as hec_app_finder.py.\n\n"
        f"Import error: {e}"
    )
    st.stop()

OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY") or "").strip()
OPENAI_MODEL = (os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip()
OPENALEX_API_KEY = (os.environ.get("OPENALEX_API_KEY") or "").strip()
CACHE_DB = (os.environ.get("CACHE_DB") or "appfinder_cache.sqlite3").strip()

# =========================
# OPTIMADE (optional dependency)
# =========================
OPTIMADE_CLIENT_OK = True
OPTIMADE_CLIENT_IMPORT_ERR = ""
try:
    from optimade.client import OptimadeClient
except Exception as e:
    OPTIMADE_CLIENT_OK = False
    OPTIMADE_CLIENT_IMPORT_ERR = str(e)

GET_PROVIDERS_OK = True
GET_PROVIDERS_IMPORT_ERR = ""
try:
    # get_providers is convenient, but its import path has changed across versions.
    from optimade.utils import get_providers  # type: ignore
except Exception as e:
    GET_PROVIDERS_OK = False
    GET_PROVIDERS_IMPORT_ERR = str(e)

# =========================
# OPTIMADE provider fallback list
# (Used if the live provider registry cannot be loaded from providers.optimade.org)
# =========================
FALLBACK_PROVIDER_LABELS = [
    "mp — Materials Project",
    "oqmd — Open Quantum Materials Database",
    "cod — Crystallography Open Database",
    "nmd — NOMAD",
    "jarvis — JARVIS",
    "mcloud — Materials Cloud",
]

# =========================
# Periodic table data + layout
# =========================
ELEMENTS = [
    "H","He",
    "Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
    "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
    "Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og",
]

PT_LAYOUT = [
    [("H", 1), ("He", 18)],
    [("Li", 1), ("Be", 2), ("B", 13), ("C", 14), ("N", 15), ("O", 16), ("F", 17), ("Ne", 18)],
    [("Na", 1), ("Mg", 2), ("Al", 13), ("Si", 14), ("P", 15), ("S", 16), ("Cl", 17), ("Ar", 18)],
    [("K", 1), ("Ca", 2), ("Sc", 3), ("Ti", 4), ("V", 5), ("Cr", 6), ("Mn", 7), ("Fe", 8), ("Co", 9), ("Ni", 10),
     ("Cu", 11), ("Zn", 12), ("Ga", 13), ("Ge", 14), ("As", 15), ("Se", 16), ("Br", 17), ("Kr", 18)],
    [("Rb", 1), ("Sr", 2), ("Y", 3), ("Zr", 4), ("Nb", 5), ("Mo", 6), ("Tc", 7), ("Ru", 8), ("Rh", 9), ("Pd", 10),
     ("Ag", 11), ("Cd", 12), ("In", 13), ("Sn", 14), ("Sb", 15), ("Te", 16), ("I", 17), ("Xe", 18)],
    [("Cs", 1), ("Ba", 2), ("La", 3), ("Hf", 4), ("Ta", 5), ("W", 6), ("Re", 7), ("Os", 8), ("Ir", 9), ("Pt", 10),
     ("Au", 11), ("Hg", 12), ("Tl", 13), ("Pb", 14), ("Bi", 15), ("Po", 16), ("At", 17), ("Rn", 18)],
    [("Fr", 1), ("Ra", 2), ("Ac", 3), ("Rf", 4), ("Db", 5), ("Sg", 6), ("Bh", 7), ("Hs", 8), ("Mt", 9), ("Ds", 10),
     ("Rg", 11), ("Cn", 12), ("Nh", 13), ("Fl", 14), ("Mc", 15), ("Lv", 16), ("Ts", 17), ("Og", 18)],
]
LANTHANOIDS = ["Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu"]
ACTINOIDS   = ["Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr"]

# Category sets for lightweight "color" indicators
ALKALI = {"Li","Na","K","Rb","Cs","Fr"}
ALKALINE_EARTH = {"Be","Mg","Ca","Sr","Ba","Ra"}
HALOGENS = {"F","Cl","Br","I","At","Ts"}
NOBLE_GASES = {"He","Ne","Ar","Kr","Xe","Rn","Og"}
METALLOIDS = {"B","Si","Ge","As","Sb","Te"}
NONMETALS = {"H","C","N","O","P","S","Se"}
POST_TRANSITION = {"Al","Ga","In","Sn","Tl","Pb","Bi","Po","Nh","Fl","Mc","Lv"}
LANTH_SET = {"La", *LANTHANOIDS, "Lu"}
ACT_SET = {"Ac", *ACTINOIDS, "Lr"}
TRANSITION = {
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn"
}

def category(sym: str) -> str:
    if sym in ALKALI:
        return "alkali"
    if sym in ALKALINE_EARTH:
        return "alkaline"
    if sym in LANTH_SET:
        return "lanth"
    if sym in ACT_SET:
        return "act"
    if sym in HALOGENS:
        return "halogen"
    if sym in NOBLE_GASES:
        return "noble"
    if sym in METALLOIDS:
        return "metalloid"
    if sym in NONMETALS:
        return "nonmetal"
    if sym in POST_TRANSITION:
        return "post"
    if sym in TRANSITION:
        return "transition"
    return "other"

# Smaller, cleaner indicators (saves width + looks nicer)
CAT_ICON = {
    "alkali": "🔴",
    "alkaline": "🟠",
    "transition": "🔵",
    "post": "🟣",
    "metalloid": "🟢",
    "nonmetal": "🟡",
    "halogen": "⚪",
    "noble": "⚫",
    "lanth": "🟢",
    "act": "🟣",
    "other": "·",
}

def toggle_elem(sym: str) -> None:
    sel = st.session_state.allowed_elements
    if sym in sel:
        sel.remove(sym)
    else:
        sel.append(sym)
    sel.sort()

def periodic_table_widget(show_group_numbers: bool = False) -> None:
    if show_group_numbers:
        header = st.columns(18)
        for g in range(18):
            header[g].caption(str(g + 1))

    for period in PT_LAYOUT:
        cols = st.columns(18)
        lookup = {g: sym for sym, g in period}
        for g in range(1, 19):
            sym = lookup.get(g, "")
            if not sym:
                cols[g - 1].write("")
                continue
            selected = sym in st.session_state.allowed_elements
            icon = CAT_ICON.get(category(sym), "·")
            label = f"✓ {sym}" if selected else f"{icon} {sym}"
            if cols[g - 1].button(label, key=f"pt_{sym}"):
                toggle_elem(sym)
                st.rerun()

    st.markdown("**Lanthanides**")
    cols = st.columns(14)
    for i, sym in enumerate(LANTHANOIDS):
        selected = sym in st.session_state.allowed_elements
        label = f"✓ {sym}" if selected else f"{CAT_ICON['lanth']} {sym}"
        if cols[i].button(label, key=f"pt_{sym}"):
            toggle_elem(sym)
            st.rerun()

    st.markdown("**Actinides**")
    cols = st.columns(14)
    for i, sym in enumerate(ACTINOIDS):
        selected = sym in st.session_state.allowed_elements
        label = f"✓ {sym}" if selected else f"{CAT_ICON['act']} {sym}"
        if cols[i].button(label, key=f"pt_{sym}"):
            toggle_elem(sym)
            st.rerun()

# =========================
# Helper utilities
# =========================
FORMULA_RE = re.compile(r"^[A-Z][A-Za-z0-9().:+\-]*$")  # permissive
HEC_RE = re.compile(r"^\([A-Z][a-z]?\d*\.?\d*(?:[A-Z][a-z]?\d*\.?\d*)+\)[A-Z][a-z]?$")

# Whole-word matching only for short acronyms (prevents false positives: "ris" in "arising", "rf" in "performance")
BOUNDARY_KEYWORDS = {
    "rf","ir","uv","ris","fss","thz","nir","mir","ebg","amc","ghz"
}

def _norm_text(s: str) -> str:
    return (
        (s or "")
        .lower()
        .replace("µ", "u")
        .replace("μ", "u")
        .replace("–", "-")
        .replace("—", "-")
    )

def _kw_match(text_lc: str, kw: str) -> bool:
    k = _norm_text(kw).strip()
    if not k:
        return False
    # Only boundary-match known acronyms; otherwise allow substring match
    if k in BOUNDARY_KEYWORDS:
        return re.search(rf"\b{re.escape(k)}\b", text_lc) is not None
    return k in text_lc

def doi_link(doi: str) -> str:
    if not isinstance(doi, str) or not doi.strip():
        return ""
    d = doi.strip()
    if d.startswith("http://") or d.startswith("https://"):
        return d
    return f"https://doi.org/{d}"

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
    t = _norm_text(text)
    hits: List[str] = []
    for kw in keywords:
        k = (kw or "").strip()
        if not k:
            continue
        if _kw_match(t, k):
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

def naive_elements_from_string(s: str) -> List[str]:
    elems = re.findall(r"[A-Z][a-z]?", s)
    uniq = []
    for e in elems:
        if e not in uniq and e in ELEMENTS:
            uniq.append(e)
    return uniq

def fmt_optimade_elements_has_all(elems: List[str]) -> str:
    # OPTIMADE spec example uses comma-separated quoted symbols:
    # elements HAS ALL "Si", "Al", "O"
    return "elements HAS ALL " + ", ".join([f"\"{e}\"" for e in elems])

def build_optimade_filter(material: str) -> Optional[str]:
    elems = naive_elements_from_string(material)
    if not elems:
        return None
    elems = sorted(set(elems))
    n = len(elems)

    # If it looks like a plain reduced formula, try exact match too
    formula_like = bool(FORMULA_RE.match(material)) and not material.strip().startswith("(")
    exact = f'chemical_formula_reduced="{material}"' if formula_like else ""

    base = f"({fmt_optimade_elements_has_all(elems)} AND nelements={n})"
    if exact:
        return f"({exact} OR {base})"
    return base

def optimade_structure_url(base_url: str, sid: str) -> str:
    return f"{base_url.rstrip('/')}/v1/structures/{sid}"

# =========================
# OpenAI Structured Outputs (JSON schema)
# =========================
CAND_SCHEMA = {
    "type": "object",
    "properties": {
        "family_interpretation": {"type": "string"},
        "candidates": {
            "type": "array",
            "minItems": 1,
            "maxItems": 200,
            "items": {
                "type": "object",
                "properties": {"material": {"type": "string"}, "notes": {"type": "string"}},
                "required": ["material", "notes"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["family_interpretation", "candidates"],
    "additionalProperties": False,
}

QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "material": {"type": "string"},
        "aliases": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
        "queries": {"type": "array", "items": {"type": "string"}, "maxItems": 30},
        "must_have_terms": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
    },
    "required": ["material", "aliases", "queries", "must_have_terms"],
    "additionalProperties": False,
}

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

# =========================
# Metamaterials/EM devices (GPT JSON schema + controlled vocab)
# =========================
META_CAND_SCHEMA = {
    "type": "object",
    "properties": {
        "goal_interpretation": {"type": "string"},
        "concepts": {
            "type": "array",
            "minItems": 1,
            "maxItems": 200,
            "items": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["concept", "notes"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["goal_interpretation", "concepts"],
    "additionalProperties": False,
}

# Keep short acronyms as whole words in gating (avoid 'ris' matching 'arising')
META_SHORT_ACRONYMS = {x.lower() for x in ["rf","ir","uv","ris","fss","amc","ebg","thz","nir","mir","ghz"]}

META_REQUIRED_TERMS = [
    "metasurface", "metamaterial", "meta-atom", "meta atom",
    "frequency selective surface", "fss",
    "ris", "reconfigurable intelligent surface",
    "reflectarray", "transmitarray",
    "reconfigurable", "tunable",
    "beam steering",
    "absorber", "polarization", "filter", "modulator", "switch",
    "electromagnetic bandgap", "ebg",
    "artificial magnetic conductor", "amc",
]

META_BAND_KEYWORDS = {
    "RF": ["rf", "radio frequency"],
    "Microwave": ["microwave", "ghz", "x-band", "s-band", "c-band", "ku-band", "ka-band"],
    "mmWave": ["mmwave", "mm-wave", "millimeter-wave", "28 ghz", "39 ghz", "60 ghz", "sub-thz", "sub thz"],
    "THz": ["terahertz", "thz"],
    "IR": ["infrared", "mid-infrared", "mid infrared", "mir"],
    "NIR": ["near-infrared", "near infrared", "nir", "telecom", "1.55 um", "1.55 µm", "1550 nm"],
    "Optical": ["optical", "visible", "photonics"],
}

META_DEVICE_TYPES = [
    "Metasurface",
    "Reconfigurable metasurface / RIS",
    "RIS",
    "FSS / filter",
    "Metamaterial absorber",
    "Antenna / resonator tuning",
    "Waveguide component",
    "Lens / holography",
    "Sensor",
]

META_TUNING = [
    "Phase-change (VO2/GST)",   # IMPORTANT: ASCII VO2 (not VO₂)
    "Varactor",
    "PIN diode",
    "MEMS",
    "Liquid crystal",
    "Graphene/ITO",
    "Ferroelectric (BST)",
    "Thermal",
    "Optical pumping",
    "Mechanical strain",
]

def openai_responses_json(schema: Dict[str, Any], user_msg: str, name: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing in .env.")
    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": "Return ONLY schema-valid JSON. Be concise and accurate."},
            {"role": "user", "content": user_msg},
        ],
        "text": {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
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
    return json.loads(out_text)

def generate_candidates_any(family: str, element_pool: List[str], n: int) -> Tuple[str, List[str]]:
    pool_str = ", ".join(element_pool) if element_pool else "(no restriction)"
    user_msg = f"""
Generate a shortlist of candidate materials for the family below.

Family description:
{family}

Constraints:
- Output at least {min(5,n)} and ideally {n} candidates.
- If an element pool is provided, use ONLY these elements: {pool_str}
- Candidates can be formulas (e.g., SrTiO3, HfO2), named materials (e.g., VO2), or doped forms (e.g., VO2:W 2%).
- Avoid duplicates.
- Keep each candidate string short (<= 30 chars).

Return JSON in schema.
""".strip()
    obj = openai_responses_json(CAND_SCHEMA, user_msg, "any_material_candidates")
    interp = obj.get("family_interpretation", "")
    raw = [c.get("material", "").strip() for c in (obj.get("candidates") or [])]
    out, seen = [], set()
    for m in raw:
        if not m:
            continue
        key = m.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
        if len(out) >= n:
            break
    return interp, out

def build_query_plan(material: str, family: str, db_context: str = "") -> Dict[str, Any]:
    user_msg = f"""
Build a literature search plan for the material below.

Material: {material}
Context/family: {family}

Database context (may be empty):
{db_context}

Requirements:
- Provide 8 to 20 OpenAlex-suitable queries (4–16 words each).
- Include a short list of aliases/synonyms if applicable.
- Provide must_have_terms to help filter irrelevant papers.
- If database IDs or known formula variants are provided above, include them in some queries.

Return JSON matching the schema.
""".strip()
    return openai_responses_json(QUERY_SCHEMA, user_msg, "material_query_plan")

def discover_tags_from_corpus(family: str, corpus: List[Dict[str, Any]], max_tags: int) -> Dict[str, Any]:
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
Propose a dynamic application-tag taxonomy based on this literature corpus.

Family description:
{family}

Corpus (titles + snippets):
{corpus_text}

Requirements:
- Return 6 to {max_tags} tags supported by the corpus.
- Each tag must include 6–14 keywords/phrases for matching.
- Tags should be application/property directions (not generic).

Return JSON matching schema.
""".strip()
    obj = openai_responses_json(TAG_SCHEMA, user_msg, "dynamic_tags")
    obj["tags"] = (obj.get("tags") or [])[:max_tags]
    return obj

def _meta_profile_to_goal(meta: Dict[str, Any]) -> str:
    bands = ", ".join(meta.get("bands") or [])
    funcs = ", ".join(meta.get("device_functions") or [])
    arch = ", ".join(meta.get("device_archetypes") or [])
    tuning = ", ".join(meta.get("tuning") or [])
    mats = ", ".join(meta.get("active_materials") or [])
    geom = (meta.get("geometry_keywords") or "").strip()
    hint = (meta.get("band_hint") or "").strip()
    parts = [
        f"Bands: {bands}" + (f" ({hint})" if hint else ""),
        f"Functions: {funcs}",
        f"Device archetypes: {arch}",
        f"Tuning: {tuning}",
        f"Active materials: {mats}",
    ]
    if geom:
        parts.append(f"Geometry keywords: {geom}")
    return "\n".join(parts)

def generate_meta_concepts(goal_text: str, meta: Dict[str, Any], n: int) -> Tuple[str, List[str]]:
    user_msg = f"""
Generate metamaterials/EM device concepts for literature search.

Goal:
{goal_text}

Profile:
{_meta_profile_to_goal(meta)}

Constraints:
- Output {n} concepts (or at least {min(6, n)}).
- Each concept MUST look like a device idea, NOT just a material name.
  Good: "VO2 reconfigurable metasurface RF switch, X-band, electrically biased"
  Bad:  "VO2" or "metasurface" or "high-k dielectric"
- Include device archetype + function + band + tuning (and optionally active material).
- Keep each concept <= 110 characters.
- Avoid duplicates. Mix VO2/GST/graphene + varactor/MEMS where relevant.

Return JSON in schema.
""".strip()
    obj = openai_responses_json(META_CAND_SCHEMA, user_msg, "meta_device_concepts")
    interp = obj.get("goal_interpretation", "")
    out: List[str] = []
    seen = set()
    for c in (obj.get("concepts") or []):
        s = (c.get("concept") or "").strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= int(n):
            break
    return interp, out

def build_meta_query_plan(concept: str, goal_text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    profile = _meta_profile_to_goal(meta)
    user_msg = f"""
Build a literature search plan for a metamaterials/EM device concept.

Device concept:
{concept}

Goal:
{goal_text}

Profile:
{profile}

Requirements:
- Provide 10 to 22 OpenAlex-suitable queries (4–18 words each).
- Queries should explicitly include metamaterials terms (metasurface/metamaterial/FSS/RIS/etc).
- Include band keywords where appropriate (microwave/mmWave/THz/IR/optical).
- Include tuning mechanism keywords (VO2/GST/varactor/MEMS/graphene/etc).
- Provide must_have_terms for filtering irrelevant papers (5–10 terms).
- Provide aliases (0–12) if useful.

Return JSON matching the schema.
""".strip()
    return openai_responses_json(QUERY_SCHEMA, user_msg, "meta_query_plan")

def _extract_tokens_for_gate(s: str) -> List[str]:
    """
    Extract gating tokens:
      - keep 3+ char tokens
      - keep whitelisted short acronyms as whole words
    """
    s_norm = _norm_text(s)
    toks = re.findall(r"[a-z0-9u\-]{3,}", s_norm)
    for a in META_SHORT_ACRONYMS:
        if re.search(rf"\b{re.escape(a)}\b", s_norm):
            toks.append(a)
    return toks

def meta_gate_relevance(works: List[Dict[str, Any]], concept: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    required_terms = [str(t) for t in (META_REQUIRED_TERMS or []) if t]

    concept_terms: List[str] = []
    concept_terms += _extract_tokens_for_gate(concept)

    for k in ("device_archetypes", "tuning", "active_materials", "device_functions"):
        for it in (meta.get(k) or []):
            concept_terms += _extract_tokens_for_gate(str(it))

    # include geometry keywords (important for EM device relevance)
    concept_terms += _extract_tokens_for_gate(str(meta.get("geometry_keywords") or ""))

    # add band keywords
    for b in (meta.get("bands") or []):
        for kw in META_BAND_KEYWORDS.get(b, []):
            concept_terms.append(str(kw))

    # de-duplicate + cap
    concept_terms = list(dict.fromkeys([_norm_text(x) for x in concept_terms if x]))[:90]

    kept: List[Dict[str, Any]] = []
    for w in works:
        txt = _norm_text((w.get("title") or "") + " " + (w.get("abstract_snippet") or ""))

        # must contain at least one metamaterials anchor term
        if not any(_kw_match(txt, t) for t in required_terms):
            continue

        # must contain at least one concept/profile term
        if not any(_kw_match(txt, t) for t in concept_terms):
            continue

        kept.append(w)

    return kept

def discover_tags_from_corpus_meta(goal_text: str, corpus: List[Dict[str, Any]], max_tags: int) -> Dict[str, Any]:
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
You are building an application-tag taxonomy for metamaterials/metasurfaces and EM devices.

Goal:
{goal_text}

Corpus:
{corpus_text}

Requirements:
- Return 6 to {max_tags} tags grounded in this corpus.
- Tags must be EM device/application directions, e.g. RIS/beam steering, RF switch/modulator, tunable absorber,
  filtering/FSS, polarization control, sensing, lens/holography, antenna tuning, etc.
- Each tag must include 6–14 matching keywords/phrases (include band/tuning terms when relevant).
- Avoid generic tags like "materials" or "review".

Return JSON matching schema.
""".strip()
    obj = openai_responses_json(TAG_SCHEMA, user_msg, "meta_dynamic_tags")
    obj["tags"] = (obj.get("tags") or [])[:max_tags]
    return obj

# =========================
# Retrieval + scoring
# =========================
def retrieve_papers_openalex(
    queries: List[str],
    per_query: int,
    cache: Any,
    use_api_key: bool,
) -> List[Dict[str, Any]]:
    """
    Retrieve papers from OpenAlex.
    - If use_api_key=True and OPENALEX_API_KEY is set, uses it.
    - If OpenAlex returns 429 Insufficient budget (paid key with $0 remaining), automatically retries once WITHOUT the key.
    """
    def _fetch(api_key: Optional[str]) -> List[Dict[str, Any]]:
        oa = hec.OpenAlexClient(api_key, cache)
        works: List[Dict[str, Any]] = []
        for q in queries:
            works.extend(oa.search_works(q, per_page=per_query))
        works = hec.dedupe_works(works)
        for w in works:
            if "abstract_snippet" not in w:
                w["abstract_snippet"] = hec.inverted_index_to_text(w.get("abstract_inverted_index"))
        return works

    key_to_use: Optional[str] = (OPENALEX_API_KEY or None) if use_api_key else None
    try:
        return _fetch(key_to_use)
    except RuntimeError as e:
        msg = str(e)
        if key_to_use and ("HTTP 429" in msg) and ("Insufficient budget" in msg or "dailyRemainingUsd" in msg or "creditsRemaining" in msg):
            st.warning(
                "OpenAlex API key has no remaining daily budget (429 Insufficient budget). "
                "Retrying without the key (free mode)."
            )
            try:
                return _fetch(None)
            except Exception as e2:
                st.error(f"OpenAlex also failed without key: {e2}")
                return []
        if "HTTP 429" in msg:
            st.warning("OpenAlex rate-limited this run (HTTP 429). Try fewer queries/papers per query, or run later.")
            return []
        raise


def quick_openalex_presence_check(query: str, cache: Any, per_page: int = 8, use_api_key: bool = False) -> int:
    """Quick heuristic: count OpenAlex works for a simple query (deduped)."""
    try:
        api_key = (OPENALEX_API_KEY or None) if use_api_key else None
        oa = hec.OpenAlexClient(api_key, cache)
        works = oa.search_works(query, per_page=int(per_page))
        works = hec.dedupe_works(works)
        return len(works)
    except Exception:
        return 0

def gate_relevance(works: List[Dict[str, Any]], anchors: List[str]) -> List[Dict[str, Any]]:
    anchors = [a.strip().lower() for a in anchors if a and a.strip()]
    if not anchors:
        return works
    kept = []
    for w in works:
        txt = ((w.get("title") or "") + " " + (w.get("abstract_snippet") or "")).lower()
        if any(a in txt for a in anchors):
            kept.append(w)
    return kept
def quick_openalex_top_works(query: str, cache: Any, per_page: int = 10, use_api_key: bool = False) -> List[Dict[str, Any]]:
    """Fetch a small set of top OpenAlex works for a query (fast; no GPT/tagging)."""
    try:
        api_key = (OPENALEX_API_KEY or None) if use_api_key else None
        oa = hec.OpenAlexClient(api_key, cache)
        works = oa.search_works(query, per_page=int(per_page))
        works = hec.dedupe_works(works)
        for w in works:
            if "abstract_snippet" not in w:
                w["abstract_snippet"] = hec.inverted_index_to_text(w.get("abstract_inverted_index"))
        return works[: int(per_page)]
    except Exception:
        return []





# -------------------------
# CSV + candidate normalization helpers (batch mode)
# -------------------------
def normalize_candidate_string(s: str) -> str:
    """Normalize candidate strings for matching/search: ASCII, trim, collapse spaces."""
    s = (s or "").strip()
    if not s:
        return ""
    # common unicode normalization
    s = s.replace("VO₂", "VO2").replace("µ", "u").replace("μ", "u")
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s

def candidate_query_variants(s: str) -> List[str]:
    """Generate multiple OpenAlex query variants to handle different CSV formats."""
    base = normalize_candidate_string(s)
    if not base:
        return []
    variants = [base]

    # Doped notation: Host:Dopant X%
    m = re.match(r"^\s*([A-Za-z0-9().+\-]+)\s*:\s*([A-Z][a-z]?)\s*(\d+(?:\.\d+)?)\s*(?:at\.?%|wt\.?%|%)\s*$", base, flags=re.I)
    if m:
        host, dop, pct = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        variants.append(host)
        variants.append(f"{host} {dop} doped")
        variants.append(f"{host} doped with {dop}")
        variants.append(f"{host} {dop} doping {pct}%")
        variants.append(f"\"{host}\" {dop} doped")

    # remove % / at% / wt% fragments
    no_pct = re.sub(r"\b\d+(?:\.\d+)?\s*(?:at\.?%|wt\.?%|%)\b", "", base, flags=re.I).strip()
    no_pct = re.sub(r"\s+", " ", no_pct).strip()
    if no_pct and no_pct not in variants:
        variants.append(no_pct)

    # replace colon with space
    no_colon = re.sub(r"\s+", " ", base.replace(":", " ")).strip()
    if no_colon and no_colon not in variants:
        variants.append(no_colon)

    # quoted exact string
    variants.append(f"\"{base}\"")

    # de-dupe
    out, seen = [], set()
    for v in variants:
        v2 = v.strip()
        if not v2:
            continue
        k = v2.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(v2)
    return out

def read_csv_robust(uploaded_file) -> pd.DataFrame:
    """
    Robust CSV reader for Streamlit upload:
    - tries common encodings (incl. utf-16)
    - auto-detects delimiter
    - falls back to python engine + on_bad_lines handling
    """
    raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()

    encodings = ("utf-8-sig", "utf-8", "utf-16", "cp1252", "latin1")
    for enc in encodings:
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("latin1", errors="replace")

    sample = text[:5000]
    delim_candidates = [",", ";", "\t", "|"]

    delimiter = ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=delim_candidates)
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ","

    try:
        return pd.read_csv(io.StringIO(text), sep=delimiter)
    except Exception:
        pass

    try:
        return pd.read_csv(io.StringIO(text), sep=delimiter, engine="python", on_bad_lines="skip")
    except TypeError:
        return pd.read_csv(io.StringIO(text), sep=delimiter, engine="python", error_bad_lines=False, warn_bad_lines=True)

def show_knn_graph_from_neighbors(
    center: str,
    neighbors: List[Dict[str, Any]],
    title: str = "Nearest neighbors + top applications",
    top_tags: Optional[List[Tuple[str, float]]] = None,
) -> None:
    """Small kNN graph: center + neighbors (+ optional top tags)."""
    G = nx.Graph()
    G.add_node(center, role="center")

    for n in neighbors:
        mname = str(n.get("material", "")).strip()
        if not mname:
            continue
        sim = float(n.get("sim", 0.0) or 0.0)
        G.add_node(mname, role="neighbor")
        G.add_edge(center, mname, weight=sim, kind="neighbor")

    if top_tags:
        for tag, sc in top_tags[:5]:
            tname = str(tag).strip()
            if not tname:
                continue
            score = float(sc or 0.0)
            node_id = f"TAG::{tname}"
            G.add_node(node_id, role="tag", label=tname, tag_score=score)
            G.add_edge(center, node_id, weight=score, kind="tag")

    net = Network(height="460px", width="100%", bgcolor="#0E1117", font_color="white")
    net.barnes_hut(gravity=-2000, central_gravity=0.2, spring_length=140, spring_strength=0.02)

    for node, data in G.nodes(data=True):
        role = data.get("role")
        if role == "center":
            net.add_node(node, label=node, color="#E45756", size=28, title=f"<b>{node}</b><br>(candidate)")
        elif role == "tag":
            label = data.get("label", node.replace("TAG::", ""))
            sc = float(data.get("tag_score", 0.0))
            net.add_node(node, label=label, color="#54A24B", size=16, title=f"<b>{label}</b><br>score={sc:.3f}")
        else:
            net.add_node(node, label=node, color="#4C78A8", size=18, title=f"<b>{node}</b><br>(neighbor)")

    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 0.0))
        kind = data.get("kind", "neighbor")
        if kind == "tag":
            net.add_edge(u, v, value=max(1, int(8 * w)), dashes=True, title=f"tag_score={w:.3f}")
        else:
            net.add_edge(u, v, value=max(1, int(10 * w)), title=f"cosine={w:.3f}")

    net.set_options("""
    var options = {
      "nodes": {"shape": "dot", "font": {"size": 13}},
      "edges": {"smooth": false, "color": {"opacity": 0.55}},
      "physics": {"stabilization": {"iterations": 220}}
    }
    """)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        html = open(tmp.name, "r", encoding="utf-8").read()

    st.markdown(f"### {title}")
    components.html(html, height=480, scrolling=False)


def format_material_name_generic(name: Any) -> str:
    s = "" if name is None else str(name)
    s = s.strip().replace("·", ".")
    s = re.sub(r"\s+", "", s)
    s = s.replace("−", "-")
    return s


def _normalize_neighbor_labels(neighbors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for n in neighbors or []:
        row = dict(n)
        row["material"] = format_material_name_generic(row.get("material", ""))
        out.append(row)
    return out


def _neighbor_indices_from_list(lib: Any, neighbors: List[Dict[str, Any]]) -> List[int]:
    mat_to_idx = {str(m): i for i, m in enumerate(getattr(lib, "materials", []) or [])}
    out: List[int] = []
    for n in neighbors or []:
        mname = str(n.get("material", "")).strip()
        if mname in mat_to_idx:
            out.append(int(mat_to_idx[mname]))
    return out


def build_application_contributions(
    lib: Any,
    neighbors: List[Dict[str, Any]],
    alpha: float = 2.0,
    top_apps: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idxs = _neighbor_indices_from_list(lib, neighbors)
    if not idxs:
        return pd.DataFrame(), pd.DataFrame()

    tag_names = list(getattr(lib, "tag_names", []) or [])
    tag_matrix = getattr(lib, "tag_matrix", None)
    if tag_matrix is None or len(tag_names) == 0:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    for n, idx in zip(neighbors, idxs):
        mname = str(n.get("material", "")).strip()
        sim = float(n.get("sim", 0.0) or 0.0)
        weight = max(sim, 0.0) ** float(alpha)
        tag_vec = tag_matrix[int(idx)]
        for j, tag in enumerate(tag_names):
            raw = float(tag_vec[j]) if j < len(tag_vec) else 0.0
            contrib = weight * max(raw, 0.0)
            if contrib <= 0:
                continue
            rows.append({
                "material": mname,
                "sim": sim,
                "application_tag": str(tag),
                "neighbor_tag_score": raw,
                "contribution": contrib,
            })

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    df_edges = pd.DataFrame(rows)
    app_scores = (
        df_edges.groupby("application_tag", as_index=False)["contribution"]
        .sum()
        .sort_values("contribution", ascending=False)
    )
    keep = app_scores.head(int(top_apps))["application_tag"].astype(str).tolist()
    df_edges = df_edges[df_edges["application_tag"].astype(str).isin(keep)].copy()
    app_scores = app_scores[app_scores["application_tag"].astype(str).isin(keep)].copy()
    if not app_scores.empty:
        denom = float(app_scores["contribution"].max()) or 1.0
        app_scores["score"] = app_scores["contribution"] / denom
    return df_edges, app_scores




def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = str(hex_color).strip().lstrip('#')
    if len(hex_color) != 6:
        return (127, 127, 127)
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def _blend_rgb(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))


def _rgba_str(rgb: Tuple[int, int, int], alpha: float) -> str:
    r, g, b = rgb
    alpha = max(0.0, min(1.0, float(alpha)))
    return f"rgba({r}, {g}, {b}, {alpha:.3f})"


def _similarity_edge_style(similarity: float, base_width: float = 6.0, width_gain: float = 10.0) -> Dict[str, Any]:
    sim = max(0.0, min(1.0, float(similarity or 0.0)))
    # Brighter, higher-contrast map for black backgrounds:
    # low similarity -> vivid magenta/red, high similarity -> bright cyan
    low_rgb = _hex_to_rgb('#FF4D6D')
    high_rgb = _hex_to_rgb('#00E5FF')
    rgb = _blend_rgb(low_rgb, high_rgb, sim)
    length = max(35, int(700 - 620 * sim))
    width = base_width + width_gain * sim
    opacity = 0.78 + 0.20 * sim
    return {
        'sim': sim,
        'rgb': rgb,
        'color': _rgba_str(rgb, opacity),
        'length': length,
        'width': width,
    }


def _node_palette(role: str, strength: float = 0.6) -> str:
    strength = max(0.0, min(1.0, float(strength or 0.0)))
    if role == 'center':
        a, b = _hex_to_rgb('#FFD166'), _hex_to_rgb('#FF9F1C')
    elif role == 'neighbor':
        a, b = _hex_to_rgb('#4CC9F0'), _hex_to_rgb('#00E5FF')
    else:
        a, b = _hex_to_rgb('#C77DFF'), _hex_to_rgb('#80FFDB')
    return '#' + ''.join(f'{c:02X}' for c in _blend_rgb(a, b, strength))

def _render_pyvis_graph(net: Network, title: str, height: int = 480) -> None:
    opts = getattr(net, "options", None)
    try:
        if isinstance(opts, dict):
            physics = opts.get("physics")
            if isinstance(physics, bool):
                opts["physics"] = {"enabled": bool(physics)}
        elif hasattr(opts, "physics") and isinstance(getattr(opts, "physics"), bool):
            opts.physics = {"enabled": bool(opts.physics)}
    except Exception:
        pass

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        html = open(tmp.name, "r", encoding="utf-8").read()
    st.markdown(f"### {title}")
    components.html(html, height=height, scrolling=False)


def show_material_network_graph(center: str, neighbors: List[Dict[str, Any]], title: str = "kNN Composition Network") -> None:
    center = format_material_name_generic(center)
    neighbors = _normalize_neighbor_labels(neighbors)

    G = nx.Graph()
    G.add_node(center, role="center")

    top_neighbors = [n for n in neighbors if str(n.get("material", "")).strip()][:8]
    for n in top_neighbors:
        mname = str(n.get("material", "")).strip()
        sim = float(n.get("sim", 0.0) or 0.0)
        G.add_node(mname, role="neighbor", sim=sim)
        G.add_edge(center, mname, weight=sim, kind="candidate")

    for i in range(len(top_neighbors)):
        for j in range(i + 1, len(top_neighbors)):
            si = float(top_neighbors[i].get("sim", 0.0) or 0.0)
            sj = float(top_neighbors[j].get("sim", 0.0) or 0.0)
            approx = min(si, sj) * 0.92
            if approx >= 0.45:
                a = str(top_neighbors[i].get("material", "")).strip()
                b = str(top_neighbors[j].get("material", "")).strip()
                if a and b:
                    G.add_edge(a, b, weight=approx, kind="neighbor")

    net = Network(height="700px", width="100%", bgcolor="#0E1117", font_color="white")
    net.barnes_hut(gravity=-5200, central_gravity=0.06, spring_length=270, spring_strength=0.004, damping=0.88)

    for node, data in G.nodes(data=True):
        if data.get("role") == "center":
            net.add_node(node, label=node, color=_node_palette("center", 0.95), size=38, title=f"<b>{node}</b><br>(candidate)")
        else:
            sim = float(data.get("sim", 0.0) or 0.0)
            size = 20 + 18 * max(0.0, min(1.0, sim))
            net.add_node(node, label=node, color=_node_palette("neighbor", sim), size=size, title=f"<b>{node}</b><br>similarity={sim:.3f}")

    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 0.0) or 0.0)
        edge_style = _similarity_edge_style(w, base_width=7.0, width_gain=11.0)
        net.add_edge(
            u,
            v,
            width=edge_style["width"],
            length=edge_style["length"],
            color=edge_style["color"],
            title=f"similarity={w:.3f}<br>edge length≈{edge_style['length']}px",
        )

    net.set_options("""
    var options = {
      "nodes": {"shape": "dot", "font": {"size": 18, "strokeWidth": 0}},
      "edges": {"smooth": false, "arrows": false, "scaling": {"min": 1, "max": 10}},
      "interaction": {
        "hover": true,
        "tooltipDelay": 120,
        "navigationButtons": true,
        "keyboard": true,
        "dragNodes": true,
        "dragView": true,
        "zoomView": true,
        "hideEdgesOnDrag": false
      },
      "physics": {"enabled": true, "solver": "barnesHut", "stabilization": {"enabled": true, "iterations": 350}}
    }
    """)
    _render_pyvis_graph(net, title=title, height=740)


def show_application_evidence_graph(
    center: str,
    neighbors: List[Dict[str, Any]],
    df_edges: pd.DataFrame,
    app_scores: pd.DataFrame,
    title: str = "Composition → Application Evidence Network",
) -> None:
    center = format_material_name_generic(center)
    neighbors = _normalize_neighbor_labels(neighbors)
    net = Network(height="640px", width="100%", bgcolor="#0E1117", font_color="white", directed=False)
    net.set_options("""
    var options = {
      "layout": {"hierarchical": {"enabled": true, "direction": "LR", "sortMethod": "directed", "nodeSpacing": 220, "levelSeparation": 260}},
      "nodes": {"shape": "dot", "font": {"size": 16, "strokeWidth": 0}},
      "edges": {
        "smooth": {"enabled": true, "type": "cubicBezier", "roundness": 0.3},
        "arrows": false,
        "scaling": {"min": 1, "max": 14}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 120,
        "navigationButtons": true,
        "keyboard": true,
        "dragNodes": true,
        "dragView": true,
        "zoomView": true,
        "hideEdgesOnDrag": false
      },
      "physics": {"enabled": false}
    }
    """)

    net.add_node(center, label=center, color=_node_palette("center", 0.95), size=36, level=0, title=f"<b>{center}</b><br>(candidate)")

    used_neighbors = []
    sim_map: Dict[str, float] = {}
    for n in neighbors[:8]:
        mname = str(n.get("material", "")).strip()
        sim = float(n.get("sim", 0.0) or 0.0)
        if not mname:
            continue
        used_neighbors.append(mname)
        sim_map[mname] = sim
        sim_strength = max(0.0, min(1.0, sim))
        edge_len = max(50, int(500 - 380 * sim_strength))
        edge_width = max(4.5, 4.5 + 9.5 * sim_strength)
        edge_style = _similarity_edge_style(sim, base_width=7.0, width_gain=11.5)
        net.add_node(mname, label=mname, color=_node_palette("neighbor", sim), size=18 + 17 * sim, level=1, title=f"<b>{mname}</b><br>similarity={sim:.3f}")
        net.add_edge(center, mname, width=edge_style["width"], length=edge_style["length"], color=edge_style["color"], title=f"similarity={sim:.3f}<br>edge length≈{edge_style['length']}px")

    score_map: Dict[str, float] = {}
    if app_scores is not None and not app_scores.empty:
        for _, r in app_scores.iterrows():
            score_map[str(r.get("application_tag", ""))] = float(r.get("score", 0.0) or 0.0)

    added_app_nodes = set()
    if df_edges is not None and not df_edges.empty:
        tmp = df_edges.copy()
        tmp["material"] = tmp["material"].map(format_material_name_generic)
        tmp = tmp[tmp["material"].isin(used_neighbors)].copy()
        if not tmp.empty:
            max_contrib = float(tmp["contribution"].max() or 0.0)
            if max_contrib <= 0:
                max_contrib = 1.0
            for _, r in tmp.iterrows():
                mname = str(r.get("material", "")).strip()
                tag = str(r.get("application_tag", "")).strip()
                contrib = float(r.get("contribution", 0.0) or 0.0)
                base = float(score_map.get(tag, 0.0) or 0.0)
                sim = float(sim_map.get(mname, 0.0) or 0.0)
                if not tag:
                    continue
                if tag not in added_app_nodes:
                    net.add_node(tag, label=tag, color=_node_palette("tag", base), size=20 + 18 * base, level=2, title=f"<b>{tag}</b><br>application score={base:.3f}")
                    added_app_nodes.add(tag)
                strength = max(0.0, min(1.0, contrib / max_contrib))
                edge_style = _similarity_edge_style(sim, base_width=7.5 + 2.0 * strength, width_gain=10.0)
                net.add_edge(
                    mname,
                    tag,
                    width=edge_style["width"],
                    length=edge_style["length"],
                    color=edge_style["color"],
                    title=(
                        f"similarity={sim:.3f}<br>"
                        f"contribution={contrib:.3f}<br>"
                        f"relative strength={strength:.3f}<br>"
                        f"edge length≈{edge_style['length']}px"
                    ),
                )

    _render_pyvis_graph(net, title=title, height=680)


def score_material(works: List[Dict[str, Any]], tags: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    now_year = time.gmtime().tm_year
    out = []
    for tag in tags:
        tag_name = tag["tag"]
        keywords = tag.get("keywords") or []
        ranked = []
        for w in works:
            txt = paper_text(w)
            hit_s, hits = compute_hit_score(txt, keywords)
            if hit_s <= 0:
                continue
            boost = recency_boost(w.get("year"), now_year)
            cite = as_float(w.get("cited_by_count"), 0.0)
            cite_boost = min(0.15, (cite / 200.0) * 0.15)
            score = min(1.0, 0.7 * hit_s + boost + cite_boost)
            ranked.append({
                "title": w.get("title"),
                "year": w.get("year"),
                "venue": w.get("venue"),
                "doi": w.get("doi"),
                "match_score": round(score, 3),
                "keyword_hits": hits,
                "abstract_snippet": (w.get("abstract_snippet") or "")[:300],
            })
        ranked.sort(key=lambda x: x["match_score"], reverse=True)
        tag_score = aggregate_tag_score([p["match_score"] for p in ranked])
        out.append({
            "tag": tag_name,
            "description": tag.get("description", ""),
            "score": round(tag_score, 3),
            "top_papers": ranked[:6],
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out

def build_long_table(material_to_scores: Dict[str, List[Dict[str, Any]]], material_meta: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for m, scores in material_to_scores.items():
        meta = material_meta.get(m, {})
        for t in scores:
            top = t.get("top_papers", []) or []
            def pick(i: int, field: str) -> str:
                if i < len(top):
                    v = top[i].get(field, "")
                    return "" if v is None else str(v)
                return ""
            rows.append({
                "material": m,
                "material_type": meta.get("material_type", ""),
                "elements": " ".join(meta.get("elements", [])),
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
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    return df

# =========================
# OPTIMADE helpers
# =========================
@st.cache_data(show_spinner=False)
def optimade_provider_options() -> List[Dict[str, str]]:
    """
    Return list of providers (id + name) from the Materials-Consortia providers list.

    Preference order:
    1) optimade.utils.get_providers (if available in this installed version)
    2) Direct HTTP to https://providers.optimade.org/v1/links
    """
    out: List[Dict[str, str]] = []

    if GET_PROVIDERS_OK:
        try:
            for item in get_providers():  # raw JSON list
                pid = str(item.get("id") or "").strip()
                attrs = item.get("attributes") or {}
                name = str(attrs.get("name") or pid)
                base_url = attrs.get("base_url")
                if not pid or base_url is None:
                    continue
                out.append({"id": pid, "name": name})
            out.sort(key=lambda x: x["id"])
            return out
        except Exception:
            pass

    try:
        r = requests.get("https://providers.optimade.org/v1/links", timeout=30)
        r.raise_for_status()
        data = (r.json() or {}).get("data", [])
        for item in data:
            pid = str(item.get("id") or "").strip()
            attrs = item.get("attributes") or {}
            name = str(attrs.get("name") or pid)
            base_url = attrs.get("base_url")
            if not pid or base_url is None:
                continue
            out.append({"id": pid, "name": name})
        out.sort(key=lambda x: x["id"])
        return out
    except Exception:
        return []

def optimade_search_structures_for_materials(
    materials: List[str],
    provider_ids: List[str],
    max_results_per_provider: int,
) -> Dict[str, Any]:
    """Run a lightweight OPTIMADE 'structures' search for each material, across selected providers."""
    if not OPTIMADE_CLIENT_OK:
        raise RuntimeError(
            "OPTIMADE client not available. Install with: pip install \"optimade[http_client]\"\n"
            f"Import error: {OPTIMADE_CLIENT_IMPORT_ERR}"
        )

    provider_ids = [p.strip() for p in provider_ids if p and p.strip()]
    if not provider_ids:
        return {"enabled": True, "providers": [], "by_material": {}}

    client = OptimadeClient(
        include_providers=set(provider_ids),
        max_results_per_provider=int(max_results_per_provider),
        use_async=False,        # more stable inside Streamlit
        http_timeout=20,
        max_attempts=2,
    )

    response_fields = [
        "chemical_formula_reduced",
        "chemical_formula_descriptive",
        "elements",
        "nelements",
        "nsites",
        "structure_features",
        "space_group_symbol_hall",
        "space_group_it_number",
    ]

    by_material: Dict[str, Any] = {}
    for m in materials:
        flt = build_optimade_filter(m)
        if not flt:
            by_material[m] = {"filter": None, "results": {}, "note": "No elements parsed from input."}
            continue

        try:
            nested = client.get(filter=flt, endpoint="structures", response_fields=response_fields)
            by_base = nested.get("structures", {}).get(flt, {})
        except Exception as e:
            by_material[m] = {"filter": flt, "results": {}, "error": str(e)}
            continue

        results_summary: Dict[str, Any] = {}
        for base_url, payload in (by_base or {}).items():
            data = payload.get("data", []) if isinstance(payload, dict) else []
            if not isinstance(data, list):
                data = []
            top = []
            for entry in data[:5]:
                attrs = (entry.get("attributes") or {}) if isinstance(entry, dict) else {}
                top.append(
                    {
                        "id": entry.get("id", ""),
                        "formula": attrs.get("chemical_formula_reduced") or attrs.get("chemical_formula_descriptive") or "",
                        "nelements": attrs.get("nelements"),
                        "nsites": attrs.get("nsites"),
                        "elements": attrs.get("elements", []),
                        "sg": attrs.get("space_group_it_number") or "",
                        "url": optimade_structure_url(str(base_url), str(entry.get("id", ""))) if entry.get("id") else "",
                    }
                )
            meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
            results_summary[str(base_url)] = {
                "count_returned": len(data),
                "meta_data_returned": meta.get("data_returned", None),
                "top": top,
                "errors": payload.get("errors", []) if isinstance(payload, dict) else [],
            }

        by_material[m] = {"filter": flt, "results": results_summary}

    return {"enabled": True, "providers": provider_ids, "by_material": by_material}

def db_context_for_material(optimade_state: Dict[str, Any], material: str) -> str:
    """Turn OPTIMADE hits into a short string to feed into GPT query planning."""
    if not optimade_state:
        return ""
    mobj = (optimade_state.get("by_material") or {}).get(material) or {}
    if not mobj:
        return ""
    res = mobj.get("results") or {}
    ids = []
    formulas = []
    for base_url, info in res.items():
        for t in (info.get("top") or [])[:3]:
            if t.get("id"):
                ids.append(f"{t['id']} @ {base_url}")
            f = (t.get("formula") or "").strip()
            if f:
                formulas.append(f)
    ids = ids[:6]
    formulas = list(dict.fromkeys(formulas))[:8]
    parts = []
    if formulas:
        parts.append("Known formula variants (from OPTIMADE hits): " + ", ".join(formulas))
    if ids:
        parts.append("Example database structure IDs: " + "; ".join(ids))
    return "\n".join(parts)

# =========================
# Main pipeline
# =========================
def _human_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    sec = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h {minutes:02d}m"

def _progress_weights(n_materials: int) -> Dict[str, float]:
    return {
        "optimade": 0.18,
        "retrieve": 0.52,
        "tagging": 0.15,
        "scoring": 0.15,
    }

def _make_status_box():
    """Compat wrapper: Streamlit 1.30+ has st.status; otherwise fall back."""
    if hasattr(st, "status"):
        return st.status("Starting…", expanded=True)

    box = st.container()

    class _Shim:
        def update(self, label: str = "", state: str = "running", expanded: bool = True):
            if not label:
                return
            if state == "complete":
                box.success(label)
            elif state == "error":
                box.error(label)
            else:
                box.info(label)

    box.info("Starting…")
    return _Shim()

def run_full_pipeline(
    materials: List[str],
    family: str,
    max_queries: int,
    papers_per_query: int,
    max_tags: int,
    use_relevance_gate: bool,
    use_gpt_query_plans: bool,
    use_optimade: bool,
    optimade_provider_ids: List[str],
    optimade_max_results_per_provider: int,
    use_openalex_api_key: bool,
    domain: str,
    meta_profile: Dict[str, Any],
    silent: bool = False,
) -> None:
    """
    Runs the end-to-end pipeline.

    Materials mode:
      OPTIMADE (optional) → OpenAlex papers → dynamic tags → scoring

    Metamaterials/EM devices mode:
      (OPTIMADE disabled) → OpenAlex papers → metamaterials-specific dynamic tags → scoring
    """
    if not materials:
        st.error("No items provided. Add candidates (materials or device concepts) in Tab 1 first.")
        return

    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY is missing. Required for dynamic tags (and GPT query plans).")
        return

    t0 = time.time()
    w = _progress_weights(len(materials))

    if not silent:
        status_box = _make_status_box()
        progress = st.progress(0.0)
        note = st.empty()
        eta_line = st.empty()
    else:
        status_box = None
        progress = None
        note = None
        eta_line = None

    def set_progress(frac: float, msg: str = "") -> None:
        if silent:
            return
        frac = max(0.0, min(1.0, float(frac)))
        try:
            progress.progress(frac)
        except Exception:
            pass
        elapsed = time.time() - t0
        if frac >= 0.03:
            remaining = elapsed * (1.0 - frac) / max(1e-6, frac)
            eta_line.caption(f"⏱ Elapsed: {_human_time(elapsed)}   •   ETA: {_human_time(remaining)}")
        else:
            eta_line.caption(f"⏱ Elapsed: {_human_time(elapsed)}")
        if msg:
            note.info(msg)

    cache = hec.CacheDB(str(PROJECT_DIR / CACHE_DB))

    st.session_state.query_plans = {}
    st.session_state.works = {}
    st.session_state.material_meta = {}
    st.session_state.tagset = None
    st.session_state.df_long = pd.DataFrame()
    st.session_state.optimade = {}

    p = 0.0

    if domain == "Materials" and use_optimade:
        if not OPTIMADE_CLIENT_OK:
            st.warning(
                "OPTIMADE is enabled but the OPTIMADE client could not be imported. "
                "Install with: `python -m pip install \"optimade[http_client]\"`"
            )
        else:
            if status_box:
                status_box.update(label="Step 1/4: OPTIMADE database search", state="running")
            set_progress(p + 0.01, "Querying OPTIMADE providers…")
            try:
                st.session_state.optimade = optimade_search_structures_for_materials(
                    materials=materials,
                    provider_ids=optimade_provider_ids,
                    max_results_per_provider=int(optimade_max_results_per_provider),
                )
                p += w["optimade"]
                set_progress(p, "OPTIMADE: done ✅")
                if status_box:
                    status_box.update(label="Step 1/4: OPTIMADE database search", state="complete")
            except Exception as e:
                set_progress(p + 0.02, f"OPTIMADE failed (continuing): {e}")
                if status_box:
                    status_box.update(label="Step 1/4: OPTIMADE database search (skipped)", state="complete")
    else:
        p += w["optimade"] * 0.2
        set_progress(p, "OPTIMADE: skipped")

    if status_box:

        status_box.update(label="Step 2/4: Retrieve papers", state="running")

    for i, item in enumerate(materials, 1):
        set_progress(p + (w["retrieve"] * (i - 1) / max(1, len(materials))),
                     f"Retrieving {i}/{len(materials)}: {item}")

        if domain == "Metamaterials / EM devices":
            mtype = "device_concept"
            elems = []
        else:
            mtype = "hec" if HEC_RE.match(item) else ("formula" if FORMULA_RE.match(item) else "named")
            elems = naive_elements_from_string(item)

        st.session_state.material_meta[item] = {"material_type": mtype, "elements": elems}

        db_ctx = db_context_for_material(st.session_state.optimade, item) if (domain == "Materials" and st.session_state.optimade) else ""

        if use_gpt_query_plans:
            try:
                if domain == "Metamaterials / EM devices":
                    plan = build_meta_query_plan(item, family, meta_profile)
                else:
                    plan = build_query_plan(item, family, db_context=db_ctx)
                plan["queries"] = (plan.get("queries") or [])[: int(max_queries)]
            except Exception:
                if domain == "Metamaterials / EM devices":
                    base_q = [
                        item,
                        f"{item} metasurface",
                        f"{item} reconfigurable metasurface",
                        f"{item} tunable metamaterial",
                        f"{item} frequency selective surface",
                        f"{item} beam steering RIS",
                    ]
                    plan = {"material": item, "aliases": [], "queries": base_q, "must_have_terms": ["metasurface", "metamaterial"]}
                else:
                    plan = {"material": item, "aliases": [], "queries": [item, f"{item} properties", f"{item} applications"], "must_have_terms": [item]}
        else:
            if domain == "Metamaterials / EM devices":
                base_q = [
                    item,
                    f"{item} metasurface",
                    f"{item} tunable metasurface",
                    f"{item} reconfigurable metamaterial",
                    f"{item} FSS filter",
                    f"{item} absorber",
                ]
                plan = {"material": item, "aliases": [], "queries": base_q, "must_have_terms": ["metasurface", "metamaterial"]}
            else:
                plan = {"material": item, "aliases": [], "queries": [item, f"{item} properties", f"{item} applications", f"{item} {family}"], "must_have_terms": [item]}

        st.session_state.query_plans[item] = plan

        works = retrieve_papers_openalex(
            plan.get("queries") or [item],
            per_query=int(papers_per_query),
            cache=cache,
            use_api_key=bool(use_openalex_api_key),
        )

        if use_relevance_gate:
            if domain == "Metamaterials / EM devices":
                works = meta_gate_relevance(works, item, meta_profile)
            else:
                anchors = [item] + (plan.get("aliases") or []) + (plan.get("must_have_terms") or [])
                if db_ctx:
                    anchors.extend(re.findall(r"[A-Za-z0-9\-]{3,}", db_ctx))
                works = gate_relevance(works, anchors)

        st.session_state.works[item] = works
        time.sleep(0.01)

    p += w["retrieve"]
    set_progress(p, "Retrieval: done ✅")
    if status_box:
        status_box.update(label="Step 2/4: Retrieve papers", state="complete")

    if status_box:

        status_box.update(label="Step 3/4: Discover dynamic tags", state="running")
    set_progress(p + 0.01, "Building corpus and discovering tags…")

    corpus = []
    seen = set()
    for works in st.session_state.works.values():
        for w0 in works:
            key = (w0.get("doi") or w0.get("id") or w0.get("title") or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            corpus.append(w0)

    if not corpus:
        st.error("No papers were retrieved after gating. Try increasing papers/query or disabling gating.")
        return

    if domain == "Metamaterials / EM devices":
        tagset = discover_tags_from_corpus_meta(family, corpus, max_tags=int(max_tags))
    else:
        tagset = discover_tags_from_corpus(family, corpus, max_tags=int(max_tags))

    st.session_state.tagset = tagset
    p += w["tagging"]
    set_progress(p, "Dynamic tags: done ✅")
    if status_box:
        status_box.update(label="Step 3/4: Discover dynamic tags", state="complete")

    if status_box:

        status_box.update(label="Step 4/4: Score and assemble results", state="running")
    tags = st.session_state.tagset.get("tags", []) if st.session_state.tagset else []
    scores: Dict[str, List[Dict[str, Any]]] = {}

    for i, item in enumerate(materials, 1):
        set_progress(p + (w["scoring"] * (i - 1) / max(1, len(materials))),
                     f"Scoring {i}/{len(materials)}: {item}")
        scores[item] = score_material(st.session_state.works.get(item, []), tags)
        time.sleep(0.005)

    st.session_state.df_long = build_long_table(scores, st.session_state.material_meta)

    (OUT_CSV_DIR / "applications_long_universal_dynamic.csv").write_text(
        st.session_state.df_long.to_csv(index=False),
        encoding="utf-8-sig",
    )

    # Build/update similarity library for fast recommendations (Materials mode only)
    if similarity is not None and domain == "Materials":
        try:
            lib = similarity.build_library_from_df_long(st.session_state.df_long)
            if lib is not None:
                similarity.save_library(lib, SIM_LIB_FILE)
        except Exception:
            pass

    set_progress(1.0, "All done ✅")
    if status_box:
        status_box.update(label="All steps complete ✅", state="complete")
    if not silent and note is not None:
        note.success("✅ Application Finder is ready. Open Tab 4 (Results).")
    if not silent:
        st.toast("Application Finder completed", icon="✅")

# =========================
# Session state
# =========================
if "domain" not in st.session_state:
    st.session_state.domain = "Materials"

if "meta_profile" not in st.session_state:
    st.session_state.meta_profile = {
        "bands": ["Microwave", "mmWave"],
        "band_hint": "e.g., X/Ku/Ka or 28–39 GHz",
        "device_archetypes": ["Metasurface", "RIS", "FSS / filter"],
        "tuning": ["Phase-change (VO2/GST)", "Varactor", "MEMS"],  # ASCII VO2
        "active_materials": ["VO2", "GST", "Graphene/ITO"],        # ASCII VO2
        "geometry_keywords": "meta-atom, Huygens, split-ring, patch array",
    }

# --- Default goal texts (one per domain) ---
MATERIALS_DEFAULT_FAMILY = (
    "Find dielectric materials with high-k, low leakage (wide band gap), and good stability "
    "up to ~300–500 °C (processing / operating conditions)."
)

META_DEFAULT_FAMILY = (
    "Tunable metasurfaces/metamaterials for EM control. Prioritize low insertion loss, wide bandwidth"
)

# Store separate goal text per mode
if "family_materials" not in st.session_state:
    st.session_state.family_materials = MATERIALS_DEFAULT_FAMILY

if "family_meta" not in st.session_state:
    st.session_state.family_meta = META_DEFAULT_FAMILY

# This is the "active" goal shown in the UI and used by the pipeline
if "family" not in st.session_state:
    st.session_state.family = (
        st.session_state.family_materials
        if st.session_state.domain == "Materials"
        else st.session_state.family_meta
    )

if "materials" not in st.session_state:
    st.session_state.materials = []
if "allowed_elements" not in st.session_state:
    st.session_state.allowed_elements = []
if "query_plans" not in st.session_state:
    st.session_state.query_plans = {}
if "works" not in st.session_state:
    st.session_state.works = {}
if "tagset" not in st.session_state:
    st.session_state.tagset = None
if "df_long" not in st.session_state:
    st.session_state.df_long = pd.DataFrame()
if "material_meta" not in st.session_state:
    st.session_state.material_meta = {}
if "optimade" not in st.session_state:
    st.session_state.optimade = {}

# =========================
# Sidebar controls
# =========================
st.sidebar.header("Run settings")
mode = st.sidebar.selectbox(
    "Mode",
    ["Materials", "Metamaterials / EM devices"],
    index=0 if st.session_state.domain == "Materials" else 1
)

# If the user changed mode: save current goal into the old bucket, then load the new bucket
prev_domain = st.session_state.domain
if mode != prev_domain:
    # save what user typed in the old mode
    if prev_domain == "Materials":
        st.session_state.family_materials = st.session_state.family
    else:
        st.session_state.family_meta = st.session_state.family

    # switch mode
    st.session_state.domain = mode

    # load the goal for the new mode
    st.session_state.family = (
        st.session_state.family_materials
        if st.session_state.domain == "Materials"
        else st.session_state.family_meta
    )
else:
    # keep buckets in sync even if mode didn't change
    if st.session_state.domain == "Materials":
        st.session_state.family_materials = st.session_state.family
    else:
        st.session_state.family_meta = st.session_state.family

if "prev_domain" not in st.session_state:
    st.session_state.prev_domain = st.session_state.domain

if st.session_state.domain != st.session_state.prev_domain:
    # set a good default family per domain ONLY if user hasn't customized it
    MATERIALS_DEFAULT = "Screening functional materials for dielectric applications (high-k, stable, wide-bandgap)."
    META_DEFAULT = (
        "Find evidence-backed metamaterials / metasurface device directions for tunable EM control"
    )

    # only overwrite if it's still the old default (or empty)
    if (st.session_state.family or "").strip() in ("", MATERIALS_DEFAULT, META_DEFAULT):
        st.session_state.family = META_DEFAULT if st.session_state.domain == "Metamaterials / EM devices" else MATERIALS_DEFAULT

    st.session_state.prev_domain = st.session_state.domain


use_gpt_candidates = st.sidebar.checkbox("Generate candidates with GPT", value=True)
use_gpt_query_plans = st.sidebar.checkbox("Use GPT query plans", value=True)
use_relevance_gate = st.sidebar.checkbox("Relevance gating", value=True)

st.sidebar.divider()
st.sidebar.subheader("OPTIMADE database search")
if st.session_state.domain == "Materials":
    use_optimade = st.sidebar.checkbox("Use OPTIMADE (databases)", value=True)
else:
    use_optimade = False
    st.sidebar.info("OPTIMADE is disabled in Metamaterials/EM devices mode (it is a materials DB standard).")

if use_optimade and not OPTIMADE_CLIENT_OK:
    st.sidebar.warning(
        "OPTIMADE is enabled but the OPTIMADE client could not be imported.\n\n"
        "Install (in the same venv that runs Streamlit):\n"
        "`python -m pip install \"optimade[http_client]\"`\n\n"
        f"Import error: {OPTIMADE_CLIENT_IMPORT_ERR}"
    )
elif use_optimade and OPTIMADE_CLIENT_OK and not GET_PROVIDERS_OK:
    st.sidebar.info(
        "OPTIMADE client is available, but `get_providers` could not be imported in your optimade version. "
        "That's okay: the app will fetch providers via the public providers endpoint instead. "
        f"(Import error: {GET_PROVIDERS_IMPORT_ERR})"
    )

provider_opts = optimade_provider_options() if OPTIMADE_CLIENT_OK else []
provider_fetch_error = ""
if OPTIMADE_CLIENT_OK and use_optimade and not provider_opts:
    provider_fetch_error = "Provider list returned empty (providers.optimade.org may be blocked or unavailable)."

provider_labels = [f"{p['id']} — {p['name']}" for p in provider_opts]
if use_optimade and OPTIMADE_CLIENT_OK and (provider_fetch_error or not provider_labels):
    provider_labels = FALLBACK_PROVIDER_LABELS.copy()
    label_to_id = {lab: lab.split(' — ', 1)[0] for lab in provider_labels}
    st.sidebar.warning(
        "OPTIMADE provider dropdown is using a built-in fallback list (could not load live provider registry). "
        "You can still run using these providers, or type manual IDs below."
    )
    if provider_fetch_error:
        st.sidebar.caption(provider_fetch_error)

label_to_id = {lab: lab.split(' — ', 1)[0].strip() for lab in provider_labels}

default_provider_ids = ["mp", "oqmd", "cod", "nmd", "jarvis", "omdb", "odbx", "mcloud"]
default_labels = [lab for lab in provider_labels if lab.split(" — ", 1)[0] in set(default_provider_ids)]

selected_labels = st.sidebar.multiselect(
    "Providers",
    options=provider_labels,
    default=default_labels if provider_labels else [],
    disabled=not (use_optimade and OPTIMADE_CLIENT_OK),
)

manual_provider_ids = ""
if use_optimade and OPTIMADE_CLIENT_OK and (provider_fetch_error or not provider_labels):
    manual_provider_ids = st.sidebar.text_input(
        "Providers (manual IDs, comma-separated)",
        value="mp,oqmd,cod",
        help="Fallback if provider list cannot be loaded. Example: mp,oqmd,cod",
    )

def _label_to_provider_id(lab: str) -> str:
    if lab in label_to_id:
        return str(label_to_id[lab])
    if ' — ' in lab:
        return lab.split(' — ', 1)[0].strip()
    return lab.strip()

optimade_provider_ids = [_label_to_provider_id(x) for x in selected_labels] if selected_labels else []
if manual_provider_ids.strip():
    optimade_provider_ids = [p.strip() for p in manual_provider_ids.split(',') if p.strip()]

optimade_max_results_per_provider = st.sidebar.slider(
    "Max results per provider (cap)",
    min_value=5,
    max_value=200,
    value=25,
    step=5,
    disabled=not (use_optimade and OPTIMADE_CLIENT_OK),
)

st.sidebar.divider()
max_queries = st.sidebar.slider("Max queries/material", 6, 30, 18, 1)
papers_per_query = st.sidebar.slider("Papers per query", 5, 50, 20, 5)
max_tags = st.sidebar.slider("Dynamic tags", 6, 12, 8, 1)

st.sidebar.divider()
st.sidebar.caption(f"Cache DB: {CACHE_DB}")

use_openalex_api_key = st.sidebar.checkbox(
    "Use OpenAlex API key (if set)",
    value=False,
    help="If you set OPENALEX_API_KEY but have $0 remaining, OpenAlex returns 429 Insufficient budget. "
         "Leave this OFF to run without the key (free mode).",
)


st.sidebar.divider()
st.sidebar.subheader("Similarity recommender (optional)")
if st.session_state.domain != "Materials":
    st.sidebar.caption("Similarity mode is currently enabled for Materials only (composition-based).")
else:
    sim_available = (similarity is not None)
    if not sim_available:
        st.sidebar.caption("Install pymatgen + matminer + scikit-learn to enable similarity mode.")
    st.sidebar.checkbox("⚡ Enable fast similarity recommendations", key="use_similarity_mode", value=False, disabled=not sim_available)
    st.sidebar.slider("Similarity: top-k neighbors", 5, 40, 15, 1, key="sim_top_k", disabled=not sim_available)
    st.sidebar.slider("Similarity: minimum cosine", 0.50, 0.95, 0.70, 0.01, key="sim_min", disabled=not sim_available)
    st.sidebar.slider("Similarity: neighbor weight α", 1.0, 4.0, 2.0, 0.1, key="sim_alpha", disabled=not sim_available)
    if sim_available:
        if st.sidebar.button("Build/update similarity library now"):
            if st.session_state.df_long is not None and not st.session_state.df_long.empty:
                try:
                    lib = similarity.build_library_from_df_long(st.session_state.df_long)
                    if lib is None:
                        st.sidebar.warning("Could not build library from current results (no featurizable materials).")
                    else:
                        similarity.save_library(lib, SIM_LIB_FILE)
                        st.sidebar.success("Similarity library updated.")
                except Exception as e:
                    st.sidebar.error(f"Similarity library build failed: {e}")
            else:
                st.sidebar.info("Run the normal pipeline once to create results, then build the library.")
# =========================
# Presets
# =========================
PRESETS: Dict[str, List[str]] = {
    "Refractory TMs": ["Ti", "Zr", "Hf", "V", "Nb", "Ta", "Mo", "W"],
    "Dielectric oxides": ["Hf", "Zr", "Ti", "Al", "Si", "O", "La", "Y", "Sr", "Ba"],
    "Perovskite oxides (ABO3-ish)": ["Ca","Sr","Ba","La","Y","Ti","Zr","Hf","Nb","Ta","O"],
    "Battery (Li-ion common)": ["Li","Ni","Mn","Co","Fe","P","O","C","Al","Cu"],
    "Chalcogenides (PCM-ish)": ["Ge","Sb","Te","Se","S","Sn","In"],
    "Nitrides": ["Ti","Zr","Hf","Al","Si","Ta","Nb","N"],
    "Catalysis TMs": ["Fe","Co","Ni","Cu","Zn","Mo","W","V","Mn"],
    "Transparent conductors": ["In","Sn","Zn","Ga","O","F"],
}

# =========================
# Tabs
# =========================
tabs = st.tabs([
    "1) Setup",
    "2) Run Application Finder",
    "3) Inspect (optional)",
    "4) Results",
    "5) New candidate (Similarity)",
    "6) Batch CSV screening",
])
# -------------------------
# Tab 1: Setup
# -------------------------
with tabs[0]:
    st.subheader("Setup")

    if st.session_state.domain == "Metamaterials / EM devices":
        st.markdown(
            """
<div class="card">
<b>Metamaterials / EM devices mode</b><br>
Define a device-level goal, choose band/device/tuning options, then generate <b>device concepts</b> (query seeds).<br>
These concepts drive literature retrieval and dynamic application tagging.
</div>
""",
            unsafe_allow_html=True,
        )

        st.session_state.family = st.text_area(
            "Goal / scope (device-focused)",
            value=st.session_state.family,
            height=110,
            help="Keep it short: device archetype + band + tuning + what you want to optimize (loss, bandwidth, switching, modulation depth).",
        )

        meta = st.session_state.meta_profile

        c1, c2 = st.columns([1, 1])
        with c1:
            meta["bands"] = st.multiselect(
                "Bands",
                options=list(META_BAND_KEYWORDS.keys()),
                default=meta.get("bands", ["Microwave", "mmWave"]),
            )
        with c2:
            meta["band_hint"] = st.text_input("Band hint (optional)", value=meta.get("band_hint", ""))

        c3, c4 = st.columns([1, 1])
        with c3:
            meta["device_archetypes"] = st.multiselect(
                "Device archetypes",
                options=META_DEVICE_TYPES,
                default=meta.get("device_archetypes", ["Metasurface", "RIS"]),
            )
        with c4:
            meta["tuning"] = st.multiselect(
                "Tuning mechanisms",
                options=META_TUNING,
                default=meta.get("tuning", ["Phase-change (VO2/GST)", "Varactor", "MEMS"]),
            )

        c5, c6 = st.columns([1, 1])
        with c5:
            meta["device_functions"] = st.multiselect(
                "Target functions",
                options=[
                    "RF switch / modulator",
                    "Beam steering / RIS",
                    "Tunable absorber / stealth",
                    "Filter / FSS",
                    "Polarization control",
                    "Sensing",
                    "Lens / holography",
                    "Antenna tuning",
                ],
                default=meta.get("device_functions", ["RF switch / modulator", "Beam steering / RIS"]),
            )
        with c6:
            meta["active_materials"] = st.multiselect(
                "Active materials / platforms",
                options=["VO2", "GST", "Graphene/ITO", "Liquid crystal", "Ferroelectric (BST)", "Varactor", "PIN diode", "MEMS"],
                default=meta.get("active_materials", ["VO2", "GST", "Graphene/ITO"]),
            )

        meta["geometry_keywords"] = st.text_input(
            "Geometry keywords (optional)",
            value=meta.get("geometry_keywords", ""),
            help="Examples: meta-atom, Huygens, split-ring, fishnet, patch array, resonator.",
        )

        st.session_state.meta_profile = meta

        st.markdown("#### Device concepts (query seeds)")
        n = st.number_input("How many concepts", min_value=1, max_value=50, value=12, step=1)

        colA, colB = st.columns([1, 1])
        with colA:
            if st.button("🧠 Generate device concepts", type="primary", disabled=not (OPENAI_API_KEY and use_gpt_candidates)):
                try:
                    interp, concepts = generate_meta_concepts(st.session_state.family, st.session_state.meta_profile, int(n))
                    st.info(interp or "Generated device concepts.")
                    st.session_state.materials = concepts
                except Exception as e:
                    st.error(str(e))
        with colB:
            if st.button("🧹 Clear concepts", key="clear_meta_concepts"):
                st.session_state.materials = []

        editable = st.text_area("Edit concepts (one per line)", value="\n".join(st.session_state.materials), height=260)
        st.session_state.materials = [x.strip() for x in editable.splitlines() if x.strip()]

        st.markdown(
            "<div class='small'>Next: go to <b>Tab 2</b> and click <b>Run Application Finder</b>.</div>",
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
        <div class="card">
        <b>Do this once:</b><br>
        1) Describe your goal (family) & pick allowed elements (optional)<br>
        2) Generate / edit your candidate materials<br>
        Then go to <b>Tab 2</b> and click <b>Run Application Finder</b>.
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.session_state.family = st.text_area(
            "Family / goal description (steers candidate generation + tag discovery)",
            value=st.session_state.family,
            height=90,
        )

        st.markdown("#### Select allowed elements (optional)")

        with st.expander("Open periodic table", expanded=False):
            st.markdown('<div class="pt-wrap">', unsafe_allow_html=True)

            st.markdown(
                """
                <div class="pt-legend">
                  <span>🔴 Alkali</span><span>🟠 Alkaline</span><span>🔵 Transition</span>
                  <span>🟣 Post/Act</span><span>🟢 Met/Lanth</span><span>🟡 Nonmetal</span>
                  <span>⚪ Halogen</span><span>⚫ Noble</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            periodic_table_widget(show_group_numbers=False)

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown("#### Preset element sets")
        preset_names = list(PRESETS.keys())
        rows = [preset_names[:4], preset_names[4:]]
        for r in rows:
            cols = st.columns(4)
            for i, name in enumerate(r):
                if cols[i].button(name, key=f"preset_{name}"):
                    st.session_state.allowed_elements = [e for e in PRESETS.get(name, []) if e in ELEMENTS]
                    st.session_state.allowed_elements.sort()
                    st.rerun()

        cA, cB = st.columns([1, 3])
        with cA:
            if st.button("Clear selection", key="clear_sel"):
                st.session_state.allowed_elements = []
                st.rerun()
        with cB:
            st.caption("Selected: " + (", ".join(st.session_state.allowed_elements) if st.session_state.allowed_elements else "(none)"))

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown("#### Candidate materials")
        n = st.number_input("How many candidates", min_value=1, max_value=50, value=10, step=1)

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🧠 Generate candidates", type="primary", disabled=not (OPENAI_API_KEY and use_gpt_candidates)):
                try:
                    interp, mats = generate_candidates_any(st.session_state.family, st.session_state.allowed_elements, int(n))
                    st.info(interp or "Generated candidates.")
                    st.session_state.materials = mats
                except Exception as e:
                    st.error(str(e))
        with col2:
            if st.button("🧹 Clear candidates", key="clear_cands"):
                st.session_state.materials = []

        editable = st.text_area("Edit (one per line)", value="\n".join(st.session_state.materials), height=220)
        st.session_state.materials = [x.strip() for x in editable.splitlines() if x.strip()]

        st.markdown(
            "<div class='small'>Next step: go to <b>Tab 2</b> and click <b>Run Application Finder</b>.</div>",
            unsafe_allow_html=True,
        )

# -------------------------
# Tab 2: Run
# -------------------------
with tabs[1]:
    st.subheader("Run Application Finder")
    st.markdown(
        """
<div class="card">
<b>This is the main action.</b><br>
Runs: retrieve papers → discover dynamic tags → score → populate Results. (OPTIMADE DB step only in Materials mode.)
</div>
""",
        unsafe_allow_html=True,
    )

    if st.button("🚀 Run Application Finder", type="primary"):
        run_full_pipeline(
            materials=st.session_state.materials,
            family=st.session_state.family,
            max_queries=int(max_queries),
            papers_per_query=int(papers_per_query),
            max_tags=int(max_tags),
            use_relevance_gate=bool(use_relevance_gate),
            use_gpt_query_plans=bool(use_gpt_query_plans),
            use_optimade=bool(use_optimade),
            optimade_provider_ids=optimade_provider_ids,
            optimade_max_results_per_provider=int(optimade_max_results_per_provider),
            use_openalex_api_key=bool(use_openalex_api_key),
            domain=str(st.session_state.domain),
            meta_profile=dict(st.session_state.meta_profile),
        )

    if st.session_state.df_long is not None and not st.session_state.df_long.empty:
        st.success("Results are ready. Open Tab 4.")

# -------------------------
# Tab 3: Inspect (optional)
# -------------------------
with tabs[2]:
    st.subheader("Inspect (optional)")
    st.caption("Use this if you want to sanity-check the DB hits, query plans, retrieved papers, or the tagset.")


    st.divider()
    st.subheader("Similarity library (optional)")

    if st.session_state.domain != "Materials":
        st.info("Similarity library is currently built/used for Materials mode only.")
    elif similarity is None:
        st.info("Similarity module not available. Install pymatgen + matminer + scikit-learn to enable it.")
    else:
        lib = similarity.load_library(SIM_LIB_FILE)
        if lib is None:
            st.warning(
                "No similarity library found yet. Run the normal pipeline once (Deep Verify) in Materials mode "
                "to generate results and build the library."
            )
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Materials", len(lib.materials))
            with c2:
                st.metric("Tags", len(lib.tag_names))
            with c3:
                st.metric("Feature dim", int(lib.X.shape[1]))
            with c4:
                st.metric("Library file", SIM_LIB_FILE.name)

            rows = []
            for i, m in enumerate(lib.materials):
                try:
                    j = int(lib.tag_matrix[i].argmax())
                    top_tag = lib.tag_names[j]
                    top_score = float(lib.tag_matrix[i][j])
                except Exception:
                    top_tag, top_score = "", 0.0
                rows.append({"material": m, "top_tag": top_tag, "top_tag_score": round(top_score, 3)})

            st.markdown("**Library materials (top tag per material)**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            with st.expander("Tag descriptions"):
                tag_rows = [{"tag": t, "description": (lib.tag_desc.get(t, "") or "")} for t in lib.tag_names]
                st.dataframe(pd.DataFrame(tag_rows), use_container_width=True, hide_index=True)

            try:
                st.download_button(
                    "Download similarity library (sim_library.pkl)",
                    data=SIM_LIB_FILE.read_bytes(),
                    file_name="sim_library.pkl",
                    mime="application/octet-stream",
                )
            except Exception:
                pass

    if st.session_state.optimade:
        with st.expander("OPTIMADE DB hits"):
            st.json(st.session_state.optimade)

    if st.session_state.query_plans:
        with st.expander("Query plans"):
            st.json(st.session_state.query_plans)

    if st.session_state.works:
        with st.expander("Sample retrieved papers"):
            for m, works in list(st.session_state.works.items())[:5]:
                st.markdown(f"**{m}** — {len(works)} papers")
                for w in works[:5]:
                    title = w.get("title") or ""
                    year = w.get("year") or ""
                    doi = w.get("doi") or ""
                    st.write(f"- {title} ({year}) {doi_link(str(doi)) if doi else ''}")

    if st.session_state.tagset:
        with st.expander("Dynamic tagset"):
            st.json(st.session_state.tagset)

# -------------------------
# Tab 4: Results
# -------------------------
with tabs[3]:
    st.subheader("Results")

    entity_label = "Material" if st.session_state.domain == "Materials" else "Device concept"
    second_view_label = "By material" if st.session_state.domain == "Materials" else "By device concept"


    df = st.session_state.df_long
    if df is None or df.empty:
        st.info("No results yet. Go to Tab 2 and run the Application Finder.")
    else:
        df = df.copy()
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)

        tags = sorted(df["application_tag"].astype(str).unique().tolist())
        materials = sorted(df["material"].astype(str).unique().tolist())

        c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
        with c1:
            view_mode = st.selectbox("View", ["By application", second_view_label], index=0)
        with c2:
            min_s = st.slider("Min score", 0.0, 1.0, 0.6, 0.01)
        with c3:
            topN = st.slider("Show top N", 5, 60, 20, 5)
        with c4:
            show_evidence = st.checkbox("Show evidence (papers)", value=True)

        st.divider()

        if view_mode == "By application":
            tag = st.selectbox("Application tag", tags, index=0)
            df_t = df[(df["application_tag"] == tag) & (df["score"] >= min_s)].copy()
            df_t.sort_values("score", ascending=False, inplace=True)
            df_t = df_t.head(topN)

            st.markdown(f"### {tag}")
            dlist = df_t["description"].dropna().astype(str)
            desc = dlist[dlist != ""].head(1).tolist()[0] if not dlist[dlist != ""].empty else ""
            if desc:
                st.caption(desc)

            for _, row in df_t.iterrows():
                material = row["material"]
                score = float(row["score"])
                elems = row.get("elements", "")

                st.markdown('<div class="card">', unsafe_allow_html=True)
                top = st.columns([2.8, 1.2])
                with top[0]:
                    if st.session_state.domain == "Materials":
                        st.markdown(f"**{material}**  \n<span class='small'>Elements: {elems}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**{material}**", unsafe_allow_html=True)
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

        else:
            material = st.selectbox(entity_label, materials, index=0)
            df_m = df[df["material"] == material].copy()
            df_m.sort_values("score", ascending=False, inplace=True)

            st.markdown(f"### {material}")

            if st.session_state.optimade and (st.session_state.optimade.get("by_material") or {}).get(material):
                mobj = (st.session_state.optimade.get("by_material") or {}).get(material) or {}
                flt = mobj.get("filter")
                res = mobj.get("results") or {}
                if flt and res:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("**Database evidence (OPTIMADE)**")
                    st.caption(f"Filter: {flt}")
                    rows = []
                    for base_url, info in res.items():
                        top = (info.get("top") or [])[:1]
                        if top:
                            t0 = top[0]
                            rows.append(
                                {
                                    "base_url": base_url,
                                    "hits": info.get("meta_data_returned") or info.get("count_returned") or 0,
                                    "example_formula": t0.get("formula", ""),
                                    "example_id": t0.get("id", ""),
                                    "link": t0.get("url", ""),
                                }
                            )
                        else:
                            rows.append({"base_url": base_url, "hits": info.get("meta_data_returned") or info.get("count_returned") or 0,
                                         "example_formula": "", "example_id": "", "link": ""})
                    if rows:
                        dfdb = pd.DataFrame(rows).sort_values("hits", ascending=False).head(10)
                        st.dataframe(dfdb[["base_url", "hits", "example_formula", "example_id"]], use_container_width=True, hide_index=True)
                        with st.expander("Links"):
                            for r in dfdb.to_dict("records"):
                                if r.get("link"):
                                    st.markdown(f"- {r['base_url']}: [{r['example_id']}]({r['link']})")
                    st.markdown("</div>", unsafe_allow_html=True)

            elems = (df_m["elements"].head(1).tolist() or [""])[0]
            if st.session_state.domain == "Materials" and elems:
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

        with st.expander("Export (optional)"):
            st.download_button(
                "Download long table CSV (material × tag)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="applications_long_universal_dynamic.csv",
                mime="text/csv",
            )
            if st.session_state.tagset:
                st.download_button(
                    "Download dynamic tagset JSON",
                    data=json.dumps(st.session_state.tagset, indent=2, ensure_ascii=False).encode("utf-8"),
                    file_name="dynamic_tags_universal.json",
                    mime="application/json",
                )



# -------------------------
# Tab 5: New candidate (Similarity)
# -------------------------
with tabs[4]:
    st.subheader("New candidate (Similarity)")

    if st.session_state.domain != "Materials":
        st.info("This workflow is currently enabled for Materials mode only.")
    elif similarity is None:
        st.warning("Similarity mode requires pymatgen + matminer + scikit-learn. Install them to use this tab.")
    else:
        cache = hec.CacheDB(str(PROJECT_DIR / CACHE_DB))

        new_mat = st.text_input(
            "Enter a NEW candidate material (formula)",
            value="",
            placeholder="e.g., Hf0.5Zr0.5O2  or  SrHfO3  or  (Ta0.2Nb0.2Ti0.2Sc0.2Hf0.2)C",
        )

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            run_check = st.button("🔎 Check if new", disabled=not new_mat.strip(), key="new_check")
        with c2:
            run_sim = st.button("⚡ Recommend by similarity", disabled=not new_mat.strip(), key="new_sim")
        with c3:
            st.caption("Tip: Build the similarity library first by running Deep Verify once in Tab 2 (Materials mode).")

        lib = similarity.load_library(SIM_LIB_FILE)

        if run_check:
            if lib is None:
                st.error("No similarity library found. Run Deep Verify once in Tab 2 to build it.")
            else:
                if new_mat.strip() in set(lib.materials):
                    st.success("✅ This candidate already exists in your similarity library (not new to the system).")
                else:
                    hits = quick_openalex_presence_check(new_mat.strip(), cache=cache, per_page=8, use_api_key=False)

                    if hits >= 3:

                        st.warning(f"⚠️ Likely NOT new in literature (OpenAlex quick hits ≈ {hits}).")


                        works = quick_openalex_top_works(new_mat.strip(), cache=cache, per_page=10, use_api_key=False)

                        if works:

                            st.markdown("### Top relevant papers (quick OpenAlex)")

                            for w in works:

                                title = (w.get("title") or "").strip()

                                year = w.get("year") or ""

                                venue = w.get("venue") or ""

                                doi = w.get("doi") or ""

                                link = doi_link(str(doi)) if doi else ""

                                line = f"- **{title}** ({year}) — {venue}"

                                if link:

                                    line += "  \n  " + link

                                st.markdown(line)


                            with st.expander("Abstract snippets (optional)"):

                                for w in works:

                                    title = (w.get("title") or "").strip()

                                    snip = (w.get("abstract_snippet") or "").strip()

                                    if snip:

                                        st.markdown(f"**{title}**")

                                        st.write(snip[:600] + ("…" if len(snip) > 600 else ""))

                        else:

                            st.info("OpenAlex returned few/no detailed results for this quick query.")


                        st.caption("If you want evidence-backed application tags, run Deep Verify in Tab 2.")

                    else:

                        st.success(f"✅ Likely NEW / sparse in literature (OpenAlex quick hits ≈ {hits}).")

                        st.caption("Use similarity recommendation below.")


        if run_sim:
            if lib is None:
                st.error("No similarity library found. Run Deep Verify once in Tab 2 to build it.")
            else:
                rec = similarity.recommend_by_similarity(
                    new_mat.strip(),
                    lib,
                    top_k=int(st.session_state.get("sim_top_k", 15)),
                    alpha=float(st.session_state.get("sim_alpha", 2.0)),
                    min_sim=float(st.session_state.get("sim_min", 0.70)),
                )
                if "error" in rec:
                    st.error(rec["error"])
                else:
                    conf = rec.get("confidence", {})
                    st.markdown("### Similarity results")
                    st.caption(
                        f"Confidence: top1_sim={conf.get('top1_sim')} • neighbors={conf.get('n_neighbors')} "
                        f"• min_sim={conf.get('min_sim')} • α={conf.get('alpha')}"
                    )

                    neighbors = _normalize_neighbor_labels(rec.get("neighbors") or [])
                    center_name = format_material_name_generic(new_mat.strip())

                    with st.expander("Nearest neighbors", expanded=True):
                        df_neighbors = pd.DataFrame(neighbors)
                        if not df_neighbors.empty:
                            st.dataframe(df_neighbors, use_container_width=True, hide_index=True)
                        else:
                            st.info("No neighbors available.")

                    show_material_network_graph(center_name, neighbors[:8], title="kNN Composition Network")

                    st.markdown("### Recommended applications")
                    df_scores = rec["tag_scores"].copy()
                    df_scores["score"] = pd.to_numeric(df_scores["score"], errors="coerce").fillna(0.0)
                    st.dataframe(df_scores.head(15), use_container_width=True, hide_index=True)

                    df_edges, app_scores = build_application_contributions(
                        lib,
                        neighbors,
                        alpha=float(st.session_state.get("sim_alpha", 2.0)),
                        top_apps=8,
                    )
                    if not df_edges.empty and not app_scores.empty:
                        show_application_evidence_graph(
                            center_name,
                            neighbors[:8],
                            df_edges,
                            app_scores,
                            title="Composition → Application Evidence Network",
                        )

                    st.info("If you want literature evidence, run Deep Verify (Tab 2) using this candidate as input.")


# -------------------------
# Tab 6: Batch CSV screening
# -------------------------
with tabs[5]:
    st.subheader("Batch CSV screening (new vs known)")

    if st.session_state.domain != "Materials":
        st.info("Batch CSV workflow is currently enabled for Materials mode only.")
        st.stop()

    if similarity is None:
        st.warning("Similarity mode requires pymatgen + matminer + scikit-learn. Install them to use this tab.")
        st.stop()

    st.markdown(
        """Upload a CSV with candidate materials. The app will process each row:

- **If NOT new** (in similarity library OR OpenAlex has hits): run **Deep Verify** and **update the similarity library**
- **If NEW/sparse**: recommend applications using **Similarity kNN transfer**

CSV must contain a column named: `material` or `composition` (first found is used).
"""
    )

    up = st.file_uploader("Upload candidates CSV", type=["csv", "xlsx", "xls"], key="batch_csv")
    if up is None:
        st.stop()

    name = (up.name or "").lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        df_in = pd.read_excel(up)
    else:
        df_in = read_csv_robust(up)

    col_pick = None
    for c in df_in.columns:
        if str(c).lower() in ("material", "composition", "formula", "candidate", "structured_formula"):
            col_pick = c
            break

    if col_pick is None:
        st.error(f"Could not find a material column. Found columns: {list(df_in.columns)}")
        st.stop()

    df_in = df_in.copy()
    df_in[col_pick] = df_in[col_pick].astype(str)
    df_in["material_norm"] = df_in[col_pick].map(normalize_candidate_string)
    df_in = df_in[df_in["material_norm"].str.len() > 0].drop_duplicates(subset=["material_norm"])

    st.write(f"Loaded {len(df_in)} unique candidates from column '{col_pick}'.")

    # Persist results so UI interactions (selectbox) do NOT force re-running the batch
    if "batch_df_res" not in st.session_state:
        st.session_state.batch_df_res = None
    if "batch_deepverify_long" not in st.session_state:
        st.session_state.batch_deepverify_long = []
    if "batch_papers" not in st.session_state:
        st.session_state.batch_papers = {}

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        hits_threshold = st.number_input("OpenAlex hits threshold (not-new)", 1, 20, 3, 1)
    with c2:
        per_probe = st.number_input("Probe papers per variant", 2, 20, 6, 1)
    with c3:
        st.markdown("<div style='height: 1.8rem'></div>", unsafe_allow_html=True)
        run_button = st.button("🚀 Run batch screening", key="run_batch", use_container_width=True)

    # Run batch once when button clicked
    if run_button:
        cache = hec.CacheDB(str(PROJECT_DIR / CACHE_DB))
        lib = similarity.load_library(SIM_LIB_FILE)

        if lib is None:
            st.warning("No similarity library found yet. Run Deep Verify once in Tab 2 to build it before batch mode.")
            st.stop()

        results_rows = []
        deepverify_long = []

        progress = st.progress(0.0)
        status = st.empty()

        def _deep_verify_one(mat: str) -> pd.DataFrame:
            # Run full pipeline for a single material (evidence mode)
            run_full_pipeline(
                materials=[mat],
                family=st.session_state.family,
                max_queries=int(max_queries),
                papers_per_query=int(papers_per_query),
                max_tags=int(max_tags),
                use_relevance_gate=bool(use_relevance_gate),
                use_gpt_query_plans=bool(use_gpt_query_plans),
                use_optimade=bool(use_optimade),
                optimade_provider_ids=optimade_provider_ids,
                optimade_max_results_per_provider=int(optimade_max_results_per_provider),
                use_openalex_api_key=bool(use_openalex_api_key),
                domain=str(st.session_state.domain),
                meta_profile=dict(st.session_state.meta_profile),
            )
            return st.session_state.df_long.copy() if st.session_state.df_long is not None else pd.DataFrame()

        for i, mat in enumerate(df_in["material_norm"].tolist(), 1):
            progress.progress((i - 1) / max(1, len(df_in)))
            status.info(f"Processing {i}/{len(df_in)}: {mat}")

            in_lib = mat in set(lib.materials)
            hits = 0 if in_lib else quick_openalex_presence_check(mat, cache=cache, per_page=int(per_probe), use_api_key=bool(use_openalex_api_key))

            is_not_new = in_lib or (hits >= int(hits_threshold))

            if is_not_new:
                # Evidence run + incremental library update
                try:
                    df_one = _deep_verify_one(mat)
                    if df_one is not None and not df_one.empty:
                        deepverify_long.append(df_one)
                        df_concat = pd.concat(deepverify_long, ignore_index=True)

                        lib_new = similarity.build_library_from_df_long(df_concat)
                        if lib_new is not None:
                            similarity.save_library(lib_new, SIM_LIB_FILE)
                            lib = lib_new

                    results_rows.append({
                        "material": mat,
                        "status": "known/not-new",
                        "openalex_hits": hits,
                        "action": "deep_verify",
                    })
                except Exception as e:
                    results_rows.append({
                        "material": mat,
                        "status": "known/not-new",
                        "openalex_hits": hits,
                        "action": f"deep_verify_failed: {e}",
                    })
            else:
                # Similarity kNN transfer
                rec = similarity.recommend_by_similarity(
                    normalize_candidate_string(mat).split(":")[0].strip(),
                    lib,
                    top_k=int(st.session_state.get("sim_top_k", 15)),
                    alpha=float(st.session_state.get("sim_alpha", 2.0)),
                    min_sim=float(st.session_state.get("sim_min", 0.70)),
                )

                neighbors = rec.get("neighbors") or []

                def _pick_nn(k: int):
                    if k < len(neighbors):
                        return neighbors[k].get("material", ""), neighbors[k].get("sim", "")
                    return "", ""

                nn1_m, nn1_s = _pick_nn(0)
                nn2_m, nn2_s = _pick_nn(1)
                nn3_m, nn3_s = _pick_nn(2)

                df_scores = rec.get("tag_scores")
                if df_scores is None:
                    results_rows.append({
                        "material": mat,
                        "status": "new/sparse",
                        "openalex_hits": hits,
                        "action": f"similarity_failed: {rec.get('error', '')}",
                    })
                else:
                    df_scores = df_scores.copy()
                    df_scores["score"] = pd.to_numeric(df_scores["score"], errors="coerce").fillna(0.0)
                    if "application_tag" not in df_scores.columns and "tag" in df_scores.columns:
                        df_scores.rename(columns={"tag": "application_tag"}, inplace=True)
                    df_scores.sort_values("score", ascending=False, inplace=True)

                    top_tag = str(df_scores.iloc[0]["application_tag"]) if not df_scores.empty else ""
                    top_score = float(df_scores.iloc[0]["score"]) if not df_scores.empty else 0.0

                    results_rows.append({
                        "material": mat,
                        "status": "new/sparse",
                        "openalex_hits": hits,
                        "action": "similarity",
                        "top_tag": top_tag,
                        "top_score": round(top_score, 3),
                        "nn1_material": nn1_m, "nn1_sim": nn1_s,
                        "nn2_material": nn2_m, "nn2_sim": nn2_s,
                        "nn3_material": nn3_m, "nn3_sim": nn3_s,
                    })

        progress.progress(1.0)
        status.success("Batch screening complete ✅")

        st.session_state.batch_df_res = pd.DataFrame(results_rows)
        st.session_state.batch_deepverify_long = deepverify_long

    # Always render from stored results (so selectbox changes don't rerun batch)
    df_res = st.session_state.batch_df_res
    deepverify_long = st.session_state.batch_deepverify_long

    if df_res is None or df_res.empty:
        st.info("Run batch screening first.")
        st.stop()

    st.dataframe(df_res, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Batch summary")
    known_count = int((df_res["status"].astype(str) == "known/not-new").sum()) if "status" in df_res.columns else 0
    new_count = int((df_res["status"].astype(str) == "new/sparse").sum()) if "status" in df_res.columns else 0
    sim_rows = df_res[df_res.get("action", pd.Series(dtype=str)).astype(str) == "similarity"].copy() if "action" in df_res.columns else pd.DataFrame()
    avg_nn1 = pd.to_numeric(sim_rows.get("nn1_sim", pd.Series(dtype=float)), errors="coerce").dropna().mean() if not sim_rows.empty else float("nan")

    s1, s2, s3 = st.columns(3)
    s1.metric("Known / not new", known_count)
    s2.metric("New / sparse", new_count)
    s3.metric("Mean top similarity", f"{avg_nn1:.3f}" if pd.notna(avg_nn1) else "—")

    st.download_button(
        "Download batch summary CSV",
        data=df_res.to_csv(index=False).encode("utf-8"),
        file_name="batch_screening_summary.csv",
        mime="text/csv",
    )

    # Inspector (graph + details)
    st.divider()
    st.subheader("Inspect one candidate (kNN graph + details)")

    pick = st.selectbox("Select a candidate", options=df_res["material"].astype(str).tolist(), key="batch_pick")
    row = df_res[df_res["material"].astype(str) == str(pick)].head(1)

    if not row.empty:
        r = row.iloc[0].to_dict()
        st.json(r)

        if r.get("action") == "similarity":
            lib = similarity.load_library(SIM_LIB_FILE)
            if lib is None:
                st.warning("Similarity library missing. Build it first (Deep Verify).")
            else:
                rec = similarity.recommend_by_similarity(
                    str(pick),
                    lib,
                    top_k=int(st.session_state.get("sim_top_k", 10)),
                    alpha=float(st.session_state.get("sim_alpha", 2.0)),
                    min_sim=float(st.session_state.get("sim_min", 0.30)),
                )

                neighbors = rec.get("neighbors") or []
                df_scores2 = rec.get("tag_scores")
                top_tags = []
                if df_scores2 is not None:
                    df_tmp = df_scores2.copy()
                    if "application_tag" not in df_tmp.columns and "tag" in df_tmp.columns:
                        df_tmp.rename(columns={"tag": "application_tag"}, inplace=True)
                    df_tmp["score"] = pd.to_numeric(df_tmp["score"], errors="coerce").fillna(0.0)
                    df_tmp.sort_values("score", ascending=False, inplace=True)
                    top_tags = [(str(rr["application_tag"]), float(rr["score"])) for _, rr in df_tmp.head(3).iterrows()]

                neighbors = _normalize_neighbor_labels(neighbors)
                center_name = format_material_name_generic(str(pick))

                if neighbors:
                    show_material_network_graph(center_name, neighbors[:8], title="kNN Composition Network")
                    st.markdown("**Nearest neighbors table**")
                    st.dataframe(pd.DataFrame(neighbors[:10]), use_container_width=True, hide_index=True)

                if df_scores2 is not None:
                    st.markdown("**Recommended applications (from similarity transfer)**")
                    df_show = df_scores2.copy()
                    if "application_tag" not in df_show.columns and "tag" in df_show.columns:
                        df_show.rename(columns={"tag": "application_tag"}, inplace=True)
                    df_show["score"] = pd.to_numeric(df_show["score"], errors="coerce").fillna(0.0)
                    df_show.sort_values("score", ascending=False, inplace=True)
                    st.dataframe(df_show.head(15), use_container_width=True, hide_index=True)

                    df_edges, app_scores = build_application_contributions(
                        lib,
                        neighbors,
                        alpha=float(st.session_state.get("sim_alpha", 2.0)),
                        top_apps=8,
                    )
                    if not df_edges.empty and not app_scores.empty:
                        show_application_evidence_graph(
                            center_name,
                            neighbors[:8],
                            df_edges,
                            app_scores,
                            title="Composition → Application Evidence Network",
                        )
        else:
            st.caption("This candidate was processed by Deep Verify (literature mode). Open Tab 4 to inspect its evidence-backed tags.")

    if deepverify_long:
        df_long_all = pd.concat(deepverify_long, ignore_index=True)
        st.download_button(
            "Download deep-verify df_long CSV",
            data=df_long_all.to_csv(index=False).encode("utf-8"),
            file_name="batch_deepverify_df_long.csv",
            mime="text/csv",
        )
