#!/usr/bin/env python3
"""
Generate materials.txt using OpenAI (GPT) given a materials-family description.

- Uses OpenAI Responses API + Structured Outputs (JSON Schema).
- Writes one composition per line to the output file.
- Designed for HEC-style families (e.g., equimolar rock-salt high-entropy carbides).

Deps:
  pip install requests python-dotenv

Env (.env supported):
  OPENAI_API_KEY=...
  OPENAI_MODEL=gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY") or "").strip()
OPENAI_MODEL = (os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip()

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment/.env")

# Accepts: (Ta0.2Nb0.2Ti0.2Sc0.2Hf0.2)C
HEC_RE = re.compile(r"^\([A-Z][a-z]?\d*\.?\d*(?:[A-Z][a-z]?\d*\.?\d*)+\)[A-Z][a-z]?$")


SCHEMA = {
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
                    "cations": {"type": "array", "items": {"type": "string"}},
                    "anion": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["composition", "cations", "anion", "notes"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["family_interpretation", "materials"],
    "additionalProperties": False,
}


def call_openai_responses(system: str, user: str) -> Dict[str, Any]:
    """
    Responses API + Structured Outputs (JSON Schema) via text.format.
    """
    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "materials_list",
                "schema": SCHEMA,
                "strict": True,
            }
        },
    }

    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=90,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI HTTP {r.status_code}: {r.text[:800]}")

    data = r.json()

    # Extract output_text
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
        raise RuntimeError("No output_text found in OpenAI response.")

    return json.loads(out_text)


def normalize_and_filter(materials: List[Dict[str, Any]], n: int) -> List[str]:
    """
    - Keep only unique compositions
    - Enforce HEC composition formatting (basic sanity)
    """
    out: List[str] = []
    seen = set()

    for m in materials:
        comp = " ".join((m.get("composition") or "").split())
        if not comp:
            continue

        # Basic sanity check for the HEC string format
        if not HEC_RE.match(comp):
            continue

        key = comp.lower()
        if key in seen:
            continue

        seen.add(key)
        out.append(comp)
        if len(out) >= n:
            break

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate materials.txt from a family description using GPT")
    ap.add_argument("--family", required=True, help="Family description (what kind of materials to generate)")
    ap.add_argument("--element_pool", default="Ta,Nb,Ti,Sc,Hf,Zr,V,Mo,W,Cr",
                    help="Comma-separated allowed cations (default is a refractory TM pool)")
    ap.add_argument("--k", type=int, default=5, help="Number of cations in the HEC (default: 5)")
    ap.add_argument("--anion", default="C", help="Anion (default: C)")
    ap.add_argument("--n", type=int, default=10, help="How many compositions to output (default: 10)")
    ap.add_argument("--out", default="materials.txt", help="Output file (default: materials.txt)")
    args = ap.parse_args()

    pool = [x.strip() for x in args.element_pool.split(",") if x.strip()]
    if len(pool) < args.k:
        raise ValueError(f"element_pool has {len(pool)} elements but k={args.k}")

    system = (
        "You are a materials scientist. Generate candidate compositions following the user's family constraints. "
        "You MUST obey the allowed element pool and formatting rules."
    )

    user = f"""
Family (high level): {args.family}

Hard constraints:
- Target: high-entropy carbides in rock-salt (NaCl) structure.
- Use ONLY these cations (no extra elements): {", ".join(pool)}
- Exactly k = {args.k} distinct cations per composition.
- Anion must be: {args.anion}
- Use equimolar fractions (each cation fraction = 1/k) written as decimals (e.g., 0.2 for k=5).
- Output format MUST be exactly like: (Ta0.2Nb0.2Ti0.2Sc0.2Hf0.2){args.anion}
- Produce at least {args.n} candidates (more is okay; we'll take the first {args.n} unique valid ones).
- Avoid duplicates and avoid reordering-only duplicates (treat different order as same composition).

Return JSON that matches the schema.
""".strip()

    plan = call_openai_responses(system, user)
    comps = normalize_and_filter(plan.get("materials", []), n=args.n)

    if len(comps) < args.n:
        # One more attempt, slightly stronger instruction
        user2 = user + "\n\nSecond attempt: Ensure all compositions strictly match the required format and use only allowed elements."
        plan2 = call_openai_responses(system, user2)
        comps2 = normalize_and_filter(plan2.get("materials", []), n=args.n)
        comps = comps2 if len(comps2) > len(comps) else comps

    if not comps:
        raise RuntimeError("No valid compositions generated. Try expanding element_pool or relaxing family text.")

    with open(args.out, "w", encoding="utf-8") as f:
        for c in comps:
            f.write(c + "\n")

    print(f"Wrote {len(comps)} compositions to {args.out}")
    print("Preview:")
    for c in comps[: min(5, len(comps))]:
        print("  ", c)


if __name__ == "__main__":
    main()