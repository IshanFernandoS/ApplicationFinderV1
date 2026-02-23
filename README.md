# Material -> Applications (MVP) — OPTIMADE + OpenAlex

This is a small, runnable prototype that:
- takes a material query (e.g., `VO2`, `TiO2`, or a plain name like `vanadium dioxide`)
- retrieves a small set of relevant papers via OpenAlex (cached locally)
- scores a handful of application tags using simple keyword evidence
- prints the top suggested applications with supporting paper titles

## Setup

1) Install dependencies:

```bash
pip install -r requirements.txt
```

2) Create a `.env` file:

- Copy `.env.example` to `.env`
- Put your `OPENALEX_API_KEY` into `.env`

```bash
cp .env.example .env
```

## Run

```bash
python material_app_finder.py "VO2"
python material_app_finder.py "TiO2" --top_k 5 --papers_per_query 8
```

JSON output:

```bash
python material_app_finder.py "VO2" --json > vo2_results.json
```

## Notes

- The MVP uses OpenAlex paper metadata (titles and, when available, abstract snippets).
- OPTIMADE is optional; if you set `OPTIMADE_BASE_URL`, the script will try a lightweight structure lookup
  for formula-like inputs to confirm availability at that provider.
- Results are cached in a small SQLite file (default: `appfinder_cache.sqlite3`) so reruns are fast
  and don’t repeatedly call APIs.

## Next upgrades (when you're ready)
- Add stronger material resolution (e.g., map common names -> formulas; or integrate Materials Project mp-api).
- Add semantic retrieval (embeddings) + better reranking.
- Add full-text OA retrieval (Unpaywall) for stronger evidence.
