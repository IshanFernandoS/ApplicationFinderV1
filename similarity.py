"""Similarity-based application recommendation for new materials.

Build a library from df_long (material × application_tag × score), featurize compositions with
matminer/pymatgen, and recommend tags for new materials via cosine similarity.

Optional dependency: if pymatgen/matminer/scikit-learn aren't installed, similarity mode is disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pickle
import re

import numpy as np
import pandas as pd

try:
    from pymatgen.core import Composition  # type: ignore
    _HAVE_PYMATGEN = True
except Exception:
    Composition = None  # type: ignore
    _HAVE_PYMATGEN = False

try:
    from sklearn.preprocessing import StandardScaler  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    _HAVE_SK = True
except Exception:
    StandardScaler = None  # type: ignore
    cosine_similarity = None  # type: ignore
    _HAVE_SK = False

try:
    from matminer.featurizers.composition import (  # type: ignore
        ElementProperty,
        Stoichiometry,
        ValenceOrbital,
        IonProperty,
        Miedema,
    )
    _HAVE_MATM = True
except Exception:
    ElementProperty = Stoichiometry = ValenceOrbital = IonProperty = Miedema = None  # type: ignore
    _HAVE_MATM = False


ELEM_RE = re.compile(r"[A-Z][a-z]?")
HEC_PAREN_RE = re.compile(r"^\s*\((.*?)\)\s*([A-Z][a-z]?)\s*$")


@dataclass
class SimilarityLibrary:
    materials: List[str]
    X: np.ndarray
    tag_names: List[str]
    tag_desc: Dict[str, str]
    tag_matrix: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray


def deps_ok() -> Tuple[bool, str]:
    if not _HAVE_PYMATGEN:
        return False, "Missing dependency: pymatgen"
    if not _HAVE_MATM:
        return False, "Missing dependency: matminer"
    if not _HAVE_SK:
        return False, "Missing dependency: scikit-learn"
    return True, ""


def _composition_from_material_string(s: str) -> Optional["Composition"]:
    if not _HAVE_PYMATGEN:
        return None
    s = (s or "").strip()
    if not s:
        return None

    m = HEC_PAREN_RE.match(s)
    if m:
        inside, anion = m.group(1), m.group(2)
        cations = ELEM_RE.findall(inside)
        seen = set()
        cats = []
        for e in cations:
            if e not in seen:
                seen.add(e)
                cats.append(e)
        if not cats:
            return None
        frac = 1.0 / float(len(cats))
        comp_dict = {c: frac for c in cats}
        comp_dict[anion] = 1.0
        return Composition(comp_dict)

    try:
        return Composition(s)
    except Exception:
        elems = ELEM_RE.findall(s)
        if not elems:
            return None
        seen = set()
        uniq = []
        for e in elems:
            if e not in seen:
                seen.add(e)
                uniq.append(e)
        if not uniq:
            return None
        frac = 1.0 / float(len(uniq))
        return Composition({e: frac for e in uniq})


def featurize_material(s: str) -> Optional[np.ndarray]:
    ok, msg = deps_ok()
    if not ok:
        raise ImportError(msg)

    comp = _composition_from_material_string(s)
    if comp is None:
        return None

    featurizers = [
        Stoichiometry(),
        ElementProperty.from_preset("magpie"),
        ValenceOrbital(),
        IonProperty(),
        Miedema(),
    ]

    feats: List[float] = []
    for f in featurizers:
        try:
            vals = f.featurize(comp)
            feats.extend([float(x) if x is not None else float("nan") for x in vals])
        except Exception:
            try:
                n = len(f.feature_labels())
            except Exception:
                n = 0
            feats.extend([float("nan")] * n)

    return np.array(feats, dtype=float)


def _pivot_df_long(df_long: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = df_long.copy()
    df["material"] = df["material"].astype(str)
    df["application_tag"] = df["application_tag"].astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)

    desc: Dict[str, str] = {}
    if "description" in df.columns:
        for tag, g in df.groupby("application_tag"):
            for x in g["description"].dropna().astype(str).tolist():
                if x.strip():
                    desc[str(tag)] = x.strip()
                    break

    mat = df.pivot_table(
        index="material",
        columns="application_tag",
        values="score",
        aggfunc="max",
        fill_value=0.0,
    )
    return mat, desc


def build_library_from_df_long(df_long: pd.DataFrame) -> Optional[SimilarityLibrary]:
    ok, msg = deps_ok()
    if not ok:
        raise ImportError(msg)
    if df_long is None or df_long.empty:
        return None

    tag_mat, tag_desc = _pivot_df_long(df_long)

    X_list: List[np.ndarray] = []
    kept: List[str] = []
    for m in tag_mat.index.tolist():
        v = featurize_material(m)
        if v is None:
            continue
        X_list.append(v)
        kept.append(m)

    if not X_list:
        return None

    X = np.vstack(X_list)

    col_means = np.nanmean(X, axis=0)
    # If a column is all-NaN, nanmean returns NaN; replace those with 0.0
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    inds = np.where(np.isnan(X))
    if inds[0].size:
        X[inds] = np.take(col_means, inds[1])

    # Safety: remove inf/-inf if any featurizer produced them
    X = np.where(np.isfinite(X), X, 0.0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)

    tag_mat2 = tag_mat.loc[kept]

    return SimilarityLibrary(
        materials=kept,
        X=Xs,
        tag_names=tag_mat2.columns.tolist(),
        tag_desc=tag_desc,
        tag_matrix=tag_mat2.values.astype(np.float32),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
    )


def save_library(lib: SimilarityLibrary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(lib, f)


def load_library(path: Path) -> Optional[SimilarityLibrary]:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def recommend_by_similarity(
    new_material: str,
    lib: SimilarityLibrary,
    top_k: int = 15,
    alpha: float = 2.0,
    min_sim: float = 0.70,
) -> Dict[str, Any]:
    ok, msg = deps_ok()
    if not ok:
        raise ImportError(msg)

    vec = featurize_material(new_material)
    if vec is None:
        return {"error": "Could not parse/featurize input as a chemical formula/composition."}

    vec = vec.astype(float)
    vec = np.where(np.isnan(vec), lib.scaler_mean, vec)
    vec = np.where(np.isfinite(vec), vec, lib.scaler_mean)

    scale = np.where(lib.scaler_scale == 0, 1.0, lib.scaler_scale)
    vecs = ((vec - lib.scaler_mean) / scale)
    # Safety: ensure no NaN/inf reaches sklearn
    vecs = np.where(np.isfinite(vecs), vecs, 0.0).astype(np.float32)

    Xlib = np.where(np.isfinite(lib.X), lib.X, 0.0).astype(np.float32)

    sims = cosine_similarity(vecs.reshape(1, -1), Xlib).ravel()
    order = np.argsort(-sims)

    neighbors: List[Dict[str, Any]] = []
    idxs: List[int] = []
    weights: List[float] = []

    for i in order[: max(50, top_k * 3)]:
        sim = float(sims[i])
        if sim < float(min_sim):
            continue
        neighbors.append({"material": lib.materials[i], "sim": round(sim, 4)})
        idxs.append(int(i))
        weights.append(sim ** float(alpha))
        if len(idxs) >= int(top_k):
            break

    if not idxs:
        fallback_k = max(1, int(top_k))
        for i in order[:fallback_k]:
            sim = float(sims[i])
            neighbors.append({"material": lib.materials[i], "sim": round(sim, 4)})
            idxs.append(int(i))
            weights.append(max(sim, 0.0) ** float(alpha))
        if not idxs:
            return {"error": "No featurized neighbors found in the similarity library."}

    W = np.array(weights, dtype=np.float32)
    W = W / (W.sum() + 1e-9)

    scores = (W.reshape(-1, 1) * lib.tag_matrix[idxs]).sum(axis=0)
    df = pd.DataFrame({"application_tag": lib.tag_names, "score": scores.astype(float)})
    df.sort_values("score", ascending=False, inplace=True)
    df["description"] = df["application_tag"].map(lambda t: lib.tag_desc.get(str(t), ""))

    return {
        "neighbors": neighbors,
        "tag_scores": df,
        "confidence": {
            "top1_sim": neighbors[0]["sim"],
            "n_neighbors": len(neighbors),
            "min_sim": float(min_sim),
            "alpha": float(alpha),
        },
    }
