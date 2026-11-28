"""FrameLens prediction interface contract (Phase 1 mock scaffolding).

All application calls go through ``predict_article(text)`` in this module so
the Streamlit UI and ML pipeline stay decoupled. In Phase 4, the mock logic
here will be replaced by loading ``models/best_model.joblib``.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "best_model.joblib"

# Illustrative cue words used ONLY for deterministic mock responses during UI
# scaffolding before models/best_model.joblib is trained and wired in.
_MOCK_BIASED_CUES = {
    "disgraceful": 0.62,
    "outrageous": 0.58,
    "reckless": 0.54,
    "radical": 0.49,
    "failed": 0.45,
    "so-called": 0.44,
    "slammed": 0.41,
    "blasted": 0.40,
    "disaster": 0.48,
    "corrupt": 0.56,
    "extreme": 0.43,
    "yet another": 0.38,
}

_MOCK_NEUTRAL_CUES = {
    "reported": -0.35,
    "announced": -0.32,
    "committee": -0.28,
    "tuesday": -0.25,
    "data": -0.30,
    "according": -0.33,
    "statement": -0.27,
    "proposal": -0.24,
}


def split_sentences(text: str) -> list[str]:
    """Split raw article text into clean, non-empty original sentences."""
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [sentence.strip() for sentence in parts if sentence.strip()]


def _mock_score_sentence(sentence: str) -> dict[str, Any]:
    """Produce a deterministic mock prediction following the contract schema."""
    lowered = sentence.lower()
    features: list[dict[str, Any]] = []
    score = -0.15

    for term, weight in _MOCK_BIASED_CUES.items():
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            score += weight
            features.append({"term": term, "weight": weight, "direction": "biased"})

    for term, weight in _MOCK_NEUTRAL_CUES.items():
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            score += weight
            features.append({"term": term, "weight": weight, "direction": "non-biased"})

    if not features:
        tokens = re.findall(r"\b[a-z]{4,}\b", lowered)
        if tokens:
            features.append({"term": tokens[0], "weight": -0.05, "direction": "non-biased"})

    is_biased = score > 0.0
    label = 1 if is_biased else 0
    label_name = "Biased" if is_biased else "Non-biased"
    # Map mock score smoothly into [0.52, 0.95] so UI confidence meters work cleanly.
    confidence = round(min(0.95, 0.55 + min(abs(score), 1.0) * 0.40), 3)

    features.sort(key=lambda item: abs(item["weight"]), reverse=True)
    return {
        "label": label,
        "label_name": label_name,
        "confidence": confidence,
        "influential_features": features[:5],
    }


def predict_article(text: str) -> dict[str, Any]:
    """Analyze raw news article text sentence by sentence.

    Parameters
    ----------
    text:
        Raw news article text string.

    Returns
    -------
    dict
        Contract dictionary containing:
        - ``is_mock``: Whether predictions come from the Phase 1 mock stub.
        - ``model_path``: Expected path to ``models/best_model.joblib``.
        - ``sentences``: List of per-sentence dicts with ``index``, ``text``,
          ``label`` (0 or 1), ``label_name``, ``confidence``, and
          ``influential_features``.
        - ``summary``: Article-level counts and share of flagged sentences.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "is_mock": True,
            "model_path": str(DEFAULT_MODEL_PATH),
            "sentences": [],
            "summary": {
                "total_sentences": 0,
                "biased_count": 0,
                "non_biased_count": 0,
                "biased_share": 0.0,
            },
        }

    raw_sentences = split_sentences(text)
    sentence_results: list[dict[str, Any]] = []
    biased_count = 0

    for idx, sentence in enumerate(raw_sentences):
        pred = _mock_score_sentence(sentence)
        if pred["label"] == 1:
            biased_count += 1
        sentence_results.append({
            "index": idx,
            "text": sentence,
            **pred,
        })

    total = len(sentence_results)
    non_biased_count = total - biased_count
    biased_share = round(biased_count / total, 4) if total else 0.0

    return {
        "is_mock": True,
        "model_path": str(DEFAULT_MODEL_PATH),
        "sentences": sentence_results,
        "summary": {
            "total_sentences": total,
            "biased_count": biased_count,
            "non_biased_count": non_biased_count,
            "biased_share": biased_share,
        },
    }
