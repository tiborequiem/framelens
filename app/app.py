"""FrameLens Streamlit UI skeleton (Phase 1).

Uses ``predict_article()`` from ``src/predict.py`` (currently mock output)
so the interface can be reviewed and iterated in parallel with ML modeling.

Run from the repository root:
    streamlit run app/app.py
"""
from __future__ import annotations

import html
from pathlib import Path
import re
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import predict_article  # noqa: E402

SAMPLE_ARTICLE = (
    "The committee met on Tuesday to discuss the new budget proposal. "
    "Opponents slammed the plan as yet another reckless and disgraceful decision. "
    "According to the official statement, revised figures will be published next week. "
    "Critics warned that the outrageous policy could become a costly disaster."
)


def _highlight_terms(sentence: str, features: list[dict]) -> str:
    """Return HTML with influential terms subtly highlighted for display."""
    term_map = {
        feat["term"].lower(): feat.get("direction", "non-biased")
        for feat in features
        if feat.get("term")
    }
    if not term_map:
        return html.escape(sentence)

    # Match longer phrases before single words in a single pass on plain text.
    ordered_terms = sorted(term_map, key=len, reverse=True)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in ordered_terms) + r")\b",
        re.IGNORECASE,
    )

    chunks: list[str] = []
    last_end = 0
    for match in pattern.finditer(sentence):
        chunks.append(html.escape(sentence[last_end : match.start()]))
        direction = term_map.get(match.group(0).lower(), "non-biased")
        bg = "#ffe3e3" if direction == "biased" else "#e7f5ff"
        fg = "#a51111" if direction == "biased" else "#0b4f8c"
        chunks.append(
            f'<mark style="background-color:{bg};color:{fg};'
            f'padding:0 3px;border-radius:3px;">{html.escape(match.group(0))}</mark>'
        )
        last_end = match.end()
    chunks.append(html.escape(sentence[last_end:]))
    return "".join(chunks)


def main() -> None:
    st.set_page_config(page_title="FrameLens — Bias Detection", layout="wide")
    st.title("FrameLens — News Language Bias Analyzer")
    st.caption(
        "FrameLens detects language patterns associated with bias at the sentence level. "
        "It does not establish factual truth, publisher trustworthiness, or political ideology."
    )

    if "article_input" not in st.session_state:
        st.session_state["article_input"] = SAMPLE_ARTICLE

    col_load, col_clear, _ = st.columns([1, 1, 4])
    if col_load.button("Load Sample Article"):
        st.session_state["article_input"] = SAMPLE_ARTICLE
    if col_clear.button("Clear"):
        st.session_state["article_input"] = ""

    article_text = st.text_area(
        "Paste news article text below:",
        key="article_input",
        height=180,
        placeholder="Paste one or more sentences from a news article...",
    )

    if st.button("Analyze Article", type="primary"):
        result = predict_article(article_text)

        if result.get("is_mock"):
            st.info(
                "UI Skeleton Mode: displaying **mock** `predict_article()` output. "
                "Real predictions will appear once `models/best_model.joblib` is wired into `src/predict.py`."
            )

        summary = result["summary"]
        if summary["total_sentences"] == 0:
            st.warning("Please enter at least one sentence to analyze.")
            return

        st.subheader("Article-Level Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Sentences", summary["total_sentences"])
        m2.metric("Flagged (Biased)", summary["biased_count"])
        m3.metric("Non-biased", summary["non_biased_count"])
        m4.metric("Biased Share", f"{summary['biased_share'] * 100:.1f}%")

        st.subheader("Sentence-by-Sentence Breakdown")
        for item in result["sentences"]:
            badge = "🔴 Biased" if item["label"] == 1 else "🔵 Non-biased"
            conf = item.get("confidence")
            conf_text = f"Confidence: {conf:.1%}" if conf is not None else "Confidence: N/A"

            with st.container(border=True):
                top_left, top_right = st.columns([3, 1])
                top_left.markdown(f"**Sentence {item['index'] + 1}** — {badge}")
                top_right.markdown(f"`{conf_text}`")

                st.markdown(
                    _highlight_terms(item["text"], item.get("influential_features", [])),
                    unsafe_allow_html=True,
                )

                features = item.get("influential_features", [])
                if features:
                    feat_str = ", ".join(
                        f"`{f['term']}` ({f['weight']:+.2f})" for f in features
                    )
                    st.caption(f"Influential cues: {feat_str}")


if __name__ == "__main__":
    main()
