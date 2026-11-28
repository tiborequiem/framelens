# FrameLens: Explainable Machine Learning for News Media Bias Detection

> **Status:** Report skeleton (14-section structure from *FrameLens Project Execution Plan*).
> Fill each section strictly with verified numbers, artifacts, and citations—never
> fabricate results or present planned work as completed.

---

## 1. Introduction
- Context and motivation for detecting biased language in news reporting.
- Scope statement: FrameLens identifies **language patterns associated with bias**; it does not determine factual truth, publisher trustworthiness, author intent, or political ideology.

## 2. Problem Definition
- Task formulation: binary sentence-level classification (`0 = Non-biased`, `1 = Biased`) extended to article-level aggregation and feature-level explanation.
- Research question: *Can traditional machine learning algorithms identify potentially biased language in news reporting using expert-annotated data, and which algorithm performs best?*

## 3. Background and Related Work
- Prior work on media bias detection and the BABE (Bias Annotations By Experts) benchmark (Spinde et al., EMNLP 2021).
- Brief note on why traditional linear/probabilistic/tree-based ML with TF-IDF provides strong interpretability baselines.

## 4. Dataset Description
- Primary dataset: BABE SG2 (`data/final_labels_SG2.csv`).
- Label schema, column inventory, expert annotators, and word-level bias span annotations.
- Clear distinction between language-bias labels and news outlet identity (outlet identity is not treated as a bias label).

## 5. Data Exploration and Preprocessing
- Cleaning steps applied in `Clean.ipynb` (non-binary label filtering, missing `type` handling, HTML/whitespace normalization).
- Dual-text design: `original_text` (preserved for display and explainability) vs. `model_text` (normalized for modeling).
- Class distribution, sentence length statistics, and article-grouped (`news_link`) 80/20 stratified train/test split (`data/train.csv`: 2,938 rows; `data/test.csv`: 735 rows; zero `news_link` overlap).

## 6. Feature Extraction
- TF-IDF representation (`ngram_range`, `min_df`, `max_df`, `max_features`, `sublinear_tf`).
- Strict pipeline isolation: TF-IDF is fitted exclusively on training folds—never on validation folds or the held-out test set.

## 7. Machine Learning Algorithms
- Majority-class benchmark (`DummyClassifier`).
- Four core classifiers compared:
  1. Multinomial Naive Bayes
  2. Logistic Regression
  3. Linear SVM
  4. Random Forest

## 8. Experimental Setup
- 5-fold `StratifiedGroupKFold` (grouped by `news_link`, `random_state=42`) on `data/train.csv`.
- Hyperparameter tuning protocol (`GridSearchCV` / controlled search) optimizing **Macro F1**.
- Final single-pass evaluation on held-out `data/test.csv`, plus hardware/software environment versions.

## 9. Results and Model Comparison
- Preliminary CV baseline table (from `reports/baseline_results.csv`) and final held-out test comparison table (`reports/results.csv`).
- Precision, Recall, and F1 for **both** classes (`Non-biased` and `Biased`), Macro F1, Accuracy, and confusion matrices.

## 10. Error Analysis
- Inspection of false positives, false negatives, and ambiguous sentences from the held-out test set.
- Comparison of model-highlighted terms against BABE's expert word-level annotations.

## 11. Explainability
- Global top positive/negative feature weights from linear models.
- Sentence-level influential word highlighting integrated via `src/predict.py` and the Streamlit dashboard (`app/app.py`).
- Explicit note: feature coefficients reflect statistical association in training data, not proof that a word is inherently biased.

## 12. Limitations and Ethical Considerations
- Bag-of-words / TF-IDF blind spots (sarcasm, multi-sentence context, subtle framing).
- Bias vs. sentiment distinction; risks of over-relying on automated bias scores.

## 13. Conclusion
- Summary of empirical findings, best-performing model (`models/best_model.joblib`), and future directions.

## 14. References
- Spinde et al. (2021) — BABE dataset & EMNLP paper.
- Scikit-learn, Pandas, Joblib, Streamlit documentation and relevant literature.
