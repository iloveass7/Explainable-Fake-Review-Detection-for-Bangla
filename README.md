# Explainable Fake Review Detection for Bangla

**Course**: CSE 4262 Data Analytics Lab  
**Institution**: Ahsanullah University of Science and Technology (AUST)  
**Group**: Lab Group 4, Group 3  

A PySpark-based explainable pipeline for detecting fake reviews in Bangla using multi-signal features, near-duplicate analysis, and high-confidence pseudo-labeling.

---

## Problem Statement

Online reviews in Bangla are often code-mixed, informal, and lack labeled data for fake review detection. Existing approaches rely on expensive transformer ensembles. This project builds a **resource-conscious, explainable classical ML pipeline** using PySpark that:

- Combines **TF-IDF, Word2Vec, stylometric, promotional, and near-duplicate (MinHash LSH)** features
- Uses **controlled semi-supervised learning** with pseudo-labels at >90% confidence
- Evaluates on **held-out gold test set** with macro-F1 and PR-AUC (imbalanced data)
- Provides **global + local explanations** (coefficients, feature importance, LIME)

---

## Datasets

| Dataset | Size | Labels | Purpose |
|---------|------|--------|---------|
| **BFRD** (Bengali Fake Review Dataset) | 9,049 reviews | 1,339 fake / 7,710 genuine | Gold standard for train/val/test |
| **BanglishRev** | 1.74M reviews | None (unlabeled) | Auxiliary pool for pseudo-labeling |

Both datasets are publicly available on Hugging Face:
- BFRD: `shawon95/Bengali-Fake-Review-Dataset`
- BanglishRev: `BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset`

---

## Pipeline Overview (6 Weeks)

```
Week 1: Data Acquisition & Audit
  ├── Download from Hugging Face
  ├── Explicit schemas, row counts, null checks
  ├── Exact deduplication
  └── Stratified 70/15/15 split (TEST LOCKED)

Week 2: Preprocessing & EDA
  ├── Bangla tokenization (bnlp-toolkit)
  ├── Handcrafted features: length, code-mix, promo keywords, emoji, URLs, duplicates
  └── Exploratory analysis with Spark SQL + visualizations

Week 3: Feature Engineering
  ├── TF-IDF (50k vocab, minDF=3) → Linear models
  ├── Word2Vec (100-dim) → Tree models
  ├── MinHash LSH (k=3 shingles, 5 hash tables) → Near-duplicate graph features
  └── Model-ready feature matrices

Week 4: Gold-Only Training
  ├── Logistic Regression, Naive Bayes, Linear SVC, Random Forest, GBT
  ├── 3-fold CV with macro-F1 optimization
  ├── Validation evaluation (Macro-F1, PR-AUC primary)
  └── Best model selected for pseudo-labeling

Week 5: Pseudo-Labeling & Ablation
  ├── LR predicts BanglishRev @ confidence > 0.90
  ├── Pseudo-labels weighted 0.5, gold weighted 1.0
  ├── Retrain all 5 models on expanded set
  ├── Ablation: Gold-only vs Gold+Pseudo, feature removal
  └── Sensitivity check @ 0.95 threshold

Week 6: Final Evaluation & Explainability
  ├── **Single evaluation on locked test set**
  ├── Confusion matrices, per-class metrics
  ├── Error analysis (FP/FN by length, code-mix, promo, duplicates)
  ├── Global: LR coefficients, RF/GBT feature importance
  ├── Local: LIME explanations (~15 test cases)
  └── Final report artifacts
```

---

## Quick Start (Kaggle)

### 1. Create Notebook
- Go to [Kaggle](https://kaggle.com) → Create → New Notebook
- Enable **Internet access** (Settings → Internet → On)

### 2. Upload Files
Drag all files from this repo into the Kaggle notebook sidebar.

### 3. Run
```python
# Option A: Individual notebooks (recommended)
# Open 01_data_acquisition_audit.ipynb → Run all cells
# Then 02_preprocessing_eda.ipynb, etc.

# Option B: Full pipeline
%run run_all.py
```

### 4. Expected Runtime
~1.5–2.5 hours on Kaggle free tier (16–30 GB RAM).

---

## Local / Databricks / Colab

```bash
# Install dependencies
pip install -r requirements.txt

# Adjust paths in notebooks:
# BASE_DIR = Path("/path/to/local/dir")  # Instead of /kaggle/working/...

# Run notebooks sequentially
jupyter notebook 01_data_acquisition_audit.ipynb
```

**Colab**: Requires Java 11 + Spark setup. See `KAGGLE_SETUP.md` for details.

---

## Key Outputs

| Artifact | Location |
|----------|----------|
| Final test metrics | `results/test_results_final.csv` |
| Best model (expanded) | `models/final_model/` |
| Best model (gold-only) | `models/final_model_gold_only/` |
| Comparison plots | `report/test_comparison.png`, `report/confusion_matrices.png` |
| Error analysis | `report/error_analysis.png` |
| LR coefficients | `report/lr_coefficients_comparison.csv` |
| Tree feature importance | `report/rf_feature_importance_test.csv`, `report/gbt_feature_importance_test.csv` |
| LIME explanations | `report/lime_explanations.json` |
| Executive summary | `report/final_summary.json` |

---

## Methodology Highlights

### Leakage Prevention
- Test set split **before any feature fitting** (Week 1)
- TF-IDF, Word2Vec, IDF, MinHash, scalers **fitted on train only**
- Pseudo-labels **never enter validation or test**

### Class Imbalance Handling
- Macro-F1 & PR-AUC as primary metrics (not accuracy)
- Instance weights for LR, SVC, RF, GBT
- Naive Bayes uses non-negative features only

### Pseudo-Labeling Controls
- Confidence threshold: **0.90** (configurable)
- Sample weight: **0.5** for pseudo, **1.0** for gold
- Volume cap if pseudo >> gold (ratio > 5:1)
- Sensitivity check at **0.95**

### Explainability
| Level | Method |
|-------|--------|
| Global (Linear) | LR coefficients mapped to vocabulary + engineered features |
| Global (Tree) | RF/GBT feature importance |
| Local | LIME on ~15 test predictions (correct + incorrect) |

---

## Expected Results (from Proposal)

| Model | Gold Macro-F1 | Gold+Pseudo Macro-F1 | Δ |
|-------|--------------|---------------------|---|
| Logistic Regression | ~0.75 | ~0.77 | +0.02 |
| Naive Bayes | ~0.70 | ~0.71 | +0.01 |
| Linear SVC | ~0.76 | ~0.78 | +0.02 |
| Random Forest | ~0.78 | ~0.79 | +0.01 |
| GBT | ~0.79 | ~0.80 | +0.01 |

*Actual results vary. Pseudo-labeling improvement is an experimental finding, not guaranteed.*

---

## Repository Structure

```
.
├── 01_data_acquisition_audit.ipynb    # Week 1
├── 02_preprocessing_eda.ipynb         # Week 2
├── 03_feature_engineering.ipynb       # Week 3
├── 04_gold_only_training.ipynb        # Week 4
├── 05_pseudo_labeling.ipynb           # Week 5
├── 06_final_evaluation.ipynb          # Week 6
├── run_all.py                         # Master runner
├── requirements.txt                   # Dependencies
├── KAGGLE_SETUP.md                    # Kaggle guide
├── G3Gr4.pdf                          # Original proposal
└── .gitignore
```

---

## Team

| Student ID | Name | Primary Responsibility |
|------------|------|------------------------|
| 20220104013 | Syed Abir Hossain | Modeling, pseudo-labeling, evaluation, explainability |
| 20220104016 | Alim Bin Yeasin | Bangla preprocessing, TF-IDF/Word2Vec, MinHash |
| 20220104028 | Shuhrid Abrar | Data acquisition, Spark SQL auditing, EDA |

---

## References

1. Shahariar et al., "Bengali fake reviews: A benchmark dataset and detection system," *Neurocomputing*, 2024.
2. Shamael et al., "BanglishRev: A Large-Scale Bangla-English and Code-mixed Dataset," *arXiv:2412.13161*, 2024.
3. Jindal & Liu, "Opinion spam and analysis," *WSDM*, 2008.
4. Ott et al., "Finding deceptive opinion spam," *ACL-HLT*, 2011.
5. Lee, "Pseudo-label: The simple and efficient semi-supervised learning method," *ICML Workshop*, 2013.
6. Ribeiro et al., "Why should I trust you? Explaining the predictions of any classifier," *KDD*, 2016.

---

## License

Academic project for CSE 4262 Data Analytics Lab. Datasets used under their respective Hugging Face repository licenses.