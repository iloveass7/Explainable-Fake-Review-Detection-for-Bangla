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
# Open notebooks/01-data-acquisition-audit.ipynb → Run all cells
# Then notebooks/02-preprocessing-eda.ipynb, etc.

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
jupyter notebook notebooks/01-data-acquisition-audit.ipynb
```

**Colab**: Requires Java 11 + Spark setup. See [`docs/KAGGLE_SETUP.md`](file:///d:/Explainable%20Fake%20Review%20Detection%20for%20Bangla/docs/KAGGLE_SETUP.md) for details.

---

## Key Outputs

| Artifact | Location | Description |
|----------|----------|-------------|
| Final Test Metrics | [`outputs/results/test_results_final.csv`](file:///d:/Explainable%20Fake%20Review%20Detection%20for%20Bangla/outputs/results/test_results_final.csv) | Final locked-test evaluation metrics across all models |
| Validation Metrics | [`outputs/results/gold_validation_metrics.csv`](file:///d:/Explainable%20Fake%20Review%20Detection%20for%20Bangla/outputs/results/gold_validation_metrics.csv) | 3-fold cross-validation metrics on gold set |
| Pseudo-Label Ablation | [`outputs/results/feature_ablation_validation.csv`](file:///d:/Explainable%20Fake%20Review%20Detection%20for%20Bangla/outputs/results/feature_ablation_validation.csv) | Feature ablation and gold vs expanded comparison |
| Comparison Plots | [`outputs/report/test_comparison.png`](file:///d:/Explainable%20Fake%20Review%20Detection%20for%20Bangla/outputs/report/test_comparison.png), [`outputs/report/confusion_matrices.png`](file:///d:/Explainable%20Fake%20Review%20Detection%20for%20Bangla/outputs/report/confusion_matrices.png) | Model performance & confusion matrix visual comparison |
| Error Analysis | [`outputs/report/error_analysis.png`](file:///d:/Explainable%20Fake%20Review%20Detection%20for%20Bangla/outputs/report/error_analysis.png), [`outputs/report/error_analysis.csv`](file:///d:/Explainable%20Fake%20Review%20Detection%20for%20Bangla/outputs/report/error_analysis.csv) | False positive and false negative error breakdowns |
| LR Coefficients | [`outputs/report/logistic_coefficients.csv`](file:///d:/Explainable%20Fake%20Review%20Detection%20for%20Bangla/outputs/report/logistic_coefficients.csv) | Global linear model feature weights |
| Feature Importance | [`outputs/report/random_forest_feature_importance.csv`](file:///d:/Explainable%20Fake%20Review%20Detection%20for%20Bangla/outputs/report/random_forest_feature_importance.csv), [`outputs/report/gbt_feature_importance.csv`](file:///d:/Explainable%20Fake%20Review%20Detection%20for%20Bangla/outputs/report/gbt_feature_importance.csv) | Tree-based model importance rankings |
| Local Explanations | [`outputs/report/lime_explanations.json`](file:///d:/Explainable%20Fake%20Review%20Detection%20for%20Bangla/outputs/report/lime_explanations.json) | LIME sample-level explanations |
| Decision Support Tables | [`outputs/decision_support/`](file:///d:/Explainable%20Fake%20Review%20Detection%20for%20Bangla/outputs/decision_support) | PySpark SQL decision tables (balance, drift, review queues) |
| Trained ML Models | [`outputs/models/`](file:///d:/Explainable%20Fake%20Review%20Detection%20for%20Bangla/outputs/models) | Serialized Spark ML pipelines and models (~11.5 MB) |
| Execution Logs | [`outputs/logs/`](file:///d:/Explainable%20Fake%20Review%20Detection%20for%20Bangla/outputs/logs) | Full stdout/stderr logs from the Kaggle run |
| IEEE Paper PDF | [`report/bangla_fake_review_ieee.pdf`](file:///d:/Explainable%20Fake%20Review%20Detection%20for%20Bangla/report/bangla_fake_review_ieee.pdf) | Compiled IEEE conference paper |

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
| Local | LIME on ~110 test predictions (correct + incorrect) |

---

## Experimental Results

All experiments were executed on Kaggle (PySpark 3.5.x environment). Test evaluation was conducted exactly once on the locked held-out test split (1,356 reviews: 201 fake, 1,155 genuine).

### 1. Locked Held-Out Test Evaluation

| Model / Condition | Macro-F1 | Accuracy | Fake PR-AUC | Fake ROC-AUC | Fake Precision | Fake Recall | Fake F1 | Genuine F1 | Confusion Matrix (F:G) |
|---|---|---|---|---|---|---|---|---|---|
| **Random Forest (Gold Only)** | **0.9194** | **0.9617** | **0.9213** | **0.9643** | 0.9306 | **0.8010** | **0.8610** | **0.9778** | [[161, 40], [12, 1143]] |
| **Random Forest (Gold + Pseudo)** | 0.9098 | 0.9580 | 0.9033 | 0.9572 | **0.9390** | 0.7662 | 0.8438 | 0.9757 | [[154, 47], [10, 1145]] |

> **Key Finding**: The Gold-Only Random Forest model achieves the highest overall test performance (**0.9194 Macro-F1**, **0.9617 Accuracy**, **0.9213 PR-AUC**). Pseudo-labeling from unconstrained e-commerce reviews showed slight degradation ($\Delta\text{Macro-F1} = -0.0096$), indicating that clean, verified gold labels remain superior to open-domain pseudo-annotations.

### 2. Validation Benchmark Across All Models (3-Fold CV on Gold)

| Model Family | Validation Macro-F1 | Accuracy | Fake PR-AUC | Fake F1 | Genuine F1 |
|---|---|---|---|---|---|
| **Random Forest** | **0.9252** | **0.9646** | **0.9235** | **0.8710** | **0.9795** |
| **Logistic Regression** | 0.9094 | 0.9558 | 0.8899 | 0.8446 | 0.9742 |
| **Gradient Boosted Trees (GBT)** | 0.9074 | 0.9535 | 0.9173 | 0.8421 | 0.9728 |
| **Decision Tree** | 0.8650 | 0.9270 | 0.8616 | 0.7735 | 0.9565 |
| **Linear SVC** | 0.7758 | 0.8990 | 0.6677 | 0.6097 | 0.9420 |
| **Naive Bayes** | 0.7139 | 0.8282 | 0.5397 | 0.5331 | 0.8947 |
| **Factorization Machine** | 0.6673 | 0.8414 | 0.3455 | 0.4267 | 0.9080 |

### 3. Feature Ablation Study

| Configuration | Feature Set | Macro-F1 | Fake PR-AUC | Fake F1 | Impact |
|---|---|---|---|---|---|
| Gold Training | **Full Multi-Signal Fusion** | **0.9094** | **0.8899** | **0.8446** | Baseline |
| Gold Training | **TF-IDF Only** | 0.7383 | 0.5919 | 0.5594 | **-0.1711 Macro-F1 (-17.1%)** |
| Gold Training | **Without Near-Duplicate** | 0.9107 | 0.8901 | 0.8468 | Comparable / within noise margin |
| Expanded Training | Full Multi-Signal Fusion | 0.9078 | 0.8843 | 0.8413 | Baseline |
| Expanded Training | TF-IDF Only | 0.7240 | 0.5938 | 0.5551 | **-0.1837 Macro-F1 (-18.4%)** |

> **Takeaway**: Relying solely on bag-of-words / TF-IDF drops detection Macro-F1 by over **17%**. Handcrafted stylometrics, promotional markers, and dense embeddings are critical for robust Bangla fake review detection.

---

## Repository Structure

```
.
├── notebooks/                              # Final executed Jupyter notebooks with Kaggle outputs
│   ├── 01-data-acquisition-audit.ipynb     # Stage 1: Data acquisition, audit, leakage-safe splits
│   ├── 02-preprocessing-eda.ipynb          # Stage 2: Bangla tokenization, handcrafted signals, EDA
│   ├── 03-feature-engineering.ipynb        # Stage 3: TF-IDF, Word2Vec, MinHash LSH feature matrices
│   ├── 04-gold-only-training.ipynb         # Stage 4: 5 classical models with 3-fold cross-validation
│   ├── 05-pseudo-labeling.ipynb            # Stage 5: Confidence pseudo-labeling & controlled ablation
│   ├── 06-final-evaluation.ipynb           # Stage 6: Locked test set evaluation & explainability (LIME)
│   └── 07-decision-support-sql-pyspark.ipynb # Stage 7: PySpark SQL decision-support evidence
├── outputs/                                # Clean experimental results and artifacts
│   ├── audit/                              # Data provenance, leakage audits, split balances
│   ├── decision_support/                   # Spark SQL analytical decision tables
│   ├── eda/                                # Exploratory data analysis distribution plots
│   ├── logs/                               # Execution logs from Kaggle runs
│   ├── models/                             # Saved PySpark ML models and pipelines (~11.5 MB)
│   ├── report/                             # Confusion matrices, error analyses, LIME explanations
│   └── results/                            # Test metrics, validation results, and prediction tables
├── report/                                 # IEEE Conference Paper
│   ├── bangla_fake_review_ieee.tex         # Complete LaTeX paper source
│   ├── bangla_fake_review_ieee.pdf         # Compiled camera-ready IEEE paper PDF
│   ├── IEEEtran.cls                        # IEEE conference LaTeX class file
│   └── report_assets/                      # Figures, charts, and pipeline methodology diagrams
├── docs/                                   # Extended documentation and project specifications
│   ├── PROJECT_REPORT.md                   # Full IEEE-style project report & findings
│   ├── DATA_ANALYTICS_DECISIONS.md         # Operational data analytics decision guide
│   ├── KAGGLE_SETUP.md                     # Kaggle execution and setup instructions
│   └── G3Gr4.pdf                           # Course project specification guideline PDF
├── requirements.txt                        # Python dependencies
├── run_all.py                              # Master automated pipeline runner
└── README.md
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