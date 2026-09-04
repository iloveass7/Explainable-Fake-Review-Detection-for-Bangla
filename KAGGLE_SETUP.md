# Kaggle Setup Guide for Explainable Fake Review Detection for Bangla

## Quick Start on Kaggle

### 1. Create New Notebook
- Go to kaggle.com → Create → New Notebook
- Select **Python** environment
- **Important**: Enable **Internet access** (needed for Hugging Face downloads)

### 2. Upload Files
Drag all 6 notebooks + `requirements.txt` + `run_all.py` into the notebook sidebar:
- `01_data_acquisition_audit.ipynb`
- `02_preprocessing_eda.ipynb`
- `03_feature_engineering.ipynb`
- `04_gold_only_training.ipynb`
- `05_pseudo_labeling.ipynb`
- `06_final_evaluation.ipynb`
- `requirements.txt`
- `run_all.py`

### 3. Configure Accelerator
- **GPU**: Not needed (Spark runs on CPU)
- **Memory**: Standard (16-30 GB) is sufficient for 100k BanglishRev sample
- **Time**: Set to max (6 hours for full pipeline)

### 4. Run Options

#### Option A: Run Individual Notebooks (Recommended for debugging)
Open each notebook and run cells sequentially. Start with `01_data_acquisition_audit.ipynb`.

#### Option B: Run Full Pipeline
Create a new cell at the top of any notebook and run:
```python
%run run_all.py
```

### 5. Expected Runtime (Kaggle Free Tier)
| Week | Task | Est. Time |
|------|------|-----------|
| 1 | Data download + audit | 5-10 min |
| 2 | Preprocessing + EDA | 10-15 min |
| 3 | TF-IDF + Word2Vec + MinHash | 15-25 min |
| 4 | 5 models with CV | 30-45 min |
| 5 | Pseudo-labeling + retrain | 20-30 min |
| 6 | Test eval + LIME | 15-20 min |
| **Total** | **Full pipeline** | **1.5-2.5 hours** |

### 6. Memory Optimization Tips
If you hit OOM errors:
```python
# In Spark config, reduce:
.config("spark.driver.memory", "6g")
.config("spark.executor.memory", "6g")
# And process BanglishRev in smaller chunks
BANGREV_SAMPLE_SIZE = 50000  # Reduce from 100k
```

### 7. Output Locations
All artifacts saved to `/kaggle/working/bangla_fake_review/`:
```
├── processed/          # Parquet datasets
├── models/             # Fitted Spark ML models
├── results/            # Validation/test predictions, metrics CSVs
├── report/             # Final plots, tables, LIME explanations
└── audit/              # Weekly audit JSONs
```

### 8. Key Files for Submission
- **Final metrics**: `results/test_results_final.csv`
- **Best model**: `models/final_model/`
- **Report figures**: `report/*.png`
- **LIME explanations**: `report/lime_explanations.json`
- **Executive summary**: `report/final_summary.json`

### 9. Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: bnlp` | Run `!pip install bnlp-toolkit` in first cell |
| Hugging Face download timeout | Use `streaming=True` (already in code) |
| Spark OOM | Reduce `BANGREV_SAMPLE_SIZE`, increase driver memory |
| MinHash slow | Reduce `numHashTables` from 5 to 3 |
| LIME timeout | Reduce `num_samples` from 500 to 200 |

### 10. Kaggle-Specific Notes
- **No persistent storage**: All data in `/kaggle/working/` is lost after session. Download `report/` and `models/` before closing.
- **30hr/week GPU limit**: Not applicable (CPU only)
- **Dataset size**: Full BanglishRev (1.74M) may not fit in free tier. Code uses 100k sample by default.
- **Internet required**: For `datasets` library to download from Hugging Face.

### 11. Expected Final Results (from proposal)
| Model | Gold Macro-F1 | Gold+Pseudo Macro-F1 | Δ |
|-------|--------------|---------------------|---|
| LR | ~0.75 | ~0.77 | +0.02 |
| NB | ~0.70 | ~0.71 | +0.01 |
| Linear SVC | ~0.76 | ~0.78 | +0.02 |
| RF | ~0.78 | ~0.79 | +0.01 |
| GBT | ~0.79 | ~0.80 | +0.01 |

*Actual results will vary. The proposal treats pseudo-labeling improvement as experimental finding, not guaranteed.*

---

## For Local/Databricks/Colab
Same notebooks work with minor path changes:
- Change `BASE_DIR = Path("/kaggle/working/bangla_fake_review")` to local path
- Increase `BANGREV_SAMPLE_SIZE` or process full corpus if RAM permits
- Colab: Use `!pip install pyspark` + Java 11 setup