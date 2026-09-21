# Data Analytics Decisions From the Bangla Fake Review Detection Experiment

**Project team:** Syed Abir Hossain; Alim Bin Yeasin; Shuhrid Abrar

## Executive decision summary

The experiment supports a practical moderation and analytics strategy:

1. Use the multi-signal Random Forest as the primary review-risk ranking model.
2. Keep TF–IDF, Word2Vec, stylometric, promotional, script, and duplicate-related signals together.
3. Do not adopt the current pseudo-labeling configuration as a default production training strategy.
4. Use the model to prioritize human investigation, not to automatically punish sellers or remove reviews.
5. Treat short, promotional reviews as a high-priority error-analysis and data-collection segment.
6. Keep duplicate-aware splitting, locked testing, audit logs, and explanation generation as permanent governance controls.

## Evidence from the final experiment

The final run evaluated 1,356 locked-test reviews containing 201 fake reviews (label 0) and 1,155 genuine reviews (label 1). The selected model was a Random Forest. This label convention is used throughout the analysis.

| Evidence | Gold-only | Gold plus pseudo-labels | Decision implication |
|---|---:|---:|---|
| Macro-F1 | 0.9194 | 0.9098 | Gold-only is the safer default |
| Accuracy | 0.9617 | 0.9580 | Pseudo-labeling does not improve overall accuracy |
| Fake-review PR-AUC | 0.9213 | 0.9033 | Gold-only ranks fake-review risk better |
| Fake-review recall | 0.8010 | 0.7662 | Pseudo-labeling misses more fake reviews |
| Genuine-review recall | 0.9896 | 0.9913 | Pseudo-labeling is slightly more conservative toward genuine reviews |
| Validation model selection | Random Forest | Random Forest | Retain Random Forest as the primary model |

The final confusion matrices were:

- Gold-only: `[[161, 40], [12, 1143]]`
- Gold plus pseudo-labels: `[[154, 47], [10, 1145]]`

Rows are actual labels and columns are predicted labels, with label 0 = fake and label 1 = genuine. Therefore, the gold-only model correctly identifies 161 of 201 fake reviews and 1,143 of 1,155 genuine reviews. The expanded model identifies 154 fake and 1,145 genuine reviews.

The pseudo-label configuration produced 7 additional fake-to-genuine errors (missed fake reviews) and 2 fewer genuine-to-fake errors (false alarms) in the locked test comparison. That trade-off is unfavorable when the goal is to identify fake reviews reliably.

## Decisions already made by the project

### 1. Select Random Forest for the final system

The validation comparison and final selection artifact identify Random Forest as the selected model. It provides strong macro-F1, fake-review PR-AUC, and a useful global feature-importance view.

### 2. Use multi-signal feature fusion

The TF–IDF-only ablation performed much worse than the full feature configuration:

| Feature configuration | Macro-F1 | Fake PR-AUC | Fake F1 |
|---|---:|---:|---:|
| Full gold configuration | 0.9094 | 0.8899 | 0.8446 |
| TF–IDF only | 0.7383 | 0.5919 | 0.5594 |

This supports the decision to combine lexical, embedding, stylometric, promotional, script, and duplicate signals instead of relying on text frequency alone.

### 3. Do not use pseudo-labeling by default

The primary pseudo-labeling branch accepted 31,605 examples after confidence and class caps, with a sample weight of 0.50. Despite this control, locked-test performance decreased. The current conclusion is not that pseudo-labeling is always harmful; it is that this configuration is not sufficiently reliable for automatic adoption.

### 4. Keep duplicate-aware data splitting

Exact duplicates and near-duplicates can make performance look artificially high if similar reviews appear in both training and test data. Duplicate-aware splitting and MinHash audits should remain part of the data-quality process, even though removing near-duplicate features changed validation performance very little.

### 5. Use macro metrics and fake-review PR-AUC

The classes are imbalanced. Accuracy alone can hide poor fake-review detection. The project therefore uses macro-F1, fake-review precision, fake-review recall, fake-review F1, PR-AUC, and ROC-AUC together.

### 6. Require explainability for moderation use

The project produces Random Forest feature importance, Logistic Regression coefficients, error analysis, and 110 LIME cases. These outputs support analyst review and make model behavior easier to audit.

## Decisions that can be made from the analytics system

### 1. Prioritize reviews for human investigation

The model can rank reviews by fake probability. A platform can use the ranking to create investigation queues:

- high-risk reviews: immediate analyst review;
- medium-risk reviews: sampled or delayed review;
- low-risk reviews: normal monitoring;
- uncertain cases: route for additional evidence rather than automatic action.

This is safer than using a single probability cutoff as an automatic removal rule.

### 2. Prioritize seller or product monitoring

Review-level predictions can be aggregated by seller, product, campaign, or time period. Useful analytics include:

- percentage of reviews above a selected risk threshold;
- sudden changes in fake-risk rate;
- repeated promotional phrases across reviews;
- unusually concentrated review timing;
- duplicate or near-duplicate review clusters;
- risk differences between products or sellers.

These aggregates can identify where analysts should investigate first. They should not be interpreted as proof of seller misconduct without additional evidence.

### 3. Focus data collection on difficult cases

The error analysis shows that false negatives are shorter and more promotional than correct cases. False positives are also strongly associated with promotional language. Therefore, additional annotation should intentionally include:

- short promotional reviews;
- legitimate discount or campaign reviews;
- mixed Bangla-English reviews;
- emoji-heavy reviews;
- reviews with informal spelling;
- reviews from new sellers and new product categories;
- examples from periods with unusual review-volume spikes.

This is an active-learning decision: label the cases where the current model is most uncertain or most likely to fail.

### 4. Create a human-review policy for false positives

False positives can damage legitimate sellers and customers. The system should define:

- an appeal process;
- analyst-visible explanations;
- a second-review requirement for high-impact actions;
- an audit trail of model version, features, probability, and final human decision;
- periodic sampling of low-risk reviews to estimate false negatives.

### 5. Monitor model drift

The feature pipeline can be used for ongoing monitoring. Track changes in:

- review length and token count;
- Bangla/Banglish code-mix ratio;
- promotional-keyword frequency;
- emoji and punctuation frequency;
- duplicate-cluster sizes;
- predicted fake-risk distribution;
- class proportions among human-reviewed cases;
- precision and recall after new labels become available.

Large changes may indicate a new campaign, a platform policy change, a product-domain shift, or model degradation.

### 6. Decide when retraining is necessary

Retraining should be triggered by evidence, not by a fixed calendar alone. Candidate triggers include:

- a sustained fall in fake-review recall;
- a rise in false positives among appealed cases;
- a substantial change in feature distributions;
- new categories or languages entering the platform;
- drift in duplicate and promotional patterns;
- a new labeled dataset becoming available.

The next training set should include reviewed false positives and false negatives, not only randomly sampled new reviews.

### 7. Decide whether new unlabeled data are useful

Unlabeled BanglishRev data are useful for vocabulary coverage, exploratory analysis, and future representation learning. They should not automatically be converted into training labels. Before reusing pseudo-labeling, compare:

- confidence thresholds;
- calibration quality;
- class balance after capping;
- domain similarity between gold and unlabeled data;
- performance on a new human-labeled holdout;
- performance by review length, script mixture, and product category.

## Decisions that should not be made from this experiment alone

The current results do not justify:

- automatically deleting a review;
- permanently banning a seller;
- declaring a review fraudulent solely from model probability;
- claiming that emojis or promotional words cause fake reviews;
- claiming that near-duplicate features caused the high score;
- claiming universal performance on all Bangla or Banglish reviews;
- claiming that pseudo-labeling is universally ineffective;
- comparing this model with external systems without matching datasets and protocols.

## Recommended operational decision framework

| Decision stage | Analytics input | Recommended action |
|---|---|---|
| Screening | RF fake probability | Rank reviews for triage |
| Evidence gathering | LIME terms, feature signals, duplicate cluster | Ask an analyst to inspect supporting evidence |
| Entity monitoring | Seller/product/time aggregates | Investigate unusual concentrations or spikes |
| Enforcement | Human decision plus policy evidence | Apply proportional action with appeal |
| Learning | Confirmed analyst labels | Add difficult cases to the next training cycle |
| Governance | Metrics, drift, audit logs | Review model health and fairness periodically |

## Evidence generated by the final PySpark SQL run

The executed notebook `Final Run/07-decision-support-sql-pyspark.ipynb` produced the following saved outputs in `Final Output/decision_support_outputs/`:

| Output | Observed result | Decision supported |
|---|---|---|
| `class_balance` | Train 6,321; validation 1,356; test 1,356. Test: 201 fake and 1,155 genuine | Use class-aware metrics and weights |
| `test_decision_comparison` | Gold-only: macro-F1 0.9194, fake PR-AUC 0.9213, fake recall 0.8010; expanded: 0.9098, 0.9033, 0.7662 | Keep gold-only as the default model condition |
| `human_review_queue` | All 1,356 test predictions ranked by normalized fake-risk score derived from Random Forest raw class votes | Prioritize human investigation |
| `error_profile` | Correct 1,299; false negatives 47; false positives 10. False negatives average 450.43 characters and 0.7234 promotional hits | Collect more short, promotional examples |
| `duplicate_exposure` | 4 exact-duplicate test rows; mean exact cluster size 1.0029 | Retain duplicate auditing and leakage controls |
| `drift_indicators` | Train/validation/test means were close: character length 659.18/655.00/667.47; token count 99.04/98.43/100.51; code-mix 0.1432/0.1469/0.1441 | No large split-level shift is indicated; continue monitoring |

These tables support operational prioritization and data-collection decisions. They do not establish causality, seller guilt, or a basis for automatic removal.

## Final recommendation

The project should be presented as an explainable decision-support system for fake-review investigation. The evidence supports Random Forest with fused features, locked-test evaluation, duplicate-aware controls, and human review. The current pseudo-labeling branch should remain an experimental option rather than a production default. The highest-value next step is to build a reviewed feedback loop focused on short promotional reviews, false positives, new domains, and calibrated risk thresholds.

## Reproducible PySpark evidence notebook

The accompanying Kaggle notebook, [07-decision-support-sql-pyspark.ipynb](Final%20Run/07-decision-support-sql-pyspark.ipynb), runs the decision-support queries with PySpark SQL and writes presentation-ready CSV outputs under `/kaggle/working/decision_support_outputs`.

It generates:

- class-balance evidence for train, validation, and test;
- model-selection and gold-only versus pseudo-label test comparisons;
- a human-review risk queue;
- error profiles for correct, false-negative, and false-positive cases;
- duplicate-exposure and leakage-control summaries; and
- train/validation/test feature-monitoring statistics.

These outputs make the report claims reproducible, but they should still be described as analytics evidence and operational support rather than proof of causality or guilt.
