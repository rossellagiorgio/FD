# Fraud Detection — Time-Aware Modeling Under Concept Drift

Predicting whether a transaction is fraudulent at the time of authorization, using a dataset explicitly designed to simulate **temporal concept drift**: fraud patterns shift from merchant/IP-driven signals (Jan–Jun) to velocity-based, low-amount fraud (Jul–Dec).

## Approach

- Confirmed the documented concept drift empirically (Mann-Whitney U test on
  fraud-case feature means, fraud-rate-by-bin comparison pre/post month 6).
- Built features while explicitly excluding post-authorization information (`post_auth_risk_score`) to avoid
  leakage.
- Used a **strict chronological split** (no random K-Fold) for both
  validation and hyperparameter tuning (`TimeSeriesSplit`), consistent with
  the dataset's validation guidance.
- Compared Logistic Regression, baseline LightGBM, and a PR-AUC-tuned
  LightGBM (`RandomizedSearchCV`, `scoring="average_precision"`), given the
  severe class imbalance (~1.6% fraud rate).
- Performed threshold calibration on the precision-recall curve, since
  ROC-AUC/PR-AUC alone don't determine an operating decision threshold.
- Retrained the selected model (Tuned GBDT) on the full training period
  (Jan–Sep), using the hyperparameters and tree count found during tuning,
  to build the final production model.
- Evaluated the production model, at the end, on the test set (`transactions_test.csv`, Oct–Dec).
- SHAP-based interpretability
- Drift comparison (pre vs post month 6)

## Project structure

```
.
├── transactions_train.csv    # not included — transactions before 2023-10-01
├── transactions_test.csv     # not included — transactions on/after 2023-10-01
├── fraud_detection_analysis.ipynb
├── README.md
```


```
pandas
numpy
matplotlib
seaborn
scikit-learn
lightgbm
scipy
```
## How to run

Place the transaction CSVs in the project root, then open the notebook.

