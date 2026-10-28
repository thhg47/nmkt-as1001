# Freeze record

- Group approval: recorded by the explicit `--approve-freeze` command
- Frozen at (UTC): 2026-10-28T14:34:30+00:00
- Test evaluated before this record: no
- Baseline: PriorModel using train-set positive rate; threshold 0.21
- Selected model: logistic_regression_C=10
- Selected model configuration: `{"C": 10.0, "class_weight": "balanced", "estimator": "LogisticRegression", "max_iter": 2000, "random_state": 20261022, "solver": "lbfgs"}`
- Selected threshold: 0.79
- Threshold rule: `p_noisy >= threshold`
- Flag-rate cap: 20%
- Features: missing_ratio, baseline_variation_mv, high_frequency_ratio, clipping_ratio, amplitude_scale_mv, device_id
- Numeric features: missing_ratio, baseline_variation_mv, high_frequency_ratio, clipping_ratio, amplitude_scale_mv
- Categorical features: device_id
- Preprocessing: StandardScaler for numeric; OneHotEncoder(handle_unknown=ignore) for categorical
- Random seed: 20261022
- Logistic Regression candidates: C in (0.01, 0.1, 1.0, 10.0, 100.0), class_weight=balanced, max_iter=2000, random_state=20261022, solver=lbfgs
- Random Forest candidate: n_estimators=300, class_weight=balanced, random_state=20261022, n_jobs=-1
- Selection rule: Threshold grid 0.01..0.99 by 0.01; minimize cost_per_100 subject to flag_rate <= flag_rate_cap; threshold ties keep the first ascending threshold; model ties prefer lower flag rate, model name, then lower C.

| Candidate | Configuration | Validation threshold | Validation cost/100 | Validation flag rate |
|---|---|---:|---:|---:|
| logistic_regression_C=10 | `{"C": 10.0, "class_weight": "balanced", "estimator": "LogisticRegression", "max_iter": 2000, "random_state": 20261022, "solver": "lbfgs"}` | 0.79 | 34.00 | 0.195 |
| logistic_regression_C=100 | `{"C": 100.0, "class_weight": "balanced", "estimator": "LogisticRegression", "max_iter": 2000, "random_state": 20261022, "solver": "lbfgs"}` | 0.79 | 34.00 | 0.195 |
| logistic_regression_C=1 | `{"C": 1.0, "class_weight": "balanced", "estimator": "LogisticRegression", "max_iter": 2000, "random_state": 20261022, "solver": "lbfgs"}` | 0.78 | 34.50 | 0.200 |
| logistic_regression_C=0.1 | `{"C": 0.1, "class_weight": "balanced", "estimator": "LogisticRegression", "max_iter": 2000, "random_state": 20261022, "solver": "lbfgs"}` | 0.78 | 35.25 | 0.193 |
| logistic_regression_C=0.01 | `{"C": 0.01, "class_weight": "balanced", "estimator": "LogisticRegression", "max_iter": 2000, "random_state": 20261022, "solver": "lbfgs"}` | 0.69 | 36.00 | 0.200 |
| random_forest | `{"class_weight": "balanced", "estimator": "RandomForestClassifier", "n_estimators": 300, "n_jobs": -1, "random_state": 20261022}` | 0.49 | 37.50 | 0.200 |
