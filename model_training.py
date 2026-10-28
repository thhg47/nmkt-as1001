
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (
    SEED,
    NUM,
    CAT,
    FEATURES,
    TARGET,
    FLAG_RATE_CAP,
    load_data,
    get_splits,
)


ROOT = Path(__file__).resolve().parent
FREEZE_PATH = ROOT / "freeze_config.json"
RECORD_PATH = ROOT / "freeze_record.generated.md"

LOGISTIC_C_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)


# ============================================================
# 1. PREPROCESSING
# ============================================================

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ]
    )


# ============================================================
# 2. COST AND THRESHOLD LOGIC
# Compatible with teammate 4's threshold-selection rules
# ============================================================

def cost_per_100(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if len(y_true) == 0:
        raise ValueError("Cannot calculate cost on empty data.")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal lengths.")

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return float(100 * (5 * fn + fp) / len(y_true))


def choose_threshold(y_true, y_prob, max_flag_rate=FLAG_RATE_CAP):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) == 0 or len(y_true) != len(y_prob):
        raise ValueError(
            "y_true and y_prob must have equal, nonzero lengths."
        )

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1.")

    if not np.isfinite(y_prob).all():
        raise ValueError("y_prob contains non-finite values.")

    if ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    if not 0 <= max_flag_rate <= 1:
        raise ValueError("max_flag_rate must be between 0 and 1.")

    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)

    best_threshold = None
    min_cost = float("inf")
    best_flag_rate = 0.0

    costs = []
    flag_rates = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        flag_rate = float(np.mean(y_pred))
        cost = cost_per_100(y_true, y_pred)

        costs.append(cost)
        flag_rates.append(flag_rate)

        # Keep the first threshold if costs are tied.
        if flag_rate <= max_flag_rate and cost < min_cost:
            min_cost = cost
            best_threshold = float(threshold)
            best_flag_rate = flag_rate

    if best_threshold is None:
        raise ValueError(
            "No threshold satisfies the flag-rate cap. "
            "Do not silently select a threshold that violates the constraint."
        )

    print(
        f"Optimal Threshold: {best_threshold:.2f} | "
        f"Cost: {min_cost:.2f} | "
        f"Flag Rate: {best_flag_rate:.1%}"
    )

    return (
        best_threshold,
        thresholds,
        np.array(costs),
        np.array(flag_rates),
    )


# ============================================================
# 3. SUMMARIZE MODEL PERFORMANCE ON VALIDATION
# ============================================================

def summarize_model(name, model, threshold, va, c_value=None):
    probabilities = model.predict_proba(va[FEATURES])[:, 1]
    y_true = va[TARGET].to_numpy(dtype=int)
    y_pred = (probabilities >= threshold).astype(int)

    result = {
        "name": name,
        "threshold": float(threshold),
        "validation_cost_per_100": cost_per_100(y_true, y_pred),
        "validation_flag_rate": float(np.mean(y_pred)),
    }

    if c_value is not None:
        result["C"] = float(c_value)

    return result


# ============================================================
# 4. TRAIN BASELINE AND CANDIDATE MODELS
# ============================================================

def select_models(tr, va):
    tr = tr.reset_index(drop=True)
    va = va.reset_index(drop=True)

    if tr.empty or va.empty:
        raise ValueError(
            "Training and validation data cannot be empty."
        )

    required_columns = FEATURES + [TARGET]

    for name, frame in [
        ("training", tr),
        ("validation", va),
    ]:
        missing = set(required_columns) - set(frame.columns)

        if missing:
            raise ValueError(
                f"{name} data are missing columns: {sorted(missing)}"
            )

        if frame[required_columns].isna().any().any():
            raise ValueError(
                f"{name} data contain missing values. "
                "Resolve the missing-value policy with the group "
                "before training."
            )

        if not frame[TARGET].isin([0, 1]).all():
            raise ValueError(
                f"{name} target must contain only 0 and 1."
            )

    x_train = tr[FEATURES]
    y_train = tr[TARGET].astype(int)
    y_val = va[TARGET].to_numpy(dtype=int)

    # Baseline
    baseline = DummyClassifier(strategy="prior")
    baseline.fit(x_train, y_train)

    baseline_probabilities = baseline.predict_proba(
        va[FEATURES]
    )[:, 1]

    t_base, _, _, _ = choose_threshold(
        y_val,
        baseline_probabilities,
        max_flag_rate=FLAG_RATE_CAP,
    )

    baseline_summary = summarize_model(
        "dummy_prior",
        baseline,
        t_base,
        va,
    )

    candidates = []

    # Logistic Regression: tune C on validation.
    for c_value in LOGISTIC_C_VALUES:
        candidate = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        C=c_value,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=SEED,
                    ),
                ),
            ]
        )

        candidate.fit(x_train, y_train)

        probabilities = candidate.predict_proba(
            va[FEATURES]
        )[:, 1]

        threshold, _, _, _ = choose_threshold(
            y_val,
            probabilities,
            max_flag_rate=FLAG_RATE_CAP,
        )

        summary = summarize_model(
            "logistic_regression",
            candidate,
            threshold,
            va,
            c_value=c_value,
        )

        candidates.append({
            "model": candidate,
            "summary": summary,
        })

    # Second candidate model: Random Forest.
    forest = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    forest.fit(x_train, y_train)

    forest_probabilities = forest.predict_proba(
        va[FEATURES]
    )[:, 1]

    forest_threshold, _, _, _ = choose_threshold(
        y_val,
        forest_probabilities,
        max_flag_rate=FLAG_RATE_CAP,
    )

    forest_summary = summarize_model(
        "random_forest",
        forest,
        forest_threshold,
        va,
    )

    candidates.append({
        "model": forest,
        "summary": forest_summary,
    })

    # Lowest validation cost wins; then prefer lower workload and stable name/C order.
    selected = min(
        candidates,
        key=lambda item: (
            item["summary"]["validation_cost_per_100"],
            item["summary"]["validation_flag_rate"],
            item["summary"]["name"],
            item["summary"].get("C", float("inf")),
        ),
    )

    selected_model = selected["model"]
    selected_summary = selected["summary"]
    selected_threshold = float(selected_summary["threshold"])

    cfg = {
        "seed": int(SEED),
        "flag_rate_cap": float(FLAG_RATE_CAP),
        "selection_rule": (
            "Test thresholds from 0.01 to 0.99 in increments "
            "of 0.01. Predict class 1 when probability >= threshold. "
            "Minimize cost per 100 subject to flag rate <= the "
            "configured cap. Keep the first threshold in case of "
            "a threshold-cost tie. If model costs tie, prefer lower "
            "flag rate, then model name and lower C. Raise an error "
            "if no threshold satisfies the cap."
        ),
        "baseline": baseline_summary,
        "candidates": [
            item["summary"] for item in candidates
        ],
        "selected_model": {
            "name": selected_summary["name"],
            "C": selected_summary.get("C"),
            "threshold": selected_threshold,
            "validation_cost_per_100": float(
                selected_summary["validation_cost_per_100"]
            ),
            "validation_flag_rate": float(
                selected_summary["validation_flag_rate"]
            ),
        },
    }

    return (
        baseline,
        float(t_base),
        selected_model,
        selected_threshold,
        cfg,
    )


# ============================================================
# 5. FREEZE CONFIGURATION
# ============================================================

def freeze_signature(cfg):
    selected = cfg["selected_model"]

    # Only deterministic model-selection settings.
    # Do not put timestamps, commit hashes or other manual notes here.
    return {
        "seed": cfg["seed"],
        "flag_rate_cap": cfg["flag_rate_cap"],
        "baseline_name": cfg["baseline"]["name"],
        "t_base": cfg["baseline"]["threshold"],
        "selected_model_name": selected["name"],
        "selected_model_C": selected.get("C"),
        "t_model": selected["threshold"],
    }


def write_freeze(cfg):
    # Check BOTH paths before writing anything.
    # Never overwrite the group's existing files.
    existing = [
        path.name
        for path in (FREEZE_PATH, RECORD_PATH)
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing file(s): "
            + ", ".join(existing)
            + ". Review or rename them with the group's agreement "
              "before trying to freeze again."
        )

    signature = freeze_signature(cfg)
    selected = cfg["selected_model"]

    # JSON contains deterministic configuration only.
    # No timestamp, commit hash, bootstrap result or manual notes.
    frozen = {
        "freeze_version": 1,
        "selection": signature,
        "validation_summary": cfg,
    }

    if selected["name"] == "logistic_regression":
        reason = (
            f"Logistic Regression with C={selected['C']} had "
            "the lowest validation cost among the evaluated "
            "candidates, subject to the flag-rate constraint."
        )
    else:
        reason = (
            "Random Forest had the lowest validation cost among "
            "the evaluated candidates, subject to the flag-rate "
            "constraint."
        )

    # Timestamp is recorded ONLY in the generated Markdown record.
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    record = f"""# Model Freeze Record (Generated Draft)

- Timestamp: {timestamp}
- Seed: {SEED}
- Features: {', '.join(FEATURES)}
- Numeric features: {', '.join(NUM)}
- Categorical features: {', '.join(CAT)}
- Preprocessing: StandardScaler for numeric features; OneHotEncoder(handle_unknown="ignore") for categorical features.
- Baseline: {cfg['baseline']['name']}
- Baseline threshold: {cfg['baseline']['threshold']:.2f}
- Selected model: {selected['name']}
- Selected Logistic Regression C: {selected.get('C')}
- Selected threshold: {selected['threshold']:.2f}
- Validation cost per 100: {selected['validation_cost_per_100']:.4f}
- Validation flag rate: {selected['validation_flag_rate']:.4f}
- Maximum flag rate: {FLAG_RATE_CAP:.2f}
- Logistic Regression parameters: class_weight="balanced", max_iter=2000
- Random Forest parameters: n_estimators=300, class_weight="balanced"

## Selection reason

{reason}

## Decision rules

- Thresholds are selected using validation data only.
- Cost per 100 = 100 * (5 * FN + FP) / N.
- Only thresholds with flag rate <= {FLAG_RATE_CAP:.2f} are feasible.
- If no threshold satisfies the flag-rate cap, stop and resolve the constraint; do not silently violate it.
- Threshold-cost ties keep the first threshold encountered.
- Model-cost ties prefer lower flag rate, then model name and lower C.
- The test set is not used for model or threshold selection.
- Final test evaluation requires --final and a verified frozen selection.

## Validation candidate results

```json
{json.dumps(cfg['candidates'], indent=2, ensure_ascii=False)}
```

## Manual completion by the group

- Bootstrap analysis: to be added by the group.
- Missing-value (NA) analysis and policy: to be added by the group.
- Git commit hash: to be added manually by the group.
- This is a generated draft. The group owner should review and merge it into the official freeze record.
"""

    # Write JSON first, then the generated draft.
    # If writing the draft fails, remove the newly created JSON.
    # Existing files were checked above and are never overwritten.
    FREEZE_PATH.write_text(
        json.dumps(
            frozen,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    try:
        RECORD_PATH.write_text(record, encoding="utf-8")
    except Exception:
        FREEZE_PATH.unlink(missing_ok=True)
        raise

    print(f"Saved frozen configuration: {FREEZE_PATH.name}")
    print(f"Saved generated draft: {RECORD_PATH.name}")
    print("Timestamp is stored only in the generated Markdown draft.")
    print("Bootstrap, NA analysis and Git commit hash remain manual.")
    print(json.dumps(signature, indent=2, ensure_ascii=False))


def check_freeze(cfg):
    if not FREEZE_PATH.exists():
        raise FileNotFoundError(
            "freeze_config.json not found. "
            "Run python model_training.py --freeze first."
        )

    frozen = json.loads(
        FREEZE_PATH.read_text(encoding="utf-8")
    )

    expected = frozen.get("selection")
    actual = freeze_signature(cfg)

    if expected != actual:
        raise AssertionError(
            "Model selection differs from the frozen configuration.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )

    print("Frozen model selection verified.")


# ============================================================
# 6. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train, select, freeze and evaluate ECG quality models."
    )

    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--freeze",
        action="store_true",
        help="Save the chosen configuration based on validation.",
    )
    modes.add_argument(
        "--final",
        action="store_true",
        help="Verify the frozen choice and run final test evaluation.",
    )

    args = parser.parse_args()

    df = load_data()
    tr, va, te = get_splits(df)

    baseline, t_base, model, t_model, cfg = select_models(tr, va)

    print("\nValidation candidate summary:")
    print(json.dumps(cfg, indent=2, ensure_ascii=False))

    if args.freeze:
        write_freeze(cfg)
        return

    if args.final:
        check_freeze(cfg)

        import evaluate

        evaluate.run_final(
            base=baseline,
            model=model,
            t_base=t_base,
            t_model=t_model,
            va=va,
            te=te,
        )

        print("Final evaluation completed.")
        return

    import evaluate

    evaluate.run_validation(
        base=baseline,
        model=model,
        t_base=t_base,
        t_model=t_model,
        va=va,
    )

    print("\nIf the group approves this selection, run:")
    print("python model_training.py --freeze")


if __name__ == "__main__":
    main()
