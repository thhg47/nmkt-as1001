
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
    cost_per_100,
)


ROOT = Path(__file__).resolve().parent
FREEZE_PATH = ROOT / "freeze_config.json"
RECORD_PATH = ROOT / "freeze_record.md"

LOGISTIC_C_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)
THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)
THRESHOLDS = np.append(THRESHOLDS, 1.0)


# --------------------------------------------------
# 1. Preprocessing
# --------------------------------------------------

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ]
    )


# --------------------------------------------------
# 2. Select threshold using validation only
# --------------------------------------------------

def choose_threshold(y, probabilities):
    y = np.asarray(y, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    if y.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("y and probabilities must be 1D arrays.")

    if len(y) == 0 or len(y) != len(probabilities):
        raise ValueError("y and probabilities must have equal nonzero length.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("y must contain only binary labels.")

    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("Probabilities must be finite and in [0, 1].")

    feasible = []

    for threshold in THRESHOLDS:
        flag_rate = float(np.mean(probabilities >= threshold))

        if flag_rate <= FLAG_RATE_CAP + 1e-12:
            cost = float(
                cost_per_100(y, probabilities, float(threshold))
            )
            feasible.append((cost, float(threshold), flag_rate))

    if not feasible:
        raise ValueError("No threshold satisfies FLAG_RATE_CAP.")

    # Tie rule: lowest cost first, then lowest threshold.
    cost, threshold, _ = min(
        feasible,
        key=lambda item: (item[0], item[1]),
    )

    return threshold


def summarize_model(name, model, threshold, va, c_value=None):
    probabilities = model.predict_proba(va[FEATURES])[:, 1]
    y = va[TARGET].to_numpy()

    result = {
        "name": name,
        "threshold": float(threshold),
        "validation_cost_per_100": float(
            cost_per_100(y, probabilities, threshold)
        ),
        "validation_flag_rate": float(
            np.mean(probabilities >= threshold)
        ),
    }

    if c_value is not None:
        result["C"] = float(c_value)

    return result


# --------------------------------------------------
# 3. Fit baseline and compare candidate models
# --------------------------------------------------

def select_models(tr, va):
    """
    Fit models on training data.
    Choose thresholds and the final model using validation only.

    Returns:
        base, t_base, model, t_model, cfg
    """
    tr = tr.reset_index(drop=True)
    va = va.reset_index(drop=True)

    if tr.empty or va.empty:
        raise ValueError("Training and validation data cannot be empty.")

    for name, frame in [("training", tr), ("validation", va)]:
        missing = set(FEATURES + [TARGET]) - set(frame.columns)

        if missing:
            raise ValueError(
                f"{name} data are missing columns: {sorted(missing)}"
            )

        if frame[FEATURES + [TARGET]].isna().any().any():
            raise ValueError(f"{name} data contain missing values.")

        if not frame[TARGET].isin([0, 1]).all():
            raise ValueError(f"{name} target must contain only 0 and 1.")

    x_train = tr[FEATURES]
    y_train = tr[TARGET].astype(int)
    y_val = va[TARGET].to_numpy(dtype=int)

    # Baseline: predicts the training-set class prior probability.
    base = DummyClassifier(
        strategy="prior",
    )
    base.fit(x_train, y_train)

    baseline_probabilities = base.predict_proba(va[FEATURES])[:, 1]
    t_base = choose_threshold(y_val, baseline_probabilities)

    baseline_summary = summarize_model(
        "dummy_prior", base, t_base, va
    )

    candidates = []

    # Logistic Regression: tune C using validation.
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

        probabilities = candidate.predict_proba(va[FEATURES])[:, 1]
        threshold = choose_threshold(y_val, probabilities)

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

    # Second candidate family: Random Forest.
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

    forest_probabilities = forest.predict_proba(va[FEATURES])[:, 1]
    forest_threshold = choose_threshold(y_val, forest_probabilities)

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

    # Candidates are kept in deterministic order:
    # Logistic Regression C values ascending, then Random Forest.
    # If validation costs tie, the earlier candidate wins.
    selected = min(
        candidates,
        key=lambda item: item["summary"]["validation_cost_per_100"],
    )

    model = selected["model"]
    selected_summary = selected["summary"]
    t_model = float(selected_summary["threshold"])

    cfg = {
        "seed": int(SEED),
        "flag_rate_cap": float(FLAG_RATE_CAP),
        "selection_rule": (
            "Minimize validation cost among thresholds satisfying "
            "the flag-rate cap. Break threshold ties by choosing "
            "the lowest threshold. Break model-cost ties by choosing "
            "the first candidate in the fixed candidate order."
        ),
        "baseline": baseline_summary,
        "candidates": [
            item["summary"] for item in candidates
        ],
        "selected_model": {
            "name": selected_summary["name"],
            "C": selected_summary.get("C"),
            "threshold": t_model,
            "validation_cost_per_100": float(
                selected_summary["validation_cost_per_100"]
            ),
            "validation_flag_rate": float(
                selected_summary["validation_flag_rate"]
            ),
        },
    }

    return base, float(t_base), model, t_model, cfg


# --------------------------------------------------
# 4. Freeze and reproducibility checks
# --------------------------------------------------

def freeze_signature(cfg):
    selected = cfg["selected_model"]

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
    signature = freeze_signature(cfg)

    frozen = {
        "freeze_version": 1,
        "selection": signature,
        "validation_summary": cfg,
    }

    FREEZE_PATH.write_text(
        json.dumps(
            frozen,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    selected = cfg["selected_model"]

    if selected["name"] == "logistic_regression":
        reason = (
            f"Logistic Regression with C={selected['C']} had the "
            "lowest validation cost among the evaluated candidates, "
            "subject to the flag-rate constraint."
        )
    else:
        reason = (
            "Random Forest had the lowest validation cost among "
            "the evaluated candidates, subject to the flag-rate "
            "constraint."
        )

    record = f"""# Model Freeze Record

## Frozen configuration

- Seed: {SEED}
- Maximum flag rate: {FLAG_RATE_CAP:.2f}
- Baseline: {cfg['baseline']['name']}
- Baseline threshold: {cfg['baseline']['threshold']:.2f}
- Selected model: {selected['name']}
- Selected Logistic Regression C: {selected.get('C')}
- Selected threshold: {selected['threshold']:.2f}

## Selection reason

{reason}

## Decision rules

- Thresholds are selected using validation data only.
- Minimize weighted cost: 5 per false negative and 1 per false positive.
- Only thresholds with flag rate <= {FLAG_RATE_CAP:.2f} are feasible.
- If thresholds have equal cost, choose the lowest threshold.
- If candidate models have equal validation cost, choose the first
  candidate in the fixed order: Logistic Regression C values ascending,
  followed by Random Forest.
- The test set is not used for model or threshold selection.
- Final test evaluation is performed only with --final, after the
  frozen selection has been checked against freeze_config.json.

## Validation candidate results

```json
{json.dumps(cfg['candidates'], indent=2, ensure_ascii=False)}
```
"""

    RECORD_PATH.write_text(record, encoding="utf-8")

    print(f"Saved frozen configuration: {FREEZE_PATH.name}")
    print(f"Saved freeze record: {RECORD_PATH.name}")
    print(json.dumps(signature, indent=2, ensure_ascii=False))


def check_freeze(cfg):
    if not FREEZE_PATH.exists():
        raise FileNotFoundError(
            "freeze_config.json not found. "
            "Run python model_training.py --freeze first."
        )

    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    expected = frozen.get("selection")
    actual = freeze_signature(cfg)

    if expected != actual:
        raise AssertionError(
            "Model selection differs from the frozen configuration.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )

    print("Frozen model selection verified.")


# --------------------------------------------------
# 5. Entry point
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train, select, freeze and evaluate ECG quality models."
    )

    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--freeze",
        action="store_true",
        help="Save the chosen model configuration based on validation.",
    )
    modes.add_argument(
        "--final",
        action="store_true",
        help="Verify the frozen choice and run final test evaluation.",
    )

    args = parser.parse_args()

    # Load and split the dataset. Test labels are not used for selection.
    df = load_data()
    tr, va, te = get_splits(df)

    # Every run reproduces selection using training and validation only.
    base, t_base, model, t_model, cfg = select_models(tr, va)

    if args.freeze:
        write_freeze(cfg)
        return

    if args.final:
        check_freeze(cfg)

        import evaluate

        evaluate.run_final(
            base=base,
            model=model,
            t_base=t_base,
            t_model=t_model,
            va=va,
            te=te,
        )
        print("Final evaluation completed.")
        return

    # Default mode: validation only. Never evaluate test here.
    import evaluate

    evaluate.run_validation(
        base=base,
        model=model,
        t_base=t_base,
        t_model=t_model,
        va=va,
    )

    print("\nValidation candidate summary:")
    print(json.dumps(cfg, indent=2, ensure_ascii=False))
    print("\nWhen the group approves the selection, run:")
    print("python model_training.py --freeze")


if __name__ == "__main__":
    main()
