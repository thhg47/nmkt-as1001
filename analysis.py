import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import evaluate
import threshold
from common import (
    CAT,
    DATA_DIR,
    FEATURES,
    FLAG_RATE_CAP,
    NUM,
    SEED,
    TARGET,
    get_splits,
    load_data,
)


OUTPUT_DIR = Path(__file__).resolve().parent / 'results'
MODEL_PATH = OUTPUT_DIR / 'frozen_models.joblib'
FREEZE_PATH = OUTPUT_DIR / 'freeze_record.json'
FREEZE_MARKDOWN_PATH = OUTPUT_DIR / 'freeze_record.md'
LOGISTIC_C_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)


class PriorModel:
    def __init__(self, positive_rate):
        self.positive_rate = float(positive_rate)

    def predict_proba(self, X):
        p = np.full(len(X), self.positive_rate)
        return np.column_stack((1 - p, p))


def make_preprocessor():
    return ColumnTransformer([
        ('numeric', StandardScaler(), NUM),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), CAT),
    ])


def fit_candidates(train, validation):
    y_train = train[TARGET]
    baseline = PriorModel(y_train.mean())
    candidates = {}

    for c in LOGISTIC_C_VALUES:
        model = make_pipeline(
            make_preprocessor(),
            LogisticRegression(
                C=c,
                class_weight='balanced',
                max_iter=1000,
                random_state=SEED,
            ),
        )
        model.fit(train[FEATURES], y_train)
        candidates[f'logistic_regression_C={c:g}'] = model

    forest = make_pipeline(
        make_preprocessor(),
        RandomForestClassifier(
            n_estimators=300,
            class_weight='balanced',
            random_state=SEED,
            n_jobs=-1,
        ),
    )
    forest.fit(train[FEATURES], y_train)
    candidates['random_forest'] = forest

    y_validation = validation[TARGET].to_numpy()
    scored = []
    for name, model in candidates.items():
        probabilities = model.predict_proba(validation[FEATURES])[:, 1]
        selected = evaluate.select_threshold(
            y_validation,
            probabilities,
            FLAG_RATE_CAP,
        )
        scored.append({
            'name': name,
            'model': model,
            'probabilities': probabilities,
            **selected,
        })

    scored.sort(key=lambda result: (
        result['cost_per_100'],
        result['flag_rate'],
        result['name'],
    ))
    return baseline, scored[0], scored


def develop():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train, validation, _ = get_splits(load_data())
    baseline, selected, all_results = fit_candidates(train, validation)
    baseline_threshold = evaluate.select_threshold(
        validation[TARGET].to_numpy(),
        baseline.predict_proba(validation[FEATURES])[:, 1],
        FLAG_RATE_CAP,
    )['threshold']

    print(f'Train segments: {len(train)}; validation segments: {len(validation)}')
    print(f'Flag-rate cap: {FLAG_RATE_CAP:.0%}')
    print('Validation candidates:')
    for result in all_results:
        print(
            f"  {result['name']}: threshold={result['threshold']:.6g}, "
            f"cost/100={result['cost_per_100']:.2f}, "
            f"flag_rate={result['flag_rate']:.3f}"
        )

    evaluate.run_validation(
        baseline,
        selected['model'],
        baseline_threshold,
        selected['threshold'],
        validation,
    )
    threshold.run_threshold_analysis(
        selected['model'],
        selected['threshold'],
        validation,
    )

    joblib.dump({
        'baseline': baseline,
        'final_model': selected['model'],
        'baseline_threshold': baseline_threshold,
        'final_threshold': selected['threshold'],
    }, MODEL_PATH)

    freeze = {
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'selection_split': 'validation',
        'test_opened': False,
        'threshold_rule': 'p_noisy >= threshold',
        'flag_rate_cap': FLAG_RATE_CAP,
        'seed': SEED,
        'features': FEATURES,
        'numeric_features': NUM,
        'categorical_features': CAT,
        'preprocessing': 'StandardScaler for numeric; OneHotEncoder(handle_unknown=ignore) for device_id',
        'selection_metric': 'thresholds 0.01..0.99 in 0.01 steps; minimize cost_per_100 subject to flag_rate <= flag_rate_cap; threshold ties keep first; model ties prefer lower flag rate then model name',
        'candidates': [
            {
                'model': result['name'],
                'threshold': result['threshold'],
                'validation_cost_per_100': result['cost_per_100'],
                'validation_flag_rate': result['flag_rate'],
            }
            for result in all_results
        ],
        'selected_model': selected['name'],
        'selected_threshold': selected['threshold'],
        'selected_validation_cost_per_100': selected['cost_per_100'],
        'selected_validation_flag_rate': selected['flag_rate'],
    }
    FREEZE_PATH.write_text(json.dumps(freeze, indent=2) + '\n', encoding='utf-8')
    candidates_markdown = '\n'.join(
        f"| {row['model']} | {row['threshold']:.6g} | "
        f"{row['validation_cost_per_100']:.2f} | {row['validation_flag_rate']:.3f} |"
        for row in freeze['candidates']
    )
    freeze_markdown = f"""# Freeze record

- Frozen at (UTC): {freeze['frozen_at_utc']}
- Test evaluated before this record: no
- Selected model: {freeze['selected_model']}
- Selected threshold: {freeze['selected_threshold']:.6g}
- Threshold rule: `{freeze['threshold_rule']}`
- Flag-rate cap: {freeze['flag_rate_cap']:.0%}
- Features: {', '.join(freeze['features'])}
- Numeric features: {', '.join(freeze['numeric_features'])}
- Categorical features: {', '.join(freeze['categorical_features'])}
- Preprocessing: {freeze['preprocessing']}
- Random seed: {freeze['seed']}
- Selection rule: {freeze['selection_metric']}

| Candidate | Validation threshold | Validation cost/100 | Validation flag rate |
|---|---:|---:|---:|
{candidates_markdown}
"""
    FREEZE_MARKDOWN_PATH.write_text(freeze_markdown, encoding='utf-8')
    print(f'Frozen model: {MODEL_PATH}')
    print(f'Freeze record: {FREEZE_MARKDOWN_PATH}')


def final_evaluation():
    if not FREEZE_PATH.is_file() or not MODEL_PATH.is_file():
        raise FileNotFoundError(
            'Run `python analysis.py --stage develop` first to create the frozen model and freeze record.'
        )
    if (Path.cwd() / 'test_predictions.csv').exists():
        raise FileExistsError(
            'test_predictions.csv already exists; refusing to overwrite or rerun test evaluation.'
        )

    freeze = json.loads(FREEZE_PATH.read_text(encoding='utf-8'))
    if freeze.get('test_opened') is not False:
        raise ValueError('Freeze record is invalid: test_opened must be false before evaluation')
    artifacts = joblib.load(MODEL_PATH)
    _, validation, test = get_splits(load_data())
    evaluate.run_final(
        artifacts['baseline'],
        artifacts['final_model'],
        artifacts['baseline_threshold'],
        artifacts['final_threshold'],
        validation,
        test,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Train/select using validation, then evaluate the frozen model on test.'
    )
    parser.add_argument(
        '--stage',
        choices=('develop', 'final'),
        default='develop',
        help='develop trains/selects and writes the freeze record; final evaluates test once',
    )
    args = parser.parse_args()
    if args.stage == 'develop':
        develop()
    else:
        final_evaluation()


if __name__ == '__main__':
    main()
