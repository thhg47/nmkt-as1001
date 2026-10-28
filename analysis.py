import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

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
    FEATURES,
    FLAG_RATE_CAP,
    NUM,
    SEED,
    TARGET,
    get_splits,
    load_data,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / 'results'
MODEL_PATH = ROOT / 'frozen_models.joblib'
FREEZE_PATH = ROOT / 'freeze_record.json'
FREEZE_MARKDOWN_PATH = ROOT / 'freeze_record.md'
LOGISTIC_C_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)
LOGISTIC_MAX_ITER = 2000


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
                max_iter=LOGISTIC_MAX_ITER,
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


def select_from_validation():
    train, validation, _ = get_splits(load_data())
    baseline, selected, all_results = fit_candidates(train, validation)
    baseline_threshold = evaluate.select_threshold(
        validation[TARGET].to_numpy(),
        baseline.predict_proba(validation[FEATURES])[:, 1],
        FLAG_RATE_CAP,
    )['threshold']
    return train, validation, baseline, selected, all_results, baseline_threshold


def develop():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train, validation, baseline, selected, all_results, baseline_threshold = (
        select_from_validation()
    )

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
    print('Validation artifacts updated. No model or freeze record was written.')


def model_configuration(name):
    if name.startswith('logistic_regression_C='):
        return {
            'estimator': 'LogisticRegression',
            'C': float(name.split('=', 1)[1]),
            'class_weight': 'balanced',
            'max_iter': LOGISTIC_MAX_ITER,
            'random_state': SEED,
            'solver': 'lbfgs',
        }
    if name == 'random_forest':
        return {
            'estimator': 'RandomForestClassifier',
            'n_estimators': 300,
            'class_weight': 'balanced',
            'random_state': SEED,
            'n_jobs': -1,
        }
    raise ValueError(f'Unknown candidate model: {name}')


def write_freeze(
    train,
    baseline,
    selected,
    all_results,
    baseline_threshold,
):
    paths = (MODEL_PATH, FREEZE_PATH, FREEZE_MARKDOWN_PATH)
    existing = [path.name for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            'Refusing to overwrite existing freeze artifact(s): '
            + ', '.join(existing)
        )

    frozen_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
    candidates = [
        {
            'model': result['name'],
            'configuration': model_configuration(result['name']),
            'threshold': float(result['threshold']),
            'validation_cost_per_100': float(result['cost_per_100']),
            'validation_flag_rate': float(result['flag_rate']),
        }
        for result in all_results
    ]
    freeze = {
        'status': 'group_approved_freeze',
        'frozen_at_utc': frozen_at,
        'selection_split': 'validation',
        'test_opened': False,
        'threshold_rule': 'p_noisy >= threshold',
        'flag_rate_cap': FLAG_RATE_CAP,
        'seed': SEED,
        'features': FEATURES,
        'numeric_features': NUM,
        'categorical_features': CAT,
        'preprocessing': {
            'numeric': 'StandardScaler',
            'categorical': 'OneHotEncoder(handle_unknown=ignore)',
        },
        'baseline': {
            'estimator': 'PriorModel',
            'positive_rate': float(train[TARGET].mean()),
            'threshold': float(baseline_threshold),
        },
        'selection_metric': (
            'Threshold grid 0.01..0.99 by 0.01; minimize cost_per_100 '
            'subject to flag_rate <= flag_rate_cap; threshold ties keep '
            'the first ascending threshold; model ties prefer lower '
            'flag rate, model name, then lower C.'
        ),
        'candidates': candidates,
        'selected_model': selected['name'],
        'selected_model_configuration': model_configuration(selected['name']),
        'selected_threshold': float(selected['threshold']),
        'selected_validation_cost_per_100': float(selected['cost_per_100']),
        'selected_validation_flag_rate': float(selected['flag_rate']),
    }

    candidates_markdown = '\n'.join(
        f"| {row['model']} | `{json.dumps(row['configuration'], sort_keys=True)}` | "
        f"{row['threshold']:.6g} | "
        f"{row['validation_cost_per_100']:.2f} | {row['validation_flag_rate']:.3f} |"
        for row in candidates
    )
    freeze_markdown = f"""# Freeze record

- Group approval: recorded by the explicit `--approve-freeze` command
- Frozen at (UTC): {frozen_at}
- Test evaluated before this record: no
- Baseline: PriorModel using train-set positive rate; threshold {baseline_threshold:.6g}
- Selected model: {selected['name']}
- Selected model configuration: `{json.dumps(freeze['selected_model_configuration'], sort_keys=True)}`
- Selected threshold: {selected['threshold']:.6g}
- Threshold rule: `{freeze['threshold_rule']}`
- Flag-rate cap: {FLAG_RATE_CAP:.0%}
- Features: {', '.join(FEATURES)}
- Numeric features: {', '.join(NUM)}
- Categorical features: {', '.join(CAT)}
- Preprocessing: StandardScaler for numeric; OneHotEncoder(handle_unknown=ignore) for categorical
- Random seed: {SEED}
- Logistic Regression candidates: C in {LOGISTIC_C_VALUES}, class_weight=balanced, max_iter={LOGISTIC_MAX_ITER}, random_state={SEED}, solver=lbfgs
- Random Forest candidate: n_estimators=300, class_weight=balanced, random_state={SEED}, n_jobs=-1
- Selection rule: {freeze['selection_metric']}

| Candidate | Configuration | Validation threshold | Validation cost/100 | Validation flag rate |
|---|---|---:|---:|---:|
{candidates_markdown}
"""

    with TemporaryDirectory(dir=ROOT) as temp_dir:
        temp = Path(temp_dir)
        staged_model = temp / MODEL_PATH.name
        staged_json = temp / FREEZE_PATH.name
        staged_markdown = temp / FREEZE_MARKDOWN_PATH.name
        joblib.dump({
            'baseline': baseline,
            'final_model': selected['model'],
            'baseline_threshold': baseline_threshold,
            'final_threshold': selected['threshold'],
            'selected_model': selected['name'],
            'selected_model_configuration': freeze['selected_model_configuration'],
        }, staged_model)
        staged_json.write_text(json.dumps(freeze, indent=2) + '\n', encoding='utf-8')
        staged_markdown.write_text(freeze_markdown, encoding='utf-8')
        for staged, destination in zip(
            (staged_model, staged_json, staged_markdown), paths
        ):
            if destination.exists():
                raise FileExistsError(
                    f'Refusing to overwrite freeze artifact: {destination}'
                )
            os.link(staged, destination)
            staged.unlink()

    print(f'Frozen model: {MODEL_PATH}')
    print(f'Freeze record: {FREEZE_MARKDOWN_PATH}')


def final_evaluation():
    if not FREEZE_PATH.is_file() or not MODEL_PATH.is_file():
        raise FileNotFoundError(
            'No approved root-level freeze artifacts found. Review validation results, '
            'then run `python analysis.py --stage freeze --approve-freeze` after group approval.'
        )
    final_outputs = (
        'metrics.csv',
        'test_predictions.csv',
        'bootstrap_ci.csv',
        'by_device.csv',
        'by_participant.csv',
        'calibration.csv',
        'naive_costs.csv',
        'figures/confusion_test.png',
    )
    existing = [name for name in final_outputs if (ROOT / name).exists()]
    if existing:
        raise FileExistsError(
            'Final output already exists; refusing to overwrite or rerun test evaluation: '
            + ', '.join(existing)
        )

    freeze = json.loads(FREEZE_PATH.read_text(encoding='utf-8'))
    if freeze.get('status') != 'group_approved_freeze':
        raise ValueError('Freeze record is not an approved group freeze')
    if freeze.get('test_opened') is not False:
        raise ValueError('Freeze record is invalid: test_opened must be false before evaluation')
    artifacts = joblib.load(MODEL_PATH)
    if (
        artifacts.get('selected_model') != freeze.get('selected_model')
        or artifacts.get('selected_model_configuration')
        != freeze.get('selected_model_configuration')
        or artifacts.get('final_threshold') != freeze.get('selected_threshold')
    ):
        raise ValueError('Frozen model artifact does not match the approved freeze record')
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
        choices=('develop', 'freeze', 'final'),
        default='develop',
        help='develop reviews validation only; freeze writes approved artifacts; final evaluates test once',
    )
    parser.add_argument(
        '--approve-freeze',
        action='store_true',
        help='Explicitly confirm group approval when using --stage freeze',
    )
    args = parser.parse_args()
    if args.stage == 'develop':
        if args.approve_freeze:
            parser.error('--approve-freeze is only valid with --stage freeze')
        develop()
    elif args.stage == 'freeze':
        if not args.approve_freeze:
            parser.error('--stage freeze requires --approve-freeze after group approval')
        train, _, baseline, selected, all_results, baseline_threshold = (
            select_from_validation()
        )
        write_freeze(
            train,
            baseline,
            selected,
            all_results,
            baseline_threshold,
        )
    else:
        if args.approve_freeze:
            parser.error('--approve-freeze is only valid with --stage freeze')
        final_evaluation()


if __name__ == '__main__':
    main()
