from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# 1. CẤU HÌNH THEO ĐỀ BÀI
SEED = 20261022
TARGET = 'quality_label'
NUM = [
    'missing_ratio',
    'baseline_variation_mv',
    'high_frequency_ratio',
    'clipping_ratio',
    'amplitude_scale_mv',
]
CAT = ['device_id']
FEATURES = NUM + CAT
assert FEATURES == NUM + CAT

FLAG_RATE_CAP = 0.20
DATA_DIR = Path(__file__).resolve().parent


def cost_per_100(y, p, t):
    """Return weighted classification cost per 100 rows using p >= t."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)

    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError('y and p must be one-dimensional arrays of equal length')
    if len(y) == 0:
        raise ValueError('cost_per_100 requires at least one observation')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('y must contain only binary labels 0 and 1')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('p must contain finite probabilities in [0, 1]')
    if not np.isscalar(t) or not np.isfinite(t) or not 0 <= t <= 1:
        raise ValueError('t must be a finite threshold in [0, 1]')

    pred = p >= t
    fn = np.count_nonzero((y == 1) & ~pred)
    fp = np.count_nonzero((y == 0) & pred)
    return 100 * (5 * fn + fp) / len(y)


def load_data(data_dir=DATA_DIR):
    df = pd.read_json(data_dir / 'ecg_quality_features.jsonl', lines=True)
    split = pd.read_json(data_dir / 'participant_split.json', typ='series')
    df['split'] = df['participant_id'].map(split)
    assert len(df) == 2000 and df['segment_id'].is_unique
    assert df['split'].notna().all(), 'Some participants have no split'
    assert df.groupby('split')['participant_id'].nunique().to_dict() == \
           {'test': 10, 'train': 30, 'validation': 10}
    assert df[FEATURES + [TARGET]].notna().all().all()
    return df.reset_index(drop=True)


def get_splits(df):
    return tuple(df[df['split'] == s].reset_index(drop=True)
                 for s in ('train', 'validation', 'test'))


# 2. HÀM CHÍNH CỦA NGƯỜI 4
def choose_threshold(y_true, y_prob, max_flag_rate=FLAG_RATE_CAP):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    
    best_threshold = None
    min_cost = float('inf')
    best_flag_rate = 0.0

    costs = []
    flag_rates = []

    for t in thresholds:
        flag_rate = float(np.mean(y_prob >= t))
        cost = cost_per_100(y_true, y_prob, float(t))

        costs.append(cost)
        flag_rates.append(flag_rate)

        if flag_rate <= max_flag_rate and cost < min_cost:
            min_cost = cost
            best_threshold = float(t)
            best_flag_rate = flag_rate

    if best_threshold is None:
        idx = int(np.argmin(flag_rates))
        best_threshold = float(thresholds[idx])
        min_cost = float(costs[idx])
        best_flag_rate = float(flag_rates[idx])

    print(f"Optimal Threshold: {best_threshold:.2f} | Cost: {min_cost:.2f} | Flag Rate: {best_flag_rate:.1%}")
    return best_threshold, thresholds, np.array(costs), np.array(flag_rates)


def plot_cost_curve(thresholds, costs, flag_rates, best_threshold, out_path=DATA_DIR / 'reports' / 'cost_curve.png'):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(thresholds, costs, label='Cost per 100 segments (5*FN + FP)', color='navy', lw=2)
    ax1.axvline(x=best_threshold, color='crimson', linestyle='--', lw=1.8,
                label=f'Optimal Threshold = {best_threshold:.2f}')
    
    valid_mask = flag_rates <= FLAG_RATE_CAP
    if np.any(valid_mask):
        min_valid_t = thresholds[valid_mask][0]
        ax1.axvspan(min_valid_t, 0.99, color='green', alpha=0.1, label='Valid Region (Flag Rate <= 20%)')

    ax1.set_xlabel('Decision Threshold (t)')
    ax1.set_ylabel('Cost per 100 Segments', color='navy')
    ax1.set_title('Validation Cost Optimization & Flag Rate Constraint')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right')

    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    df = load_data()
    train_df, val_df, test_df = get_splits(df)

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_val, y_val = val_df[FEATURES], val_df[TARGET]

    preprocessor = ColumnTransformer([
        ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), NUM),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CAT),
    ])

    model = make_pipeline(
        preprocessor,
        RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight='balanced')
    )

    model.fit(X_train, y_train)
    y_val_prob = model.predict_proba(X_val)[:, 1]

    best_t, thresholds, costs, flag_rates = choose_threshold(y_val, y_val_prob)
    plot_cost_curve(thresholds, costs, flag_rates, best_t)
