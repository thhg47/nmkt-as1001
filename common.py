import numpy as np


SEED = 20261022
TARGET = 'quality_label'
FEATURES = [
    'missing_ratio',
    'baseline_variation_mv',
    'high_frequency_ratio',
    'clipping_ratio',
    'amplitude_scale_mv',
    'device_id',
]


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
