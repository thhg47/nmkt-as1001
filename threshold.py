from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import FLAG_RATE_CAP, FEATURES, TARGET, cost_per_100

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / 'figures'
GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)
ALT_CAPS = (0.10, 0.15, 0.20, 0.30)


def threshold_sweep(y, p, grid=GRID):
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    thresholds = np.asarray(grid, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError('y and p must be non-empty one-dimensional arrays of equal length')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('y must contain only binary labels 0 and 1')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('p must contain finite probabilities in [0, 1]')
    if thresholds.ndim != 1 or len(thresholds) == 0:
        raise ValueError('grid must be a non-empty one-dimensional array')
    if not np.isfinite(thresholds).all() or ((thresholds < 0) | (thresholds > 1)).any():
        raise ValueError('grid thresholds must be finite values in [0, 1]')
    n_pos = int((y == 1).sum())
    return pd.DataFrame([dict(threshold=float(t), flag_rate=float((p >= t).mean()),
                              recall=(((p >= t) & (y == 1)).sum() / n_pos
                                      if n_pos else np.nan),
                              cost_per_100=float(cost_per_100(y, p, t)))
                         for t in thresholds])


def choose_threshold(y, p, cap=FLAG_RATE_CAP):
    """Choose minimum cost under the cap; ties keep the first ascending threshold."""
    if not 0 <= cap <= 1:
        raise ValueError('cap must be in [0, 1]')
    sw = threshold_sweep(y, p)
    ok = sw[sw.flag_rate <= cap]
    if ok.empty:
        raise ValueError(f'No threshold satisfies flag rate <= {cap}')
    return float(ok.loc[ok.cost_per_100.idxmin(), 'threshold'])


def plot_sweep(sw, t_chosen, path, cap=FLAG_RATE_CAP):
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    ax.plot(sw.threshold, sw.recall, label='Recall')
    ax.plot(sw.threshold, sw.flag_rate, label='Flag rate')
    ax.axhline(cap, ls='--', color='gray', label=f'Flag-rate cap = {cap:.2f}')
    ax.axvline(t_chosen, ls=':', color='k', label=f'Selected threshold = {t_chosen:.2f}')
    ax.set_xlabel('Decision threshold'); ax.set_ylabel('Fraction (validation)')
    ax.set_ylim(0, 1.02); ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.savefig(path, dpi=200, bbox_inches='tight'); plt.close(fig)


def plot_cost(sw, t_chosen, path, cap=FLAG_RATE_CAP):
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    ax.plot(sw.threshold, sw.cost_per_100, label='Cost per 100 (5·FN + FP)')
    feas = sw[sw.flag_rate <= cap + 1e-12]
    if not feas.empty:
        ax.axvspan(feas.threshold.min(), feas.threshold.max(), color='green', alpha=0.1,
                    label=f'Flag rate ≤ {cap:.2f}')
    ax.axvline(t_chosen, ls=':', color='k', label=f'Selected threshold = {t_chosen:.2f}')
    ax.set_xlabel('Decision threshold'); ax.set_ylabel('Cost per 100 (validation)')
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.savefig(path, dpi=200, bbox_inches='tight'); plt.close(fig)


def operating_constraint_tables(y, p, cap=FLAG_RATE_CAP, alt_caps=ALT_CAPS):
    sw = threshold_sweep(y, p)
    free = sw.loc[sw.cost_per_100.idxmin()]                      # không ràng buộc
    t_c = choose_threshold(y, p, cap); con = sw.loc[sw.threshold == t_c].iloc[0]
    summary = pd.DataFrame([dict(rule='unconstrained minimum cost', **free.to_dict()),
                            dict(rule=f'flag-rate cap {cap:.2f}', **con.to_dict())])
    rows = []
    for c in alt_caps:                                           # chỉ ngưỡng đã quan sát trên validation
        try:
            t = choose_threshold(y, p, c); r = sw.loc[sw.threshold == t].iloc[0]
            rows.append(dict(cap=c, **r.to_dict()))
        except ValueError:
            rows.append(dict(cap=c, threshold=np.nan, flag_rate=np.nan, recall=np.nan, cost_per_100=np.nan))
    return sw, summary, pd.DataFrame(rows)


def run_threshold_analysis(model, t_model, va):
    """Giai đoạn develop: chỉ validation. Tạo bảng và 2 hình cho báo cáo."""
    import evaluate
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    va = va.reset_index(drop=True)
    y = va[TARGET].to_numpy(); p = model.predict_proba(va[FEATURES])[:, 1]
    selected_threshold = choose_threshold(y, p)
    if not np.isclose(t_model, selected_threshold):
        raise ValueError(
            f't_model ({t_model}) differs from threshold.py selection '
            f'({selected_threshold}); use the same validation rule.'
        )
    sw, summary, alt = operating_constraint_tables(y, p)
    sw.to_csv(ROOT / 'validation_threshold_sweep.csv', index=False)
    summary.to_csv(ROOT / 'validation_operating_constraint.csv', index=False)
    alt.to_csv(ROOT / 'validation_alt_caps.csv', index=False)
    at05 = pd.DataFrame([evaluate.metrics_row('final_model', 'validation', y, p, 0.5),
                         evaluate.metrics_row('final_model', 'validation', y, p, t_model)])
    at05.to_csv(ROOT / 'validation_threshold_0.5_vs_selected.csv', index=False)
    plot_sweep(sw, t_model, FIG_DIR / 'validation_recall_flagrate_vs_threshold.png')
    plot_cost(sw, t_model, FIG_DIR / 'validation_cost_vs_threshold.png')
    return summary, alt, at05