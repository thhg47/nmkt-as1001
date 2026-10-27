from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import SEED, FEATURES, TARGET, cost_per_100


METRIC_COLS = ['model','split','threshold','n','tp','fp','fn','tn','accuracy',
               'precision','recall','specificity','f1','flag_rate','cost_per_100']
FIG_DIR = Path(__file__).resolve().parent / 'figures'


# ---------- 1. Chỉ số ----------
def counts(y, pred):
    y = np.asarray(y).astype(int); pred = np.asarray(pred).astype(bool)
    tp = int(((y==1) & pred).sum());  fp = int(((y==0) & pred).sum())
    fn = int(((y==1) & ~pred).sum()); tn = int(((y==0) & ~pred).sum())
    return tp, fp, fn, tn

def metrics_row(model, split, y, p, t):
    y = np.asarray(y)
    p = np.asarray(p)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError('y and p must be one-dimensional arrays of equal length')
    if len(y) == 0:
        raise ValueError('metrics_row requires at least one observation')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('y must contain only binary labels 0 and 1')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('p must contain finite probabilities in [0, 1]')
    if not 0 <= t <= 1:
        raise ValueError('t must be a threshold in [0, 1]')
    pred = p >= t
    tp, fp, fn, tn = counts(y, pred)
    n = len(y)
    if tp + fp + fn + tn != n:
        raise ValueError('confusion counts do not sum to the number of observations')
    div = lambda a, b: a / b if b > 0 else 'NA'
    f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 'NA'
    return dict(model=model, split=split, threshold=t, n=n, tp=tp, fp=fp, fn=fn, tn=tn,
                accuracy=(tp+tn)/n, precision=div(tp, tp+fp), recall=div(tp, tp+fn),
                specificity=div(tn, tn+fp), f1=f1, flag_rate=(tp+fp)/n,
                cost_per_100=100*(5*fn + fp)/n)


# ---------- 2. Bootstrap theo người ----------
def participant_bootstrap(df, scores, B=2000, seed=SEED):
    """scores: {tên: (p, t)}. Cùng mẫu bootstrap cho mọi mô hình (so sánh có cặp)."""
    if B <= 0:
        raise ValueError('B must be greater than zero')
    y = df[TARGET].to_numpy(); pid = df['participant_id'].to_numpy()
    if len(y) == 0:
        raise ValueError('participant_bootstrap requires at least one observation')
    if any(len(p) != len(y) for p, _ in scores.values()):
        raise ValueError('Each score array must match the number of rows in df')
    people = np.unique(pid)
    idx_by = {k: np.where(pid == k)[0] for k in people}
    rng = np.random.default_rng(seed)
    out = {k: [] for k in scores}
    for _ in range(B):
        chosen = rng.choice(people, size=len(people), replace=True)
        idx = np.concatenate([idx_by[k] for k in chosen])
        for name, (p, t) in scores.items():
            out[name].append(cost_per_100(y[idx], np.asarray(p)[idx], t))
    return {k: np.array(v) for k, v in out.items()}


# ---------- 3. Phân tích theo nhóm ----------
def group_table(df, p, t, by, model_name):
    df = df.reset_index(drop=True)
    p = np.asarray(p)
    if len(p) != len(df):
        raise ValueError('p must match the number of rows in df')
    rows = []
    for g, d in df.groupby(by):
        r = metrics_row(model_name, str(g), d[TARGET], p[d.index], t)
        r['n_participants'] = d['participant_id'].nunique()
        r['prevalence'] = d[TARGET].mean()
        rows.append(r)
    return pd.DataFrame(rows).rename(columns={'split': by})

def person_table(df, p, t):
    d = df.assign(pred=(np.asarray(p) >= t).astype(int))
    g = d.groupby('participant_id')
    out = pd.DataFrame({
        'devices': g['device_id'].agg(lambda s: ','.join(sorted(s.unique()))),
        'n': g.size(),
        'n_noisy': g[TARGET].sum(),
        'fn': g.apply(lambda x: int(((x[TARGET]==1) & (x['pred']==0)).sum())),
        'fp': g.apply(lambda x: int(((x[TARGET]==0) & (x['pred']==1)).sum())),
    })
    out['cost_per_100'] = 100*(5*out['fn'] + out['fp'])/out['n']
    return out.sort_values('cost_per_100', ascending=False)

def naive_costs(df):
    y = df[TARGET].to_numpy()
    return {'flag_none': 100*5*y.sum()/len(y),
            'flag_all':  100*(len(y)-y.sum())/len(y)}

def calibration_table(y, p, bins=(0, .1, .2, .3, .5, .7, 1.0001)):
    d = pd.DataFrame({'y': y, 'p': p})
    d['bin'] = pd.cut(d['p'], bins=bins, right=False)
    return d.groupby('bin', observed=True).agg(n=('y','size'),
                                               mean_pred=('p','mean'), observed=('y','mean'))


# ---------- 4. Hình ----------
def plot_confusion(y, p, t, path):
    tp, fp, fn, tn = counts(y, np.asarray(p) >= t)
    m = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(m, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, m[i, j], ha='center', va='center')
    ax.set_xticks([0, 1], ['pred 0', 'pred 1']); ax.set_yticks([0, 1], ['true 0', 'true 1'])
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# ---------- 5. Kiểm tra bắt buộc ----------
def check_test_predictions(tp_df, test_df):
    assert len(tp_df) == 400 and tp_df['segment_id'].is_unique
    assert set(tp_df['segment_id']) == set(test_df['segment_id'])
    assert tp_df['p_noisy'].between(0, 1).all()
    assert ((tp_df['p_noisy'] >= tp_df['threshold']).astype(int) == tp_df['y_pred']).all()

def check_metrics(m):
    assert (m[['tp','fp','fn','tn']].sum(axis=1) == m['n']).all()


# ---------- 6. Điểm vào ----------
def run_validation(base, model, t_base, t_model, va):
    """Giai đoạn develop: CHỈ dùng validation. Thử code trước khi đóng băng."""
    va = va.reset_index(drop=True)
    pb = base.predict_proba(va[FEATURES])[:, 1]
    pm = model.predict_proba(va[FEATURES])[:, 1]
    m = pd.DataFrame([metrics_row('baseline', 'validation', va[TARGET], pb, t_base),
                      metrics_row('final_model', 'validation', va[TARGET], pm, t_model)])[METRIC_COLS]
    check_metrics(m)
    print(m.to_string(index=False))
    boot = participant_bootstrap(va, {'baseline': (pb, t_base), 'final_model': (pm, t_model)}, B=300)
    print({k: np.percentile(v, [2.5, 97.5]).round(1) for k, v in boot.items()})
    print(group_table(va, pm, t_model, 'device_id', 'final_model'))

def run_final(base, model, t_base, t_model, va, te):
    """Giai đoạn final: CHỈ gọi sau khi freeze-v1 đã có tag. Chạy test đúng một lần."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    va = va.reset_index(drop=True); te = te.reset_index(drop=True)
    pvb, pvm = (m.predict_proba(va[FEATURES])[:, 1] for m in (base, model))
    ptb, ptm = (m.predict_proba(te[FEATURES])[:, 1] for m in (base, model))

    # metrics.csv (4 hàng)
    metrics = pd.DataFrame([
        metrics_row('baseline',    'validation', va[TARGET], pvb, t_base),
        metrics_row('final_model', 'validation', va[TARGET], pvm, t_model),
        metrics_row('baseline',    'test',       te[TARGET], ptb, t_base),
        metrics_row('final_model', 'test',       te[TARGET], ptm, t_model)])[METRIC_COLS]
    check_metrics(metrics)
    metrics.to_csv('metrics.csv', index=False)

    # test_predictions.csv (mô hình cuối)
    tp_df = pd.DataFrame({'segment_id': te['segment_id'], 'participant_id': te['participant_id'],
                          'y_true': te[TARGET], 'p_noisy': ptm,
                          'y_pred': (ptm >= t_model).astype(int), 'threshold': t_model})
    check_test_predictions(tp_df, te)
    tp_df.to_csv('test_predictions.csv', index=False)

    # Phân tích bổ sung
    boot = participant_bootstrap(te, {'baseline': (ptb, t_base), 'final_model': (ptm, t_model)})
    diff = boot['baseline'] - boot['final_model']
    pd.DataFrame({'baseline_lo_hi': np.percentile(boot['baseline'], [2.5, 97.5]),
                  'final_lo_hi':    np.percentile(boot['final_model'], [2.5, 97.5]),
                  'diff_lo_hi':     np.percentile(diff, [2.5, 97.5])},
                 index=['2.5%', '97.5%']).to_csv('bootstrap_ci.csv')
    group_table(te, ptm, t_model, 'device_id', 'final_model').to_csv('by_device.csv', index=False)
    person_table(te, ptm, t_model).to_csv('by_participant.csv')
    calibration_table(te[TARGET].to_numpy(), ptm).to_csv('calibration.csv')
    pd.Series(naive_costs(te)).to_csv('naive_costs.csv', header=['cost_per_100'])
    plot_confusion(te[TARGET].to_numpy(), ptm, t_model, FIG_DIR / 'confusion_test.png')