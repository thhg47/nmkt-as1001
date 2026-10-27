import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

df = pd.read_json('ecg_quality_features.jsonl', lines=True)
split = pd.read_json('participant_split.json', typ='series')
df['split'] = df['participant_id'].map(split)

drop_cols = ['review_note', 'participant_id', 'duration_s', 'split', 'target']
if 'target' in df.columns:
    y = df['target']
elif 'label' in df.columns:
    y = df['label']
else:
    y = df.iloc[:, -1] 

X = df.drop(columns=[c for c in drop_cols if c in df.columns])

train_idx = df['split'] == 'train'
val_idx = df['split'] == 'validation'
test_idx = df['split'] == 'test'

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

model.fit(X_train, y_train)
y_val_prob = model.predict_proba(X_val)[:, 1]

def cost_per_100(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    N = len(y_true)
    if N == 0:
        return 0.0
    return 100 * (5 * fn + fp) / N

def choose_threshold(y_true, y_prob, max_flag_rate=0.20):
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_threshold = 0.5
    min_cost = float('inf')
    
    costs = []
    flag_rates = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        flag_rate = np.mean(y_pred)
        cost = cost_per_100(y_true, y_pred)
    
        costs.append(cost)
        flag_rates.append(flag_rate)
        
        if flag_rate <= max_flag_rate:
            if cost < min_cost:
                min_cost = cost
                best_threshold = t

    print(f"Ngưỡng tối ưu: {best_threshold:.2f} với chi phí: {min_cost:.2f}")
    return best_threshold, thresholds, costs, flag_rates

best_t, thresholds, costs, flag_rates = choose_threshold(y_val, y_val_prob)

plt.figure(figsize=(8, 5))
plt.plot(thresholds, costs, label='Cost per 100 segments', color='navy', lw=2)
plt.axvline(x=best_t, color='crimson', linestyle='--', label=f'Optimal Threshold: {best_t:.2f}')
plt.xlabel('Decision Threshold')
plt.ylabel('Cost Value')
plt.title('Cost Optimization Curve on Validation Set')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig('reports/cost_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("Đã lưu biểu đồ chi phí tại reports/cost_curve.png thành công!")
