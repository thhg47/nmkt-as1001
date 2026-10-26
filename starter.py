"""Data loading only. Implement the assessed analysis yourself.
Run: python starter.py
Uses only the Python standard library.
"""
from pathlib import Path
import json

DATA_DIR = Path(__file__).resolve().parent

def load_data(data_dir=DATA_DIR):
    with (data_dir / 'ecg_quality_features.jsonl').open(encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]
    with (data_dir / 'participant_split.json').open(encoding='utf-8') as f:
        split_map = json.load(f)
    groups = {name: [] for name in ('train', 'validation', 'test')}
    for record in records:
        groups[split_map[record['participant_id']]].append(record)
    return groups

if __name__ == '__main__':
    data = load_data()
    for name, records in data.items():
        print(name, len(records), 'segments',
              len({r['participant_id'] for r in records}), 'participants')
    # TODO: Select defensible predictors; document feature eligibility.
    # TODO: Fit preprocessing and models using training data only.
    # TODO: Select hyperparameters and thresholds using validation data.
    # TODO: Freeze the analysis plan before final test evaluation.
    # TODO: Export the required report, predictions, and reproducibility files.
