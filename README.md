# ECG Quality Assessment — Student Data Package

All material is in English. Read `ecg_quality_student.tex` for the lesson and assessment brief. No worked solutions are included.

## Files
- `ecg_quality_features.jsonl`: 2,000 records, one JSON object per line.
- `participant_split.json`: prescribed participant-level train/validation/test assignment.
- `data_dictionary.md`: variable definitions, provenance, and limitations.
- `starter.py`: standard-library loader only; no completed analysis.
- `submission_checklist.md`: required hand-in structure and file schemas.
- `ecg_quality_student.tex`: standalone student handout; no external assets required.

## Getting started
Keep all files in one directory and run `python starter.py`. For analysis, use Python 3 with your chosen scientific libraries (for example NumPy, pandas, scikit-learn, and matplotlib), or an equivalent reproducible R workflow. Record the exact versions you actually use. No particular paid AI service is required.

To load a table with pandas, use `pd.read_json('ecg_quality_features.jsonl', lines=True)`. The prescribed split is supplied separately; join it by participant ID. Do not replace it with a random row split for the primary evaluation.

## Model training and evaluation
Install the pinned dependencies with `python -m pip install -r requirements.txt`.

Run `python analysis.py --stage develop` to fit the baseline, logistic-regression candidates, and random forest on train; choose the model and threshold using validation only; and write the frozen model plus `results/freeze_record.md`. This also writes validation threshold-sweep/operating-constraint CSVs and threshold plots. Review and retain that freeze record before test evaluation.

Only after the model and threshold are frozen, run `python analysis.py --stage final` once from this directory. This evaluates the frozen models on test and writes `metrics.csv`, `test_predictions.csv`, `bootstrap_ci.csv`, `by_device.csv`, `by_participant.csv`, `calibration.csv`, `naive_costs.csv`, and `figures/confusion_test.png`. The final stage refuses to overwrite an existing `test_predictions.csv`.

The repository also retains `model_training.py`, which offers the equivalent `python model_training.py`, `python model_training.py --freeze`, then `python model_training.py --final` workflow. Use one workflow consistently; do not mix the freeze artifacts from the two entry points.

## Provenance and limits
Generated for this assignment with a fixed seed (20261022). These are synthetic feature-level observations, not patient records and not features extracted from actual ECG waveforms. Participant effects, device offsets, overlapping class distributions, and stochastic labels were simulated. The labels are simulated quality judgments, not clinical or expert-validated ground truth. The post-review note is included for the feature-eligibility audit. There are no raw signals: filtering, waveform morphology analysis, and clinical conclusions are outside the evidence supplied.

Test labels are visible for a self-contained teaching exercise. Treat them as sealed until the method and threshold are frozen. This is a procedural holdout, not a secure hidden benchmark.

## Timing and submission
150-minute workshop plus an estimated 4–6 hours of independent work. Submit one team archive (2–4 members) and one individual reflection per student. The instructor announces the deadline and LMS destination. The absence of a deadline in this package is not an extension.
