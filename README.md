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

`analysis.py` is the single supported training/evaluation entry point. The older `model_training.py` and separately supplied `model_training-4.py` are retained as reference drafts only; they are not supported workflows. The Logistic Regression configuration from the latter (`max_iter=2000`) is applied in `analysis.py`.

Run the following commands from this directory:

1. `python analysis.py --stage develop` fits candidates on train and selects the model/threshold using validation only. It writes validation reports and plots; it does not create or overwrite freeze artifacts.
2. After the group reviews the validation results and agrees to freeze, run `python analysis.py --stage freeze --approve-freeze`. This writes the model artifact and `freeze_record.md` plus its JSON companion in the project root. The operation refuses to overwrite any existing freeze artifact.
3. Only after that freeze, run `python analysis.py --stage final` once. It writes `metrics.csv`, `test_predictions.csv`, `bootstrap_ci.csv`, `by_device.csv`, `by_participant.csv`, `calibration.csv`, `naive_costs.csv`, and `figures/confusion_test.png` in the project root. It refuses to run when there is no approved freeze or when any final output already exists.

All generated paths are relative to the script's location, regardless of the shell's current directory. The random seed is `20261022`. Required environment versions are pinned in `requirements.txt`.

The existing files under `results/` named `freeze_record.*` and `frozen_models.joblib` are legacy, unapproved draft artifacts. They are retained for provenance but are not accepted by the final stage; only a new root-level freeze created by the explicit approval command is valid.

`feature_audit.csv` is included at the required path. The group must supply its own `initial_decision.md` unchanged and complete the report and AI log using its actual decisions and AI-use record; these cannot be truthfully generated from the code alone.

## Provenance and limits
Generated for this assignment with a fixed seed (20261022). These are synthetic feature-level observations, not patient records and not features extracted from actual ECG waveforms. Participant effects, device offsets, overlapping class distributions, and stochastic labels were simulated. The labels are simulated quality judgments, not clinical or expert-validated ground truth. The post-review note is included for the feature-eligibility audit. There are no raw signals: filtering, waveform morphology analysis, and clinical conclusions are outside the evidence supplied.

Test labels are visible for a self-contained teaching exercise. Treat them as sealed until the method and threshold are frozen. This is a procedural holdout, not a secure hidden benchmark.

## Timing and submission
150-minute workshop plus an estimated 4–6 hours of independent work. Submit one team archive (2–4 members) and one individual reflection per student. The instructor announces the deadline and LMS destination. The absence of a deadline in this package is not an extension.
