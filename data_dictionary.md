# Data dictionary

One row represents one simulated 10-second ECG segment. Each of 50 simulated participants contributes 40 rows. All quantities below are simulated directly; there are no underlying waveforms supplied. Neither device code identifies a commercial device.

| Field | Type / unit | Meaning and availability |
|---|---|---|
| segment_id | string | Unique segment identifier; available at acquisition. |
| participant_id | string | Simulated participant identifier; use for grouping and audit. |
| device_id | D1 or D2 | Simulated acquisition-device category; available at acquisition. |
| duration_s | seconds | Segment duration; constant at 10. |
| missing_ratio | fraction, 0–1 | Simulated proportion of missing samples. |
| baseline_variation_mv | mV | Simulated low-frequency baseline variability summary. |
| high_frequency_ratio | fraction, 0–1 | Simulated relative energy for a nominal 40–100 Hz band at a nominal 250 Hz sampling rate. It is a generated proxy, not a calculated spectrum. |
| clipping_ratio | fraction, 0–1 | Simulated proportion of saturated samples. |
| amplitude_scale_mv | mV | Simulated signal-amplitude scale summary. |
| review_note | string | Post-review acquisition decision; written after the quality judgment. |
| quality_label | 0 or 1 | Simulated target: 1 = noisy / review requested; 0 = acceptable. |

`participant_split.json` maps every participant to exactly one split: 30 train, 10 validation, 10 test. Device categories occur across splits. Labels are supplied for all records so students can reproduce final evaluation after freezing their method.

Do not infer clinical thresholds or feature-extraction standards from these synthetic values. The intended deployment question is generalization to unseen participants using the same two simulated device categories. Neither device generalization nor diagnostic performance has been established.
