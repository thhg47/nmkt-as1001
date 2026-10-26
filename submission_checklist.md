# Submission requirements

Submit `ECGQ_TeamXX.zip` containing the following (replace TeamXX with your assigned identifier):

1. `report.pdf`: English, 6–8 pages excluding references and AI-log appendix. Include team IDs, track, problem definition, data audit, method, results, operating constraint, track-specific analysis, limitations, and contributions. Include numbered figures and tables.
2. `analysis.ipynb` with saved outputs **or** `analysis.py` (an equivalent R script is acceptable). It must reproduce the submitted results from the original data.
3. `requirements.txt` with exact versions, or an equivalent environment specification for R.
4. `README.md`: setup instructions, one run command, seed(s), relative input/output paths, and expected outputs. Do not rely on absolute machine-specific paths.
5. `feature_audit.csv`: columns `feature,available_at_prediction,include,reason` for every input field. Do not use the target as a predictor.
6. `test_predictions.csv`: exactly one row per prescribed test segment; columns `segment_id,participant_id,y_true,p_noisy,y_pred,threshold`. Use the selected final model. `p_noisy` must be in [0,1]; prediction rule is `p_noisy >= threshold`.
7. `metrics.csv`: columns `model,split,threshold,n,tp,fp,fn,tn,accuracy,precision,recall,specificity,f1,flag_rate,cost_per_100`. Include baseline and final model on validation and test. Report undefined ratios as NA and explain them. Cost per 100 = `100*(5*fn+fp)/n`.
8. `ai_log.md`: tool/model as displayed, date, task, relevant prompt/excerpt, accepted/rejected/uncertain claims, verification evidence, and changes made. If no AI was used, write `No AI used` and document a peer critique instead.
9. `initial_decision.md`: the group's reasoning before AI consultation; retain it unchanged.
10. `freeze_record.md`: selected features, preprocessing, model configuration, threshold, random seeds, and a timestamp recorded before inspecting test performance.

Also submit `StudentID_reflection.pdf` individually (300–400 words): an initial misconception, a checked claim, a decision you can defend, your contribution, and one remaining limitation. The individual in-class quiz and oral defense are completed without AI.

Do not submit a copied AI response as your report, credentials, private data, or virtual-environment folders. Include any small helper scripts needed to reproduce the work. The instructor will place the original provided data beside the entry-point script before running it; your README must follow that layout.

Before submission, rerun in a clean session. Verify prediction IDs match the 400 test segments exactly, all probabilities are bounded, and reported confusion counts sum to 400. The manual 100-segment exercise is separate from the 2,000-row practical dataset.
