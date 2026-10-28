# Initial Decision: ECG Quality Assessment Model Development

**Task:** ECG Quality Assessment Model Development  
**Date:** October 2026  

---

## 1. Initial Decision -  LE NGUYEN DUC HUNG ID 2652685

### **1. Critique of Proposed Workflow & Additional Flaws**
1. **Standardization Leakage:** Scaling data across all 2,000 segments uses global statistics, leaking test distribution into training and causing optimistic bias.
2. **Participant Leakage:** Random segment-level splitting places the same participant in train and test sets, evaluating subject memorization rather than generalization to new participants.
3. **Target Leakage:** Including `review_note` (post-eval note) and `participant_id` introduces future outcome data unavailable at inference time.
4. **Test Set Tuning & Metric Misalignment:** Tuning 20 models on the test set causes snooping bias. Accuracy also fails to penalize high-cost False Negatives.
5. **Imputation Before Splitting (Data Leakage):** Imputing missing values using the mean/median of the full dataset before splitting leaks test values into training.
6. **Synthetic Oversampling (SMOTE) Across Full Data:** Applying SMOTE or noise augmentation on the entire dataset creates synthetic test samples that overlap with train samples.
7. **Ignoring Device-Specific Shift:** Evaluating models without stratifying or accounting for `device_id` (D1 vs D2) risks poor generalization across different hardware setups.

### **2. Revised Workflow Strategy**
1. **Feature Audit:** Exclude `review_note`, `participant_id`, and `duration_s`.
2. **Group Split:** Partition data by participant ID (`participant_split.json`) into Train (30), Val (10), and Test (10).
3. **Train-Only Preprocessing:** Fit `SimpleImputer` and `StandardScaler` strictly on Train; transform Val and Test independently.
4. **Validation Tuning:** Train baseline model on Train and tune decision threshold on Val to minimize:
   $$\text{Cost per 100} = \frac{100 \times (5\text{FN} + \text{FP})}{N}$$
5. **Freeze Methodology:** Lock hyperparameters, scalers, and threshold in `freeze_record.md` before testing.
6. **Final Evaluation:** Run one-pass inference on Test to produce `metrics.csv` and `test_predictions.csv`.

---

## 2. Initial Decision - NGUYEN DUC MINH ID 2652458

### **1. Critique of Proposed Workflow & Additional Flaws**
1. **Probability Calibration Drift:** Using raw model scores as probabilities without calibration, leading to suboptimal thresholds on the test set.
2. **Class Imbalance Neglect:** Training on imbalanced data without class weights, causing the model to bias toward the majority class.
3. **Hardware-Specific Feature Bias:** Ignoring feature variance across different hardware setups (e.g., D1 vs. D2), risking poor generalization.
4. **Hyperparameter Snooping:** Tuning multiple configurations on the validation set without complexity penalties, leading to optimistic bias.
5. **Metric Disconnection:** Optimizing generic metrics (Accuracy/AUC) instead of the actual business cost function (5:1).
6. **Feature Scale Discrepancy:** Combining features with vastly different scales without normalization, disproportionately weighting larger values.
7. **Unlogged Thresholds:** Manually adjusting thresholds without recording history, preventing reproducibility and auditing.

### **2. Revised Workflow Strategy**
1. **Class-Aware Weighting:** Integrate `class_weight='balanced'` to handle label imbalance from the start.
2. **Hardware Stratification:** Audit and control data distribution by `device_id` before splitting.
3. **Probability Calibration:** Calibrate prediction probabilities on the validation set prior to threshold sweeping.
4. **Direct Cost Optimization:** Search for optimal thresholds directly using the business cost function ($5\text{FN} + \text{FP}$).
5. **Strict Artifact Freezing:** Lock all weights, scalers, and optimal thresholds into `freeze_record.md`.
6. **One-Shot Test Execution:** Run inference exactly once on the test set to output `metrics.csv` and `test_predictions.csv`.

---

## 3. Initial Decision - TA NGUYEN MINH DANG ID 2652430

### **1. Critique of Proposed Workflow & Potential Flaws:**
1. **Standardization Leakage:** Scaling data across all segments using global statistics leaks test distributions into training, causing optimistic performance bias.
2. **Participant Leakage:** Random segment-level splitting places segments from the same participant into both train and test sets, evaluating subject memorization rather than true generalization to unseen participants.
3. **Target Leakage:** Including post-evaluation notes (`review_note`) and target labels introduces future outcome data that is unavailable at real-time inference.
4. **Test Set Tuning & Metric Misalignment:** Tuning hyperparameters or thresholds on the test set leads to snooping bias, while standard accuracy fails to penalize high-cost False Negatives.
5. **Imputation Before Splitting (Data Leakage):** Imputing missing values using global mean/median statistics before splitting leaks test values into the training phase.
6. **Synthetic Oversampling (SMOTE) Across Full Data:** Applying SMOTE or data augmentation on the entire dataset before splitting causes synthetic test samples to overlap with training instances.
7. **Ignoring Device-Specific Shift:** Evaluating models without accounting for hardware variations across different devices (`device_id`) risks poor cross-device generalization.

### **2. Revised Workflow Strategy:**
1. **Participant-Level Split:** Partition data strictly by participant ID into Train, Validation, and Test sets to prevent data leakage.
2. **Train-Only Preprocessing:** Fit `SimpleImputer` and `StandardScaler` strictly on the Train set; transform Validation and Test sets independently.
3. **Feature Filtering:** Exclude `review_note`, `participant_id`, `duration_s`, and target labels to avoid leakage and inference-time errors.
4. **Direct Cost Optimization:** Search for optimal thresholds on the validation set directly using the business cost function ($5\text{FN} + \text{FP}$) while keeping the noise flag rate $\le 0.20$.
5. **Strict Artifact Freezing:** Lock all models, scalers, and optimal thresholds into `freeze_record.md`.
6. **One-Shot Test Execution:** Run inference exactly once on the test set to output `metrics.csv` and `test_predictions.csv`.

---

## 4. Initial Decision - PHAM THANH HANG ID 2652680

### **1. Key Questions & Answers:**
* **What does this assignment ask to predict?** It asks to predict whether an ECG segment is noisy (label 1 for noisy/review requested, label 0 for acceptable). This is signal quality assessment, not disease diagnosis.
* **Which columns to use as features, which to drop, and why?** Only use features that are available at prediction time. You must drop the target variable, identifiers, constant fields, and post-review information.
* **How to split data into train/validation/test?** Data is split by participants (based on `participant_split.json`): 30 participants for training (1,200 segments), 10 for validation (400 segments), and 10 for test (400 segments).
* **What models to use, and what is the baseline?** Use a simple class-prior predictor as the baseline, and an interpretable candidate model like logistic regression or a shallow decision tree.
* **How to choose the threshold? Are FN and FP equally costly?** FN and FP are not equally costly; missing a noisy segment (FN) is heavier than a false positive. The hypothetical cost is $C = 5\text{FN} + \text{FP}$. Choose the threshold using validation data while considering cost and a flag-rate limit of 0.20.
* **What remains uncertain?** The labels are simulated judgments rather than clinically validated ground truth, and a threshold meeting validation constraints is not guaranteed to meet them on the test data.

---

## 3. Initial Decision - Pham Van Bao Phong ID 2652465

### **1. Critique of Proposed Workflow & Potential Flaws:**
1. **Standardization & Preprocessing Leakage:** Applying global scalers and imputers across the entire dataset before splitting introduces test distribution leaks into training, resulting in optimistic evaluation bias.
2. **Participant Overfitting:** Random segment-level partitioning fails to evaluate true generalization because segments from the same individual appear in both training and testing folds.
3. **Target & Post-Evaluation Leakage:** Including post-review annotations (`review_note`) and target variables creates illegal shortcuts using future data unavailable at inference time.
4. **Metric Disalignment & Cost Asymmetry:** Utilizing standard accuracy ignores the 5-to-1 asymmetric business penalty ($C_{10} = 5\text{FN} + \text{FP}$) associated with missing noisy ECG records.
5. **Operational Constraint Violations:** Unchecked probability thresholds can breach the strict operational mandate that limits noise flag rates to a maximum of 20% ($\le 0.20$).
6. **Hardware Variation Neglect:** Disregarding device-specific dynamics (`device_id` between $D1$ and $D2$) risks performance degradation across heterogeneous recording hardware.

### **2. Revised Workflow Strategy:**
1. **Participant-Level Partitioning:** Split records strictly by participant IDs using `participant_split.json` into Train (30), Validation (10), and Test (10).
2. **Strict Train-Only Transformations:** Fit `SimpleImputer` and `StandardScaler` exclusively on the training subset; transform validation and test folds independently.
3. **Feature Selection & Sanitization:** Exclude `review_note`, `participant_id`, `duration_s`, and target labels from feature matrices.
4. **Validation-Gated Threshold Sweeping:** Optimize decision thresholds solely on the validation set using the asymmetric cost function while enforcing the $\le 20\%$ flag-rate cap.
5. **Artifact Freezing:** Lock optimal hyperparameters, scalers, and decision thresholds into `freeze_record.md`.
6. **One-Shot Test Inference:** Execute final evaluation exactly once on the test dataset to generate `metrics.csv` and `test_predictions.csv`.
