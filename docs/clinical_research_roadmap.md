# Open-KBP Dose + Uncertainty Prediction: Research Roadmap Toward Clinical Use

This document summarizes the current baseline status, relevant research directions, and recommended next steps for extending the Open-KBP dose prediction project toward a clinically meaningful dose + uncertainty prediction system.

## 1. Current Project Status

The current baseline pipeline uses the Open-KBP dataset for head-and-neck radiotherapy dose prediction.

Current data split:

- Training: 200 patients
- Validation: 40 patients
- Test: 100 patients

Current model input:

- CT image: `(128, 128, 128, 1)`
- Structure masks: `(128, 128, 128, 10)`
  - OARs: `Brainstem`, `SpinalCord`, `RightParotid`, `LeftParotid`, `Esophagus`, `Larynx`, `Mandible`
  - Targets: `PTV56`, `PTV63`, `PTV70`

Current model output:

- 3D dose distribution: `(128, 128, 128, 1)`

Current baseline model:

- Simplified 3D U-Net from the Open-KBP starter code
- Current starter configuration uses a very small model capacity, with `initial_number_of_filters=1`
- Loss: voxel-wise mean absolute error
- Prediction output is masked by `possible_dose_mask` after inference

Current validation results:

| Model | Training data | Validation data | Dose score | DVH score |
|---|---:|---:|---:|---:|
| baseline_full_20 | full training set | 40 validation patients | 15.526 | 28.913 |
| baseline_full_50 | full training set | 40 validation patients | 8.451 | 10.152 |
| baseline_full_100 | full training set | 40 validation patients | 7.962 | 10.666 |
| baseline_full_150 | full training set | 40 validation patients | 8.555 | 13.247 |

Important observations:

- Epoch 100 gives the best voxel-level dose score.
- Epoch 50 gives the best DVH score.
- Epoch 150 degrades on both dose and DVH metrics.
- Dose score and DVH score are not fully aligned, so future model selection should not rely on dose MAE alone.

Patient-level dose MAE comparison between epoch 50 and epoch 100:

- `baseline_full_50`: mean 8.451, min 5.097, max 13.840
- `baseline_full_100`: mean 7.962, min 4.418, max 13.112
- Epoch 100 improves dose MAE for 33/40 validation patients.
- Epoch 100 worsens dose MAE for 7/40 validation patients.
- Largest improvement cases: `pt_210`, `pt_203`, `pt_227`, `pt_223`, `pt_204`
- Largest degradation cases: `pt_230`, `pt_225`, `pt_240`, `pt_233`, `pt_218`

Interpretation:

- Epoch 100 is generally better at voxel-level dose prediction.
- Epoch 50 is still clinically interesting because its DVH score is better.
- `pt_210` is a useful improvement case.
- `pt_230` is a useful failure/degradation case.

## 2. What Is Still Missing for Clinical Use

The current project is a research prototype. It is not yet close to clinical trial or clinical deployment level.

Major missing components:

1. Stronger model architecture
2. Clinically meaningful loss functions
3. Robust uncertainty quantification
4. External validation on institutional or multi-center data
5. Clinical criteria evaluation beyond Open-KBP dose/DVH score
6. Failure-case detection and safety analysis
7. Integration with treatment plan optimization or dose mimicking
8. Reproducible model versioning and QA workflow

The most important conceptual limitation is that a predicted dose map is not yet a deliverable radiotherapy treatment plan. Clinical treatment requires machine-deliverable plans, beam geometry, fluence or MLC constraints, and physical dose calculation.

Therefore, a clinically relevant future pipeline should look like:

```text
CT + structure masks
        ↓
dose prediction + uncertainty prediction
        ↓
dose mimicking / inverse planning
        ↓
deliverable treatment plan
        ↓
clinical criteria evaluation and QA
```

## 3. Relevant Literature and Ideas to Borrow

### 3.1 OpenKBP Grand Challenge

Reference:

- Babier et al., "OpenKBP: The open-access knowledge-based planning grand challenge"
- https://arxiv.org/abs/2011.14076

Useful ideas:

- OpenKBP provides a fair benchmark for dose prediction.
- Evaluation uses both dose score and DVH score.
- Top teams often used generalizable techniques such as ensembles.

Relevance to this project:

- Continue using OpenKBP as the benchmark.
- Always report both dose score and DVH score.
- Do not rely only on voxel-level MAE.

### 3.2 OpenKBP-Opt

Reference:

- Babier et al., "OpenKBP-Opt: An international and reproducible evaluation of 76 knowledge-based planning pipelines"
- https://arxiv.org/abs/2202.08303

Useful ideas:

- Dose prediction alone is not the full automated planning pipeline.
- Predicted dose can be passed into optimization models.
- Dose mimicking and inverse planning can convert predictions into plans.
- Prediction quality and final plan quality are correlated, but not identical.

Relevance to this project:

- In the thesis, clearly separate dose prediction from clinical treatment planning.
- Future clinical application should include an optimization stage.
- A strong predicted dose distribution is useful, but clinical usefulness must eventually be evaluated after optimization.

### 3.3 Deep Learning Dose Prediction for Plan QA

Reference:

- Gronberg et al., "Deep Learning-Based Dose Prediction for Automated, Individualized Quality Assurance of Head and Neck Radiation Therapy Plans"
- https://arxiv.org/abs/2209.14277

Useful ideas:

- Dose prediction can be used not only to generate plans, but also to assess plan quality.
- Predicted dose can act as a patient-specific reference for detecting suboptimal clinical plans.
- Clinical evaluation used target and OAR dose metrics, not only voxel-wise error.

Relevance to this project:

- A realistic near-term clinical use case is plan QA rather than fully automatic treatment planning.
- Uncertainty maps could help flag unreliable predictions or questionable regions.
- Patient-level and structure-level metrics should be added.

### 3.4 Deep Evidential Learning for Dose Uncertainty

Reference:

- Tan et al., "Deep Evidential Learning for Radiotherapy Dose Prediction"
- https://arxiv.org/abs/2404.17126

Useful ideas:

- Evidential learning can produce uncertainty estimates for dose prediction.
- Epistemic uncertainty was reported to correlate with prediction errors.
- Aleatoric uncertainty reacted more to CT noise.
- The paper also discusses confidence intervals for predicted DVHs.

Relevance to this project:

- This is directly aligned with the thesis topic: dose + uncertainty prediction.
- It provides a stronger uncertainty direction after an MC dropout prototype.
- DVH confidence intervals are clinically more interpretable than only voxel-wise uncertainty maps.

### 3.5 Distance-Aware Diffusion Models

Reference:

- Zhang et al., "DoseDiff: Distance-aware Diffusion Model for Dose Prediction in Radiotherapy"
- https://arxiv.org/abs/2306.16324

Useful ideas:

- Binary masks alone may be insufficient because radiotherapy dose strongly depends on distance to targets and OARs.
- Signed distance maps can encode distance from structures.
- Diffusion models may reduce over-smoothing and improve visual dose quality.

Relevance to this project:

- Signed distance maps are a practical improvement even without adopting a full diffusion model.
- Distance-to-PTV and distance-to-OAR channels can be added to the current U-Net input.
- Diffusion is interesting but likely too large as the immediate next MSc step.

### 3.6 Transformer or Hybrid Architectures

Reference:

- Gheshlaghi et al., "A Cascade Transformer-based Model for 3D Dose Distribution Prediction in Head and Neck Cancer Radiotherapy"
- https://arxiv.org/abs/2307.12005

Useful ideas:

- Hybrid transformer/convolutional architectures can capture broader spatial context.
- Multi-task or cascade designs can connect structure understanding with dose prediction.

Relevance to this project:

- Transformer-based architectures are promising, but may be too complex for the immediate next step.
- A more practical route is first to upgrade the current model to a stronger 3D U-Net or ResUNet.

## 4. Recommended Model Improvements

### 4.1 Increase Baseline Model Capacity

The current model uses `initial_number_of_filters=1`, which is too small for serious dose prediction.

Recommended experiments:

- `initial_number_of_filters=8`
- `initial_number_of_filters=16`
- `initial_number_of_filters=32`, if GPU memory allows

Expected benefit:

- Better representation of anatomical and dose patterns
- Lower dose MAE
- Potentially improved DVH metrics

Risks:

- More GPU memory use
- Longer training
- More overfitting risk

### 4.2 Move Toward 3D ResUNet or Dense Dilated U-Net

Recommended architecture changes:

- Add residual blocks
- Add normalization consistently
- Keep skip connections
- Consider dilated convolutions for larger receptive field

Reason:

- Dose depends on both local anatomy and broader spatial context.
- A stronger U-Net-like architecture is easier to justify and debug than jumping directly to diffusion or transformer models.

### 4.3 Add Distance Transform Channels

Recommended additional input channels:

- Signed distance to PTV70
- Signed distance to PTV63
- Signed distance to PTV56
- Distance to selected OARs, especially spinal cord and brainstem

Reason:

- Dose fall-off is strongly distance-dependent.
- Distance maps provide smoother spatial information than binary masks.

Practical first version:

- Add unsigned or signed distance maps for the three PTV masks.
- Compare against the same U-Net without distance maps.

## 5. Recommended Loss and Training Strategy

### 5.1 Masked MAE

Current training uses ordinary MAE. A better first step is masked MAE inside `possible_dose_mask`.

Recommended loss:

```text
masked_mae = sum(abs(pred - true) * possible_dose_mask) / sum(possible_dose_mask)
```

Reason:

- Evaluation dose score is computed inside the possible dose region.
- Training should match the evaluation target more directly.

### 5.2 Structure-Weighted Loss

Recommended extension:

```text
loss = masked_mae
     + lambda_ptv * PTV_error
     + lambda_oar * OAR_overdose_error
```

Reason:

- PTV underdose and OAR overdose are more clinically important than uniform voxel error.

Suggested first implementation:

- Higher weight for PTV70/PTV63/PTV56 voxels.
- Additional penalty when OAR predicted dose exceeds true dose or clinical thresholds.

### 5.3 DVH-Aware Model Selection

Do not select the best checkpoint using dose MAE only.

Recommended saved checkpoints:

- Best dose score model
- Best DVH score model
- Best combined score model

Potential combined score:

```text
combined_score = dose_score + alpha * dvh_score
```

The value of `alpha` should be chosen transparently and reported.

### 5.4 Early Stopping

The current results show epoch 150 is worse than epoch 50/100. Future training should use validation-based early stopping.

Recommended early stopping signals:

- Validation dose score
- Validation DVH score
- Combined score

## 6. Recommended Uncertainty Roadmap

### 6.1 Phase 1: MC Dropout Prototype

Goal:

- Produce a first uncertainty map with minimal code changes.

Method:

- Keep dropout active during inference.
- Run repeated stochastic predictions:

```text
T = 20 or 30
prediction_1, prediction_2, ..., prediction_T
```

Outputs:

- Mean predicted dose
- Voxel-wise standard deviation
- Absolute error map
- Uncertainty-error correlation

Initial patients:

- `pt_224`: current visualization patient
- `pt_210`: epoch 100 strongly improves over epoch 50
- `pt_230`: epoch 100 strongly degrades versus epoch 50

Evaluation:

- Spatial correlation between uncertainty and absolute error
- Patient-level correlation between mean uncertainty and dose MAE
- Whether high uncertainty highlights failure cases

### 6.2 Phase 2: Deep Ensembles

Goal:

- More reliable epistemic uncertainty estimate.

Method:

- Train multiple models with different random seeds.
- Average predictions and compute variance.

Pros:

- Often strong and easy to interpret.

Cons:

- Expensive to train.

### 6.3 Phase 3: Deep Evidential Learning

Goal:

- Predict uncertainty in one forward pass.

Why it matters:

- Directly aligned with current radiotherapy uncertainty literature.
- Can separate epistemic and aleatoric uncertainty.
- Can potentially produce DVH confidence intervals.

Suggested thesis contribution:

- Compare MC dropout and evidential uncertainty on Open-KBP validation patients.
- Evaluate whether uncertainty predicts voxel error and patient-level failure.

## 7. Recommended Clinical Evaluation Metrics

In addition to Open-KBP dose score and DVH score, add structure-level metrics.

Targets:

- PTV70 D99, D95, D1
- PTV63 D99, D95, D1
- PTV56 D99, D95, D1

OARs:

- Brainstem D0.1cc
- SpinalCord D0.1cc
- Parotid mean dose
- Esophagus mean dose
- Larynx mean dose
- Mandible D0.1cc

Additional analyses:

- Patient-level dose MAE distribution
- Structure-level error distribution
- Worst-case patient analysis
- Error maps for best and worst patients
- Uncertainty versus error calibration plots

## 8. Recommended Next Steps

### Step 1: Complete Baseline Analysis

Deliverables:

- Generate epoch 50 vs 100 min/max error comparison for:
  - `pt_210`: large improvement case
  - `pt_230`: large degradation case
- Add these plots to the qualitative analysis.
- Summarize why epoch 50 and epoch 100 are both important.

### Step 2: Implement Masked MAE

Deliverables:

- Modify training loss to use `possible_dose_mask`.
- Train a small experiment:
  - same model capacity
  - same epoch schedule
  - compare against current baseline

Evaluation:

- Dose score
- DVH score
- Patient-level dose MAE
- Failure cases

### Step 3: Increase Model Capacity

Deliverables:

- Train models with larger filter counts, starting with 8 or 16.
- Keep the same evaluation scripts.

Evaluation:

- Check whether larger models improve both dose and DVH.
- Watch for overfitting and training instability.

### Step 4: Add MC Dropout Uncertainty

Deliverables:

- Inference script for repeated stochastic predictions.
- Output:
  - mean predicted dose
  - uncertainty standard deviation
  - absolute error
- Visualization for `pt_224`, `pt_210`, `pt_230`.

Evaluation:

- Voxel-wise uncertainty-error correlation
- Patient-level uncertainty-error correlation
- Qualitative maps

### Step 5: Move Toward Clinical-Aware Uncertainty

Deliverables:

- Structure-level uncertainty summaries.
- DVH confidence intervals.
- Failure-case flagging rule.

Possible rule:

```text
Flag patient/structure if predicted uncertainty is high near PTV boundary or OAR high-dose region.
```

## 9. Suggested Thesis Narrative

A coherent MSc thesis narrative could be:

1. Reproduce and understand the Open-KBP baseline.
2. Show that dose score and DVH score do not always agree.
3. Add patient-level and qualitative failure-case analysis.
4. Improve the training objective with masked and structure-aware losses.
5. Extend dose prediction to uncertainty prediction.
6. Evaluate whether uncertainty helps detect unreliable predictions and clinically relevant failure cases.

This story is stronger than only trying many architectures, because it connects prediction accuracy, clinical relevance, and model reliability.

## 10. Immediate Action List

Highest priority:

1. Generate comparison figures for `pt_210` and `pt_230`.
2. Add a baseline analysis section to the existing planning document.
3. Implement masked MAE as the first model improvement.
4. Train a controlled masked-loss experiment.
5. Start MC dropout uncertainty on selected patients.

Do not do yet:

- Do not jump directly to diffusion models.
- Do not attempt clinical claims from Open-KBP alone.
- Do not optimize only for voxel MAE.
- Do not ignore failure cases where DVH or patient-level MAE worsens.

