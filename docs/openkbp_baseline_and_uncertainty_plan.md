# OpenKBP Baseline Pipeline and Uncertainty Plan

这份文档用于 MSc thesis 项目的当前阶段：先把 OpenKBP baseline 讲清楚，再规划如何从 dose prediction 扩展到 dose + uncertainty prediction。目标读者是假设刚接触医学图像、TensorFlow 和 U-Net 的自己。

## 1. 项目现在在做什么

临床任务是 radiotherapy treatment planning 里的 dose prediction。

简单说：

- CT image 描述患者身体内部的密度/解剖信息。
- Segmentation masks 描述医生或算法标出来的结构区域，例如肿瘤靶区和正常器官。
- 模型输入 CT + masks，输出每个 3D voxel 上预测的 radiation dose。
- 之后要进一步输出 uncertainty map，表示模型对每个 voxel 的 dose prediction 有多不确定。

OpenKBP 数据集是 head-and-neck cancer radiotherapy planning 数据。每个患者都被整理成固定的 3D grid：`128 x 128 x 128`。

## 2. 仓库结构

当前主要目录如下：

```text
open-kbp/
├── main.py
├── main_dryrun.py
├── PROJECT_NOTES.md
├── provided-data/
│   ├── train-pats/
│   ├── validation-pats/
│   └── test-pats/
├── provided_code/
│   ├── data_loader.py
│   ├── data_shapes.py
│   ├── batch.py
│   ├── network_architectures.py
│   ├── network_functions.py
│   ├── dose_evaluation_class.py
│   └── utils.py
├── scripts/
│   └── inspect_patient.py
├── outputs/
│   └── inspect_patient/
└── results/
    ├── baseline/
    ├── baseline_full_20/
    ├── baseline_full_50/
    ├── baseline_full_100/
    ├── baseline_full_150/
    └── submissions/
```

最重要的代码入口：

- `main.py`: 训练模型、预测 validation/test、计算 OpenKBP 指标、打包 submission。
- `provided_code/data_loader.py`: 从 CSV 文件读取 CT、dose、structure masks 等，转换成 TensorFlow 可以吃的 5D tensor。
- `provided_code/network_architectures.py`: 定义 baseline 3D U-Net。
- `provided_code/network_functions.py`: 包装训练、保存模型、加载模型、预测 dose。
- `provided_code/dose_evaluation_class.py`: 计算 dose score 和 DVH score。
- `scripts/inspect_patient.py`: 已写好的患者可视化/数据检查脚本。

## 3. 数据是什么

每个 patient 文件夹里通常包含：

```text
ct.csv
dose.csv
possible_dose_mask.csv
voxel_dimensions.csv
Brainstem.csv
SpinalCord.csv
RightParotid.csv
LeftParotid.csv
Esophagus.csv
Larynx.csv
Mandible.csv
PTV56.csv
PTV63.csv
PTV70.csv
```

这里有两类结构：

- OARs, organs at risk: 正常器官，例如 Brainstem、SpinalCord、Parotid、Esophagus、Larynx、Mandible。治疗时希望这些器官少受剂量。
- Targets / PTVs: 肿瘤治疗靶区，例如 PTV56、PTV63、PTV70。数字大致对应 prescribed dose level，治疗时希望这些区域获得足够剂量。

`provided_code/data_shapes.py` 固定了模型 tensor shape：

| 数据 | Shape | 含义 |
|---|---:|---|
| CT | `(128, 128, 128, 1)` | 一个 3D CT 灰度体 |
| Dose | `(128, 128, 128, 1)` | 每个 voxel 的真实 dose |
| Structure masks | `(128, 128, 128, 10)` | 10 个 ROI mask channel |
| Possible dose mask | `(128, 128, 128, 1)` | 允许 dose 非零的区域 |
| Voxel dimensions | `(3,)` | 每个 voxel 的物理尺寸 |

训练时 batch 维度会加在最前面，所以 CT batch 是：

```text
(batch_size, 128, 128, 128, 1)
```

## 4. DataLoader 做了什么

`DataLoader` 的核心职责是把原始 CSV 变成统一 tensor。

重要模式：

- `training_model`: 加载 `dose`, `ct`, `structure_masks`, `possible_dose_mask`, `voxel_dimensions`。
- `dose_prediction`: 预测时只加载 `ct`, `structure_masks`, `possible_dose_mask`, `voxel_dimensions`。
- `evaluation`: 评估 ground truth 时加载 `dose`, `structure_masks`, `possible_dose_mask`, `voxel_dimensions`。
- `predicted_dose`: 评估 prediction CSV 时加载模型输出的 dose。

需要注意：

- CT 和 dose 在 CSV 里是 sparse format，只保存非零 voxel 的 index 和 value。
- ROI mask CSV 只保存 mask 内 voxel 的 index。
- `shape_data()` 会把 sparse CSV 恢复成 dense 3D tensor。
- 结构 mask 的 10 个 channel 顺序是：

```text
Brainstem, SpinalCord, RightParotid, LeftParotid, Esophagus,
Larynx, Mandible, PTV56, PTV63, PTV70
```

## 5. Baseline 模型

模型定义在 `provided_code/network_architectures.py`。

这是一个小型 3D U-Net 风格模型：

```text
Inputs:
  CT:        (128, 128, 128, 1)
  ROI masks: (128, 128, 128, 10)

Concatenate:
  (128, 128, 128, 11)

Encoder:
  Conv3D blocks with stride (2, 2, 2)
  spatial size gradually downsampled

Bottleneck:
  lowest-resolution latent representation

Decoder:
  Conv3DTranspose blocks
  skip connections concatenate encoder features back in

Output:
  predicted dose: (128, 128, 128, 1)
```

为什么叫 U-Net：

- 左边 encoder 压缩空间分辨率，学习全局上下文。
- 右边 decoder 恢复空间分辨率，输出 voxel-wise prediction。
- skip connections 把浅层空间信息传给 decoder，帮助恢复边界和局部结构。

当前 baseline 特别小：

```python
initial_number_of_filters = 1
```

README 也说明这不是 state-of-the-art，只是 placeholder。真实研究中常见起点会是 16、32 或 64 个初始 filters，但显存占用会明显增加。

## 6. Loss、optimizer 和训练

模型 compile 在 `define_generator()` 末尾：

```python
generator.compile(loss="mean_absolute_error", optimizer=self.gen_optimizer)
```

也就是说训练 loss 是 MAE：

```text
mean(|true_dose - predicted_dose|)
```

optimizer 在 `provided_code/network_functions.py`：

```python
Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
```

训练流程在 `PredictionModel.train_model()`：

1. 查找 `results/<model_name>/models/epoch_*.h5`，如果已有 checkpoint，会从最高 epoch 继续。
2. 设置 DataLoader 为 `training_model`。
3. 每个 epoch shuffle training patient list。
4. 每个 batch 执行：

```python
train_on_batch([batch.ct, batch.structure_masks], [batch.dose])
```

5. 按 `save_frequency` 保存模型。
6. 用 `keep_model_history` 控制只保留最近若干 checkpoint。

当前 `main.py` 设定：

```python
prediction_name = "baseline_full_150"
num_epochs = 150
test_time = False
```

所以它会训练/恢复 `baseline_full_150`，然后在 validation set 上预测和评估。

## 7. Prediction 和 possible dose mask

预测流程在 `PredictionModel.predict_dose()`：

```python
dose_pred = self.generator.predict([batch.ct, batch.structure_masks])
dose_pred = dose_pred * batch.possible_dose_mask
```

这一步很重要：模型输出会被 `possible_dose_mask` 乘一下，mask 外的 dose 被强制设为 0。

然后代码把 dense 3D dose 转回 sparse CSV：

```python
dose_to_save = sparse_vector_function(dose_pred)
```

预测结果保存到：

```text
results/<model_name>/validation-predictions/pt_*.csv
```

## 8. Evaluation 指标

OpenKBP baseline 主要有两个指标：dose score 和 DVH score。两个都是越低越好。

### Dose score

代码在 `DoseEvaluator.evaluate()`：

```python
patient_dose_error =
    sum(abs(reference_dose - predicted_dose)) / sum(possible_dose_mask)
```

含义：

- 在 possible dose region 内计算 voxel-wise absolute error。
- 对所有患者取平均。
- 更像深度学习里的 voxel-level MAE。

### DVH score

DVH 是 dose-volume histogram。它不只是看每个 voxel 是否准确，而是看某个器官/靶区整体收到的剂量统计是否准确。

代码计算的 DVH metrics：

OARs:

- `D_0.1_cc`: 该器官最热点附近 0.1 cc 的剂量，关注高剂量伤害风险。
- `mean`: 该器官平均剂量。

Targets:

- `D_99`: 99% 靶区体积至少收到的 dose，关注覆盖不足。
- `D_95`: 95% 靶区体积至少收到的 dose。
- `D_1`: 最高剂量附近，关注热点。

DVH score 是 reference DVH metrics 和 predicted DVH metrics 的平均绝对差。

关键理解：

- Dose score 好，不一定 DVH score 最好。
- DVH 更接近临床评价，因为 radiotherapy plan 关心器官和靶区的 dose-volume 约束。

## 9. 当前 baseline 复现实验表

下面分数是用现有 `results/*/validation-predictions` 重新计算得到的。`baseline_full_*` 的含义是：训练阶段使用了全部 training set，评估阶段使用了全部 40-patient validation set。`baseline` 早期目录只有 5 个 prediction CSV，因此不能和完整 validation 结果直接比较。

| Model | Training data used | Validation patients evaluated | Dose score | DVH score | 备注 |
|---|---|---:|---:|---:|---|
| `baseline` | early/dry-run setup | 5 | 23.516 | 52.456 | 早期小规模结果，非完整 validation |
| `baseline_full_20` | full training set | full validation set, 40 | 15.526 | 28.913 | 完整 validation |
| `baseline_full_50` | full training set | full validation set, 40 | 8.451 | 10.152 | 当前 DVH 最好 |
| `baseline_full_100` | full training set | full validation set, 40 | 7.962 | 10.666 | 当前 dose score 最好 |
| `baseline_full_150` | full training set | full validation set, 40 | 8.555 | 13.247 | 比 100 epoch 退化 |

初步解释：

- 20 到 50 epoch 提升很大，说明模型还在学习基本 dose pattern。
- 100 epoch 的 voxel-level dose score 最好，但 DVH 比 50 epoch 稍差。
- 150 epoch 两个指标都变差，可能已经过拟合，或者训练波动导致临床结构上的剂量统计变差。
- 下一步做实验记录时，不应只看 training loss；必须同时保存 validation dose score 和 DVH score。

## 10. 从 dose prediction 扩展到 uncertainty prediction

项目目标是 dose + uncertainty prediction。这里的 uncertainty 可以先分成两种：

- Aleatoric uncertainty: 数据本身或任务本身的不确定性。例如同样 CT/mask 可能存在多种 clinically acceptable plans。
- Epistemic uncertainty: 模型不知道导致的不确定性。例如训练数据不足、模型参数不确定。

推荐从低风险路线开始，不要一开始大改 baseline。

### Step 1: 先做 deterministic baseline 分析

先把当前 dose-only baseline 讲清楚，并补齐：

- 患者 slice 可视化：CT、真实 dose、预测 dose、absolute error、ROI overlay。
- validation table：不同 epoch 的 dose score / DVH score。
- per-patient score table：找出最好/最差患者。
- per-ROI DVH error table：看哪些结构最难预测。

这一阶段不改变模型结构，风险最低。

### Step 2: MC dropout uncertainty

这是最适合作为第一个 uncertainty baseline 的方法，因为当前网络已经用了 `SpatialDropout3D(0.2)`。

做法：

1. 预测时保持 dropout active。
2. 对同一个 patient 做 T 次 stochastic forward passes，例如 T=20。
3. 得到 T 个 dose predictions。
4. mean prediction 作为最终 dose。
5. voxel-wise standard deviation 作为 uncertainty map。

输出：

```text
mean_dose(x, y, z)
std_dose(x, y, z)
```

优点：

- 不需要训练多个模型。
- 和现有模型最接近。
- 容易可视化 uncertainty map。

风险：

- Keras 里加载 `.h5` 后让 dropout 在 inference 保持 active 需要小心实现。
- 当前 dropout 只在 decoder 部分，uncertainty 质量未必强，但足够做第一个 thesis baseline。

### Step 3: Deep ensemble

训练多个相同结构但不同 random seed / shuffle / initialization 的模型，例如 5 个：

```text
baseline_seed_1
baseline_seed_2
...
baseline_seed_5
```

预测时：

- mean across models = final dose。
- std across models = epistemic uncertainty。

优点：

- 通常比 MC dropout 更稳定。
- 容易解释。

缺点：

- 训练成本是 N 倍。
- 需要更好的实验管理。

### Step 4: Heteroscedastic regression

把模型输出从 1 channel 改成 2 channels：

```text
output[..., 0] = predicted mean dose
output[..., 1] = predicted log variance
```

loss 用 Gaussian negative log likelihood：

```text
0.5 * exp(-log_var) * (y - mean)^2 + 0.5 * log_var
```

优点：

- 模型直接学习数据噪声/任务不确定性。
- 更像正式 uncertainty prediction method。

缺点：

- 要改模型 output、loss、prediction saving、evaluation 和 visualization。
- 如果没有 calibration 分析，uncertainty 可能只是漂亮但不可信的 heatmap。

建议顺序：

```text
deterministic baseline analysis
-> MC dropout
-> deep ensemble
-> heteroscedastic output, if time allows
```

## 11. 下一次 meeting 可汇报内容

建议汇报结构：

1. Project objective
   - Input: CT + segmentation masks.
   - Output now: dose distribution.
   - Output next: dose + uncertainty map.

2. Dataset
   - OpenKBP, head-and-neck radiotherapy.
   - 200 train, 40 validation, 100 test.
   - Each patient is `128 x 128 x 128`.
   - Structures include OARs and PTV targets.

3. Baseline pipeline
   - DataLoader converts sparse CSV to dense tensors.
   - 3D U-Net takes CT and ROI masks.
   - Loss is dose MAE.
   - Prediction is masked by possible dose mask.
   - Evaluation uses dose score and DVH score.

4. Reproduction results
   - 20 epoch: dose 15.526, DVH 28.913.
   - 50 epoch: dose 8.451, DVH 10.152.
   - 100 epoch: dose 7.962, DVH 10.666.
   - 150 epoch: dose 8.555, DVH 13.247.
   - Key observation: 100 epoch best dose score, 50 epoch best DVH score.

5. Next plan
   - Add prediction/error visualization.
   - Add per-patient and per-ROI analysis.
   - Implement MC dropout uncertainty first.
   - Compare uncertainty map with absolute dose error as a sanity check.

## 12. Immediate next tasks

Recommended next coding tasks, in order:

1. Add a script `scripts/evaluate_existing_predictions.py`
   - Evaluate any `results/<model>/validation-predictions` folder.
   - Save a CSV summary table.

2. Add a script `scripts/visualize_prediction_error.py`
   - For one patient and one model, plot CT, true dose, predicted dose, absolute error, and ROI overlay.

3. Add experiment log file
   - For example `results/experiment_summary.csv`.
   - Columns: model name, epoch, train patients, validation patients, dose score, DVH score, notes.

4. Implement MC dropout prototype
   - Start with one patient.
   - Save mean dose and std uncertainty visualization.
   - Do not change training code until the inference prototype is understood.

5. Prepare meeting slides
   - Use the baseline table and visualizations.
   - Emphasize the difference between dose score and DVH score.
