我是伯明翰大学 MSc AI&ML 学生，项目方向是医学图像分析。导师 Qingjie Meng 给我的 MSc thesis 项目是 radiotherapy dose and uncertainty prediction。临床合作方是 Antony。项目目标是：使用 CT images 和 segmentation masks 作为输入，同时预测 radiotherapy dose distributions 和 uncertainty maps。

当前使用公开数据集和代码仓库：
https://github.com/ababier/open-kbp#data

我的 fork：
https://github.com/xrl2408/open-kbp.git

WSL 项目路径：
/home/yufu/projects/open-kbp

当前环境：
- WSL2 Ubuntu
- conda 环境名：openkbp
- Python 3.10
- TensorFlow 2.11
- Keras 2.11
- cudatoolkit 11.2
- cudnn 8.1
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12GB VRAM
- nvidia-smi 可用
- TensorFlow 能识别 GPU
- 会出现 TensorRT、NUMA、compute capability 12.0 / PTX JIT warning，这些目前不是致命错误
- 第一次 batch 很慢，主要是 PTX JIT 编译；后面 batch 明显变快

数据状态：
- Open-KBP 数据已经下载完成
- train: 200 patients
- validation: 40 patients
- test: 100 patients

已经做过的检查：
- 写过 scripts/inspect_patient.py 检查 pt_1
- CT shape: (128, 128, 128)
- Dose shape: (128, 128, 128)
- Possible dose mask shape: (128, 128, 128)
- Structure masks shape: (128, 128, 128, 10)
- Voxel dimensions: [3.906, 3.906, 2.5]
- 输出图保存到 outputs/inspect_patient/pt_1_overview.png

已经跑过 baseline：
1. 小规模 dry run 成功：
   - DVH score: 56.999
   - dose score: 21.573

2. 更完整 baseline 验证成功：
   - validation 40 patients
   - DVH score: 28.913
   - dose score: 15.526
3. 完整baseline 验证成功
   - epoch 20 50 100 150都已经跑过，详细可查看output目录里
我现在的目标：
1. 先理解 open-kbp 仓库结构和 baseline pipeline。这个你帮我写一份文档给我详细解释，我对医学领域完全不了解，在AI领域只能算初学者，对tensorflow，unet的代码也不熟悉
2. 整理 baseline 的输入、模型、loss、训练、validation、dose score 和 DVH score。
3. 下一步规划如何从 dose prediction 扩展到 dose + uncertainty prediction。
4. 需要优先做可视化、实验记录、baseline 复现表格，以及下一次 meeting 可以汇报的内容。
5. 请不要急着大改代码，先阅读项目结构并给我一个清晰的下一步计划。

2026-06-22 继续进展：
- 已新增文档：docs/openkbp_baseline_and_uncertainty_plan.md
- 文档内容包括：仓库结构、数据格式、DataLoader、baseline 3D U-Net、loss、训练流程、prediction、dose score、DVH score、已有 baseline 复现实验表、uncertainty prediction 路线、下一次 meeting 可汇报内容。
- 已用现有 validation predictions 重新计算 baseline 表；其中 baseline_full_* 的含义是训练阶段使用全部 training set，评估阶段使用全部 40-patient validation set：
  - baseline: early/dry-run setup, 5 validation patients, dose 23.516, DVH 52.456
  - baseline_full_20: full training set + full 40-patient validation set, dose 15.526, DVH 28.913
  - baseline_full_50: full training set + full 40-patient validation set, dose 8.451, DVH 10.152
  - baseline_full_100: full training set + full 40-patient validation set, dose 7.962, DVH 10.666
  - baseline_full_150: full training set + full 40-patient validation set, dose 8.555, DVH 13.247
- 关键观察：100 epoch 的 dose score 最好，但 50 epoch 的 DVH score 最好；150 epoch 退化。后续实验不能只看 voxel-level dose MAE，也要看 DVH。
- 建议下一步先写两个小脚本：evaluate_existing_predictions.py 和 visualize_prediction_error.py，然后再做 MC dropout uncertainty prototype。

2026-06-22 脚本和可视化进展：
- 已新增 scripts/evaluate_existing_predictions.py
  - 作用：读取 results/<model>/validation-predictions，重新计算 dose score 和 DVH score。
  - 当前输出：docs/experiment_summary.csv
- 已新增 scripts/visualize_prediction_error.py
  - 作用：对单个 validation patient 画 CT、ROI overlay、true dose、predicted dose、absolute error。
  - 当前输出：
    - outputs/prediction_error/baseline_full_50_pt_224_error.png
    - outputs/prediction_error/baseline_full_100_pt_224_error.png
- 已新增 .gitignore，避免误提交 results、模型权重、cache；但允许提交 outputs/prediction_error/*.png 这类选定可视化结果。
