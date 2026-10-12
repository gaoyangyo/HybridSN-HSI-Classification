# HybridSN 高光谱图像分类复现（Pavia University）

> 课程/组内考核任务 · 复现论文 **HybridSN: Exploring 3-D–2-D CNN Feature Hierarchy for
> Hyperspectral Image Classification**

---

## ⚠️ 数据获取（重要，官方源已失效）

论文脚注给出的官方数据源 `www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes`
目前**返回 HTTP 403，无法下载**。

本项目使用的数据来自官方开源仓库打包的副本（已核对尺寸、类别数、逐类样本数与论文一致）：

- 来源：`https://github.com/gokriznastic/HybridSN` → `data/PaviaU.mat`、`data/PaviaU_gt.mat`
- 若 GitHub 不可直连，可用代理：`https://gh-proxy.com/https://github.com/.../archive/refs/heads/master.zip`

把两个文件放进 `data/` 即可：

```
data/
├── PaviaU.mat        (610 × 340 × 103, uint16)
└── PaviaU_gt.mat     (610 × 340, uint8, 0=背景, 1..9=类别)
```

**数据实测信息**（用 MATLAB v5 头解析验证）：

| 项目 | 值 |
|---|---|
| 空间尺寸 | 610 × 340 |
| 波段数 | 103（430–860 nm） |
| 数据类型 | **uint16**（原始 DN 值，必须转 float 并归一化） |
| 类别数 | 9 |
| 标注像元 | 42,776（背景 164,624） |

逐类样本数（严重不平衡，是分析的重点）：

| # | 类别 | 样本数 | # | 类别 | 样本数 |
|---|---|---|---|---|---|
| 1 | Asphalt | 6,631 | 6 | Bare Soil | 5,029 |
| 2 | Meadows | 18,649 | 7 | Bitumen | 1,330 |
| 3 | Gravel | 2,099 | 8 | Self-Blocking Bricks | 3,682 |
| 4 | Trees | 3,064 | 9 | Shadows | 947 |
| 5 | Painted metal sheets | 1,345 | | **合计** | **42,776** |

---

## 环境配置

```bash
# 1) 建虚拟环境（Python 3.10–3.14 均可；本机实测 3.14.3 + torch 2.14 可用）
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 2) 安装依赖
pip install -r requirements.txt

# 3) 验证环境（会打印 GPU / CUDA / 各库版本）
python -c "import sys; sys.path.insert(0,'src'); from utils import env_report; print(env_report())"
```

> **国内网络提示**：若 pip 很慢，加镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。
> 若用了代理工具（如 Clash 监听 127.0.0.1:7897），命令行需要显式设置：
> `set HTTPS_PROXY=http://127.0.0.1:7897`（PowerShell 用 `$env:HTTPS_PROXY=...`）。

---

## 目录结构

```
HybridSN-HSI-Classification/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── pavia.yaml              # 全部超参数集中管理
├── src/
│   ├── model.py                # HybridSN 网络定义（含结构消融变体）
│   ├── data.py                 # 读取 / PCA / 分层划分 / patch 构建
│   ├── train.py                # 训练 + 测试 + 出图
│   ├── predict.py              # 整幅影像预测 + 四联图
│   ├── metrics.py              # OA / AA / Kappa / 逐类精度 / 混淆矩阵
│   └── utils.py                # 随机种子 / 设备 / 环境信息 / 绘图风格
├── data/                       # 数据集（不入库）
└── results/<tag>/              # 每次实验的全部产出
```

---

## 运行方法

### 1. 自检模型结构（复现论文表 I 的参数量）

```bash
python src/model.py
```

预期输出关键行：

```
IP 配置 (C=16, B=30, S=25) 可训练参数 : 5,122,176   论文报告 : 5,122,176
✅ 与论文表 I 完全一致
```

**这是"论文理解与模型结构"最硬的自证**：参数量逐层对上论文表 I，说明网络结构没有理解错。

### 2. 论文配置主实验（30% 训练，PCA 15，patch 25）

```bash
python src/train.py --config configs/pavia.yaml
```

### 3. 整幅影像预测与结果图（任务 4）

```bash
python src/predict.py --exp results/paper_30pct --scope all
```

### 4. 消融实验（任务 5，至少 3 组）

```bash
# 因素 A：训练样本比例（推荐主做，可直接对标论文补充材料表 7/8/9）
python src/train.py --train-ratio 0.05 --tag ratio05
python src/train.py --train-ratio 0.10 --tag ratio10
python src/train.py --train-ratio 0.30 --tag ratio30

# 因素 B：结构消融（验证论文核心主张）
python src/train.py --variant no2d  --tag arch_no2d    # 去掉 2-D CNN
python src/train.py --variant two3d --tag arch_two3d   # 3-D 卷积 3 层变 2 层

# 因素 C：PCA 主成分数
python src/train.py --pca 10 --tag pca10
python src/train.py --pca 30 --tag pca30

# 汇总对比表（见下节脚本）
```

**铁律**：除研究的那个因素外，其余参数一律不动，`seed` 固定为 42，
划分索引存在 `results/<tag>/split.npz`，保证各组实验可比。

---

## 主要参数设置（与论文对照）

| 参数 | 本项目 | 论文 | 说明 |
|---|---|---|---|
| PCA 主成分数 | 15 | 15 | 论文对 UP 用 15（IP 用 30） |
| Patch 空间窗口 | 25 × 25 | 25 × 25 | 论文表 IV 对比过 19/21/23/25 |
| 训练比例 | 0.3 | 0.3 | ★ **按类别分层抽样** |
| Batch size | 256 | 256 | |
| Epoch | 100 | 100 | 论文约 50 epoch 收敛 |
| 优化器 | Adam | Adam（带 momentum） | |
| 学习率 | 0.001 | 0.001 | |
| 损失 | CrossEntropy | Softmax 交叉熵 | |
| Batch Norm | 不用 | 不用 | |
| 数据增强 | 不用 | 不用 | |
| Dropout | 0.4 | 未给出 | 沿用官方 Keras 代码 |

### 🔑 关键复现细节：划分方式是"按类别分层"

论文 30% 训练时，UP 的训练/测试样本数是 **12832 / 29944**。从补充材料的逐类
Train/Test 列可以反推：**每个类别各取 30%**，而不是对全部标注像元随机取 30%。

本项目在 `data.py::stratified_split` 中显式实现了按类分层。
若改用全图随机 30%，各类数量会有出入，**结果与论文不可比**。

---

## 主要结果

> 运行 Fill-in：下表在完成实验后填入。论文参考值来自补充材料。

### 与论文对比（UP，30% 训练）

| 方法 | OA (%) | Kappa | AA (%) |
|---|---|---|---|
| SVM（论文） | 94.34 | 92.50 | 92.98 |
| 2-D-CNN（论文） | 97.86 | 97.16 | 96.55 |
| 3-D-CNN（论文） | 96.53 | 95.51 | 97.57 |
| M3D-CNN（论文） | 95.76 | 94.50 | 95.08 |
| SSRN（论文） | 99.90 | 99.87 | 99.91 |
| **HybridSN（论文）** | **99.98** | **99.98** | **99.97** |
| **HybridSN（本复现）** | *待填* | *待填* | *待填* |

### 训练比例消融（对照论文补充材料表 7/8/9）

| 训练比例 | 本复现 OA | 本复现 AA | 本复现 Kappa | 论文 OA | 论文 AA |
|---|---|---|---|---|---|
| 5 % | *待填* | *待填* | *待填* | 99.35 ± 1.1 | 98.65 ± 0.8 |
| 10 % | *待填* | *待填* | *待填* | 99.72 ± 0.1 | 99.20 ± 0.2 |
| 20 % | *待填* | *待填* | *待填* | 99.86 ± 0.1 | 99.71 ± 0.1 |
| 30 % | *待填* | *待填* | *待填* | 99.98 ± 0.01 | 99.97 ± 0.01 |

### 结构消融

| 变体 | 参数量 | OA (%) | AA (%) | Kappa |
|---|---|---|---|---|
| full（论文结构） | *待填* | *待填* | *待填* | *待填* |
| two3d（少一个 3-D 卷积） | *待填* | *待填* | *待填* | *待填* |
| no2d（去掉 2-D CNN） | *待填* | *待填* | *待填* | *待填* |

> 提示：`python src/model.py` 会打印各变体的参数量。去掉 2-D CNN 后参数量会**暴涨**，
> 这正是论文"使用混合 CNN 降低了模型复杂度"的直接证据。

---

## 结果分析要点（写报告用）

1. **OA / AA / Kappa 必须一起看**。UP 中 Meadows 一类就占 18,649/42,776 ≈ 44%，
   OA 会被大类主导。论文补充材料里 IP 数据集上 SSRN 的 OA（98.45）甚至高于
   HybridSN（98.39），但 AA 只有 86.19（HybridSN 98.01）—— 这是"只看 OA 会误导"的典型例子。
2. **类别不平衡**：Meadows 18,649 vs Shadows 947，差近 20 倍。若发现 AA 明显低于 OA，
   说明小类被牺牲，可尝试类别加权或平衡采样。
3. **错误的空间分布**：看 `maps/panel.png` 的第 (d) 幅。patch 方法用邻域信息，
   在**地物边界和细碎区域**最容易出错，因为一个 patch 里混了多种地物。
4. **本方法的已知局限**（可作为批判性分析）：
   - PCA 在全图（含测试像元）上拟合 → 轻微信息泄漏（可用 `--pca-fit train` 做对照）
   - 逐像元随机划分 → 相邻像元 patch 重叠达 96%，空间自相关使精度偏乐观
   - patch 边界用零填充 → 影像边缘引入假结构
   - 参数量 92% 集中在第一层全连接，特征提取部分其实很轻

---

## 来源与致谢（学术诚信声明）

- **原论文**：S. K. Roy, G. Krishna, S. R. Dubey, and B. B. Chaudhuri,
  "HybridSN: Exploring 3-D–2-D CNN Feature Hierarchy for Hyperspectral Image Classification,"
  *IEEE Geoscience and Remote Sensing Letters*, vol. 17, no. 2, pp. 277–281, Feb. 2020.
  DOI: [10.1109/LGRS.2019.2918719](https://doi.org/10.1109/LGRS.2019.2918719)
- **官方 Keras 实现**：<https://github.com/gokriznastic/HybridSN>（作者本人开源）
- **PyTorch 参考实现**：<https://github.com/Pancakerr/HybridSN>
- **数据集**：Pavia University（ROSIS 传感器，2001，意大利帕维亚）
- **补充材料**：论文 Supplementary Material，含逐类精度与 5%/10%/20% 训练比例结果

> 本项目在参考实现的基础上，自行完成了数据读取、按类别分层划分、PCA/归一化流程、
> 训练与评价管线、结构消融接口和结果可视化，并对论文的划分方式与潜在信息泄漏问题
> 做了核对与分析。**公开代码的来源均已在此声明，不冒充原创。**
