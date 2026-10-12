# data 目录说明

原始数据集**不入库**（体积大）。请把以下两个文件放到本目录：

```
data/
├── PaviaU.mat
└── PaviaU_gt.mat
```

## 获取方式

论文脚注的官方源 `www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes`
目前已失效（实测返回 **HTTP 403**）。请改用下面任一方式：

### 方式 1：从官方开源仓库获取（推荐，已核对与论文一致）

```bash
# GitHub 可直连时
curl -L -o hybridsn.zip https://github.com/gokriznastic/HybridSN/archive/refs/heads/master.zip

# GitHub 不可直连时走代理
curl -L -o hybridsn.zip https://gh-proxy.com/https://github.com/gokriznastic/HybridSN/archive/refs/heads/master.zip
```

解压后把 `HybridSN-master/data/PaviaU.mat` 和 `PaviaU_gt.mat` 复制到本目录。
同目录下还有 `Indian_pines_corrected.mat`、`Salinas_corrected.mat` 等，可作扩展实验。

### 方式 2：Kaggle

搜索 "Pavia University" 数据集下载，注意核对尺寸应为 `610 × 340 × 103`。

## 文件格式

| 文件 | 变量名 | 尺寸 | dtype | 说明 |
|---|---|---|---|---|
| `PaviaU.mat` | `paviaU` | (610, 340, 103) | **uint16** | 原始 DN 值，需转 float 并归一化 |
| `PaviaU_gt.mat` | `paviaU_gt` | (610, 340) | uint8 | 0 = 背景（164,624 个），1–9 = 类别（42,776 个） |

## 自检

不需要任何第三方库也能核对文件是否完好：

```bash
python ../hybridsn/mat_probe.py PaviaU.mat PaviaU_gt.mat
```

应输出 `dims: [610, 340, 103]`、9 个类别、标注像元 42,776。
