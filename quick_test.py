# -*- coding: utf-8 -*-
"""HybridSN 复现 · 单文件速测脚本

不依赖 src/ 下任何模块，一个文件跑通完整流程：
    数据读取 → 归一化 → PCA → 按类别分层划分 → 25×25 patch
    → HybridSN 网络 → 训练 → 评价（OA / AA / Kappa + 逐类精度 + 混淆矩阵）

用法（在项目目录下）：
    D:\\Users\\conda\\envs\\hybridsn\\python.exe quick_test.py

预计耗时：GPU 约 1 分钟。
"""
import os
import sys
import time

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

# ============================ 配置 ============================
DATA_DIR    = 'data'
PCA_B       = 15        # 论文对 Pavia University 用 15 个主成分
PATCH       = 25        # 论文表 IV：19/21/23/25 里 25×25 最优
TRAIN_RATIO = 0.30      # 论文：30% 训练 / 70% 测试
BATCH       = 256       # 论文：256
EPOCHS      = 5         # 这里是速测；论文正式实验是 100 轮
LR          = 1e-3      # 论文：0.001（Adam）
SEED        = 42

CLASSES = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets',
           'Bare Soil', 'Bitumen', 'Self-Blocking Bricks', 'Shadows']

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)


def hr(title):
    print('\n' + '=' * 70)
    print(title)
    print('=' * 70)


# ====================== 1. 环境 ======================
hr('1. 环境检查')
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Python    :', sys.version.split()[0])
print('PyTorch   :', torch.__version__)
print('设备      :', dev, '|', torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU')
print('numpy     :', np.__version__)

# ====================== 2. 数据 ======================
hr('2. 读取 Pavia University')
cube = sio.loadmat(os.path.join(DATA_DIR, 'PaviaU.mat'))['paviaU']
gt = sio.loadmat(os.path.join(DATA_DIR, 'PaviaU_gt.mat'))['paviaU_gt'].astype(np.int64)
print('数据立方体 :', cube.shape, ' dtype=', cube.dtype, ' 取值范围', cube.min(), '~', cube.max())
print('真值标签   :', gt.shape, '  类别数', gt.max(), '  标注像元', int((gt > 0).sum()))
for c in range(1, gt.max() + 1):
    print('   {}. {:<22} {:>6}'.format(c, CLASSES[c - 1], int((gt == c).sum())))

# ====================== 3. 预处理 ======================
hr('3. 归一化 + PCA（只降光谱维，空间尺寸不变）')
t0 = time.time()
data = cube.astype(np.float32) / np.float32(cube.max())      # uint16 -> [0,1]
M, N, D = data.shape
pca = PCA(n_components=PCA_B, whiten=False)
data = pca.fit_transform(data.reshape(-1, D)).reshape(M, N, PCA_B).astype(np.float32)
print('原始波段数 : {} -> PCA 后 {}'.format(D, PCA_B))
print('累计解释方差比 : {:.4f}'.format(pca.explained_variance_ratio_.sum()))
print('空间尺寸   : {} × {}  (保持不变)'.format(M, N))
print('预处理耗时 : {:.1f} 秒'.format(time.time() - t0))

# ====================== 4. 按类别分层划分 ======================
hr('4. 按类别分层划分（每个类别各取 30%）')
rng = np.random.default_rng(SEED)
tr, te = [], []
for c in range(1, gt.max() + 1):
    rows, cols = np.where(gt == c)
    perm = rng.permutation(len(rows))
    k = int(np.floor(len(rows) * TRAIN_RATIO))
    tr.append(np.stack([rows[perm[:k]], cols[perm[:k]]], 1))
    te.append(np.stack([rows[perm[k:]], cols[perm[k:]]], 1))
tr = np.concatenate(tr)
te = np.concatenate(te)
print('训练像元 : {}'.format(len(tr)))
print('测试像元 : {}'.format(len(te)))
print('合计     : {}  (= 全部标注像元)'.format(len(tr) + len(te)))
print('论文参考 : 12832 / 29944（差 ±1 是我这里用 floor 取整所致，无影响）')

# ====================== 5. patch 构建 ======================
hr('5. 构建 {0}×{0} patch（零填充边缘）'.format(PATCH))
pad = PATCH // 2
padded = np.pad(data, ((pad, pad), (pad, pad), (0, 0)), mode='constant')


def get_patch(coords, idx):
    r, c = coords[idx]
    win = padded[r:r + PATCH, c:c + PATCH, :]                # (S, S, B)
    return np.ascontiguousarray(win.transpose(2, 0, 1))[None]  # (1, B, S, S)


X_tr = np.stack([get_patch(tr, i) for i in range(len(tr))])
y_tr = np.array([gt[r, c] - 1 for r, c in tr])
X_te = np.stack([get_patch(te, i) for i in range(len(te))])
y_te = np.array([gt[r, c] - 1 for r, c in te])
print('训练集张量 :', X_tr.shape, ' 标签', y_tr.shape)
print('测试集张量 :', X_te.shape, ' 标签', y_te.shape)
print('内存占用   : {:.1f} MB'.format((X_tr.nbytes + X_te.nbytes) / 1024 / 1024))


# ====================== 6. HybridSN 网络 ======================
hr('6. HybridSN 网络（论文表 I 结构）')
class HybridSN(nn.Module):
    """3 个 3-D 卷积提光谱-空间联合特征 → Reshape → 1 个 2-D 卷积做空间抽象 → 3 层全连接"""

    def __init__(self, n_cls, bands, patch, dropout=0.4):
        super().__init__()
        # 注意：PyTorch Conv3d 输入是 (N,C,D,H,W)，D 是光谱维，
        # 所以 kernel_size 顺序是 (光谱深, 空间高, 空间宽)，光谱深放第一位
        self.c3 = nn.ModuleList([
            nn.Conv3d(1, 8, (7, 3, 3)),
            nn.Conv3d(8, 16, (5, 3, 3)),
            nn.Conv3d(16, 32, (3, 3, 3)),
        ])
        sp = patch - 2 * 3          # 25 -> 19
        bd = bands - 6 - 4 - 2      # 15 -> 3
        self.sp, self.bd = sp, bd
        self.ch = 32 * bd           # 32 × 3 = 96
        self.c2 = nn.Conv2d(self.ch, 64, 3)
        flat = (sp - 2) ** 2 * 64   # 17×17×64 = 18496
        self.fc1, self.do1 = nn.Linear(flat, 256), nn.Dropout(dropout)
        self.fc2, self.do2 = nn.Linear(256, 128), nn.Dropout(dropout)
        self.fc3 = nn.Linear(128, n_cls)

    def forward(self, x):
        for c in self.c3:
            x = torch.relu(c(x))
        n = x.size(0)
        x = x.permute(0, 2, 1, 3, 4).reshape(n, self.ch, self.sp, self.sp)
        x = torch.relu(self.c2(x))
        x = x.reshape(n, -1)
        return self.fc3(self.do2(torch.relu(self.fc2(self.do1(torch.relu(self.fc1(x)))))))


model = HybridSN(9, PCA_B, PATCH).to(dev)
n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('可训练参数 : {:,}'.format(n_param))
print('论文参考   : 4,844,793（UP 配置：9 类 / PCA 15 / 窗口 25）')
print('逐层参数量 :')
for nm, mod in [('conv3d_1', model.c3[0]), ('conv3d_2', model.c3[1]), ('conv3d_3', model.c3[2]),
                ('conv2d_1', model.c2), ('fc1', model.fc1), ('fc2', model.fc2), ('fc3', model.fc3)]:
    print('   {:<10} {:>12,}'.format(nm, sum(p.numel() for p in mod.parameters())))

with torch.no_grad():
    probe = torch.randn(2, 1, PCA_B, PATCH, PATCH, device=dev)
    print('前向形状自检 :', tuple(probe.shape), '->', tuple(model(probe).shape))

# ====================== 7. 训练 ======================
hr('7. 训练（{} 轮 / batch {} / lr {} / Adam）'.format(EPOCHS, BATCH, LR))
opt = torch.optim.Adam(model.parameters(), lr=LR)
crit = nn.CrossEntropyLoss()
Xtr_t = torch.from_numpy(X_tr).float()
ytr_t = torch.from_numpy(y_tr).long()


def predict(X, batch=512):
    """分批推理。

    ★ 4 GB 显存的关键限制：绝不能把整个测试集一次搬上 GPU。
      29,947 个样本一次性前向，第一层卷积就要分配约 3.9 GiB，必然 OOM。
      所以必须按 batch 分批送进显存。
    """
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).float().to(dev)
            outs.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(outs)

hist = []
t0 = time.time()
for ep in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(Xtr_t))
    tot, cor = 0.0, 0
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i + BATCH]
        xb = Xtr_t[idx].to(dev)
        yb = ytr_t[idx].to(dev)
        opt.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()
        opt.step()
        tot += loss.item() * len(yb)
        cor += (out.argmax(1) == yb).sum().item()
    model.eval()
    acc = (predict(X_te) == y_te).mean()
    hist.append((tot / len(perm), cor / len(perm), float(acc)))
    print('  epoch {}/{}  loss {:.4f}  train_acc {:.2f}%  test_acc {:.2f}%'
          .format(ep, EPOCHS, tot / len(perm), cor / len(perm) * 100, acc * 100))
print('训练耗时 : {:.1f} 秒'.format(time.time() - t0))

# ====================== 8. 评价 ======================
hr('8. 评价指标')
y_pred = predict(X_te)
y_true = y_te

cm = np.zeros((9, 9), dtype=np.int64)
for t, p in zip(y_true, y_pred):
    cm[t, p] += 1
oa = (y_true == y_pred).mean()
accs = [cm[c, c] / cm[c].sum() if cm[c].sum() else 0 for c in range(9)]
aa = float(np.mean(accs))
po = np.trace(cm) / cm.sum()
pe = float((cm.sum(0) * cm.sum(1)).sum()) / cm.sum() ** 2
kappa = (po - pe) / (1 - pe)

print('测试样本数 :', len(y_true))
print('OA（总体精度）   : {:.2f} %'.format(oa * 100))
print('AA（平均精度）   : {:.2f} %'.format(aa * 100))
print('Kappa 系数       : {:.4f}'.format(kappa))
print('')
print('{:<4}{:<24}{:>7}{:>10}{:>10}{:>10}'.format('类别', '名称', '样本', '精度%', '精确率', 'F1'))
print('-' * 66)
for c in range(9):
    tp = cm[c, c]
    fp = cm[:, c].sum() - tp
    fn = cm[c, :].sum() - tp
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    print('{:<4}{:<24}{:>7}{:>10.2f}{:>10.4f}{:>10.4f}'.format(
        c + 1, CLASSES[c], int(cm[c].sum()), accs[c] * 100, prec, f1))

print('')
print('论文参考值（30% 训练）: OA 99.98% / AA 99.97% / Kappa 99.98%')
print('注意：本次只训练了 {} 轮（论文是 100 轮），且为单次随机划分，'.format(EPOCHS))
print('      所以数值会低于论文，属正常。跑满 100 轮后会明显接近。')

# ====================== 9. 出图 ======================
try:
    os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.getcwd(), '.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    e = range(1, EPOCHS + 1)
    ax[0].plot(e, [h[0] for h in hist], 'o-', color='tab:red')
    ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Loss'); ax[0].set_title('训练损失')
    ax[0].grid(alpha=.3)
    ax[1].plot(e, [h[1] * 100 for h in hist], 'o-', label='训练精度')
    ax[1].plot(e, [h[2] * 100 for h in hist], 's-', label='测试精度')
    ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Accuracy (%)')
    ax[1].set_title('精度收敛'); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle('HybridSN 速测 / Pavia University / {}% 训练 / {} 轮'.format(
        int(TRAIN_RATIO * 100), EPOCHS))
    fig.tight_layout()
    fig.savefig('quick_test_curves.png', dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    norm = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(9)); ax.set_yticks(range(9))
    ax.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(CLASSES, fontsize=8)
    ax.set_xlabel('预测类别'); ax.set_ylabel('真实类别')
    ax.set_title('混淆矩阵（按行归一化）\nOA={:.2f}%  AA={:.2f}%  Kappa={:.4f}'
                 .format(oa * 100, aa * 100, kappa))
    for i in range(9):
        for j in range(9):
            if cm[i, j]:
                ax.text(j, i, '{:.2f}'.format(norm[i, j]), ha='center', va='center',
                        fontsize=6.5, color='white' if norm[i, j] > .5 else 'black')
    fig.colorbar(im, ax=ax, shrink=.85)
    fig.savefig('quick_test_confusion.png', dpi=130)
    plt.close(fig)
    print('\n已出图: quick_test_curves.png / quick_test_confusion.png')
except Exception as exc:
    print('\n[警告] 出图失败（不影响指标）:', exc)

print('\n速测完成。')
