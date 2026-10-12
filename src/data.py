"""Pavia University 数据读取、PCA 降维、按类别分层划分、patch 构建。

设计要点（这几条直接决定你的结果能否与论文对比）：

1. **按类别分层抽样**，不是全图随机抽样。
   论文 30% 训练时 UP 的训练/测试是 12832/29944，从补充材料的逐类 Train/Test 列
   反推可知：每个类别各取 30%。若用全图随机 30%，各类数量会有出入，结果不可比。

2. **PCA 只在光谱维做**，空间尺寸 M、N 保持不变（论文明确强调）。

3. **归一化**：paviaU 是 uint16 原始 DN 值，必须转 float 并缩放，
   否则数值在几千量级，训练直接发散。

4. **patch 按需切片**，不预先构建全部 patch。
   全图 207,400 个 25×25×15 的 patch 约 7.8 GB，一次性构建必然爆内存。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.io as sio


@dataclass
class PaviaU:
    """一个数据集切片的容器。"""
    cube: np.ndarray      # (M, N, B) float32，PCA 之后
    gt: np.ndarray        # (M, N) int64，0 表示背景
    num_classes: int
    class_names: list


# Pavia University 的 9 个地物类别（顺序即标签 1..9）
PAVIA_CLASSES = [
    'Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets',
    'Bare Soil', 'Bitumen', 'Self-Blocking Bricks', 'Shadows',
]


def load_pavia(mat_path: str, gt_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """读取 .mat 并转成 float32 / int64。

    paviaU    : (610, 340, 103) uint16
    paviaU_gt : (610, 340)      uint8
    """
    cube = sio.loadmat(mat_path)['paviaU']            # 保持原始 uint16，便于统计时如实报告
    gt = sio.loadmat(gt_path)['paviaU_gt'].astype(np.int64)
    assert cube.shape[:2] == gt.shape, f'尺寸不匹配: {cube.shape} vs {gt.shape}'
    return cube, gt


def normalize(cube: np.ndarray, mode: str = 'global', train_mask: np.ndarray | None = None
              ) -> Tuple[np.ndarray, dict]:
    """把 uint16 DN 值缩放到 [0, 1]。

    mode='global' : 用整幅影像的最大值（多数公开实现的做法，与论文一致）
    mode='train'  : 只用训练像元统计（更严谨，可作对照实验）
                    —— 这属于"信息泄漏"分析的一个小实验，写进报告是加分项
    """
    if mode == 'global':
        denom = float(cube.max())
    elif mode == 'train':
        if train_mask is None:
            raise ValueError("mode='train' 需要提供 train_mask")
        denom = float(cube[train_mask].max())
    else:
        raise ValueError(f'未知 mode: {mode}')
    stats = {'denom': denom, 'mode': mode}
    # 必须先转 float32：uint16 直接除会得到 float64，白白占用一倍内存
    return cube.astype(np.float32) / np.float32(denom), stats


def apply_pca(cube: np.ndarray, n_components: int, fit_mask: np.ndarray | None = None
              ) -> Tuple[np.ndarray, dict]:
    """沿光谱维做 PCA：D -> n_components，保持 (M, N) 不变。

    fit_mask 为 None 时在**全图**上拟合（论文做法，存在轻微信息泄漏）；
    传入训练像元掩膜则只在训练集上拟合（更严谨）。
    """
    from sklearn.decomposition import PCA

    m, n, d = cube.shape
    flat = cube.reshape(-1, d)                       # (M*N, D)

    if fit_mask is None:
        fit_rows = flat
        info = 'PCA 在全图像元上拟合（与论文一致，但含轻微信息泄漏）'
    else:
        fit_rows = flat[fit_mask.reshape(-1)]
        info = 'PCA 只在训练像元上拟合（更严谨）'

    pca = PCA(n_components=n_components, whiten=False)
    pca.fit(fit_rows)
    out = pca.transform(flat).reshape(m, n, n_components).astype(np.float32)
    return out, {
        'info': info,
        'explained_variance_ratio_sum': float(pca.explained_variance_ratio_.sum()),
        'n_components': n_components,
    }


def stratified_split(gt: np.ndarray, train_ratio: float = 0.3, seed: int = 42
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """按类别分层抽样，返回 (训练像元坐标, 测试像元坐标)，各为 (K, 2) 的 (row, col)。

    每个类别取 floor(n_c * train_ratio) 个作训练。
    ★ 说明：论文报告的逐类样本数在 3 个类上比 floor 多 1（合计 12832 vs 12829），
      属四舍五入差异，无关紧要；关键是"按类分层"而不是全图随机。
    """
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for c in range(1, gt.max() + 1):
        rows, cols = np.where(gt == c)
        n = len(rows)
        perm = rng.permutation(n)
        n_train = int(np.floor(n * train_ratio))
        train_idx.append(np.stack([rows[perm[:n_train]], cols[perm[:n_train]]], axis=1))
        test_idx.append(np.stack([rows[perm[n_train:]], cols[perm[n_train:]]], axis=1))
    return np.concatenate(train_idx, 0), np.concatenate(test_idx, 0)


def pad_cube(cube: np.ndarray, patch_size: int) -> np.ndarray:
    """按论文/官方实现的做法，用 0 在空间维两侧填充，使边缘像元也能取到完整 patch。

    [警告] 零填充会在影像边缘引入不存在的'黑边'结构，这是 HybridSN 的一个已知局限，
       可以在报告的"错误分析"里讨论（错误是否集中在地物边界）。
    """
    r = patch_size // 2
    return np.pad(cube, ((r, r), (r, r), (0, 0)), mode='constant', constant_values=0)


class PatchDataset:
    """按需切 patch，避免把全部 patch 一次性搬进内存。

    坐标一律用**原始影像**的 (row, col)，内部自动偏移 padding。
    """

    def __init__(self, padded_cube: np.ndarray, coords: np.ndarray, gt: np.ndarray,
                 patch_size: int):
        self.cube = padded_cube
        self.coords = coords
        self.gt = gt
        self.patch_size = patch_size
        self.pad = patch_size // 2

    def __len__(self) -> int:
        return len(self.coords)

    def patch(self, i: int) -> np.ndarray:
        r, c = self.coords[i]
        r += self.pad
        c += self.pad
        h = self.patch_size // 2
        # (S, S, B) -> (1, B, S, S)，与 Conv3d 的 (N, C, D, H, W) 对齐
        win = self.cube[r - h:r + h + 1, c - h:c + h + 1, :]
        return np.ascontiguousarray(win.transpose(2, 0, 1)[None, ...])

    def label(self, i: int) -> int:
        r, c = self.coords[i]
        return int(self.gt[r, c]) - 1          # 1..9 -> 0..8

    def __getitem__(self, i: int):
        return self.patch(i), self.label(i)


def build_all_coords(gt: np.ndarray) -> np.ndarray:
    """全图所有标注像元的坐标（用于整幅分类图）。"""
    rows, cols = np.where(gt > 0)
    return np.stack([rows, cols], axis=1)


def dataset_summary(cube_raw: np.ndarray, gt: np.ndarray) -> dict:
    """任务 3 要求展示的数据统计。"""
    info = {
        'shape': tuple(cube_raw.shape[:2]),
        'bands': int(cube_raw.shape[2]),
        'dtype_raw': str(cube_raw.dtype),
        'value_range': (float(cube_raw.min()), float(cube_raw.max())),
        'num_classes': int(gt.max()),
        'labeled_pixels': int((gt > 0).sum()),
        'background_pixels': int((gt == 0).sum()),
        'per_class': {int(c): int((gt == c).sum()) for c in range(1, gt.max() + 1)},
    }
    return info
