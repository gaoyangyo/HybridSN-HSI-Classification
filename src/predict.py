"""全图预测与结果出图（任务 4）。

用法：
  python src/predict.py --exp results/paper_30pct          # 用训练好的模型
  python src/predict.py --exp results/paper_30pct --scope all

产出（results/<tag>/maps/ 下）：
  panel.png        四联图：伪彩色 / Ground Truth / 分类结果 / 错误分布
  cmap.npy         整幅分类标签图（0=背景）
  pred_labeled.npy 仅标注像元的预测标签

★ 内存提示：整幅 610×340 = 207,400 个像元的 patch 约 7.8 GB，
  绝对不能一次性构建，必须像这里一样按 batch 边造边推。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import (PAVIA_CLASSES, PatchDataset, apply_pca, build_all_coords,
                  load_pavia, normalize, pad_cube, stratified_split)
from metrics import compute_all, format_report
from model import HybridSN
from utils import ensure_dirs, get_device, save_json, set_seed, setup_matplotlib


class TorchPatchDS(Dataset):
    def __init__(self, pd):
        self.pd = pd

    def __len__(self):
        return len(self.pd)

    def __getitem__(self, i):
        x, _ = self.pd[i]
        return torch.from_numpy(np.asarray(x, dtype=np.float32))


def parse_args():
    p = argparse.ArgumentParser(description='HybridSN 全图预测')
    p.add_argument('--exp', required=True, help='训练输出目录，如 results/paper_30pct')
    p.add_argument('--data-dir', default=None)
    p.add_argument('--scope', choices=['labeled', 'all'], default='all',
                   help='labeled=只预测标注像元；all=整幅影像（默认，任务 4 要的整幅图）')
    p.add_argument('--batch-size', type=int, default=1024)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    exp = args.exp
    with open(os.path.join(exp, 'config.json'), 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    with open(os.path.join(exp, 'metrics.json'), 'r', encoding='utf-8') as f:
        train_metrics = json.load(f)

    set_seed(cfg['seed'])
    device, dev_name = get_device()
    print(f'设备: {device} ({dev_name})   实验: {exp}')

    # ---------- 用与训练完全相同的流程重建数据（保证一致性） ----------
    data_dir = args.data_dir or cfg['data_dir']
    cube_raw, gt = load_pavia(os.path.join(data_dir, 'PaviaU.mat'),
                              os.path.join(data_dir, 'PaviaU_gt.mat'))

    split = np.load(os.path.join(exp, 'split.npz'))
    train_coords, test_coords = split['train_coords'], split['test_coords']
    train_mask = np.zeros(gt.shape, dtype=bool)
    train_mask[train_coords[:, 0], train_coords[:, 1]] = True

    cube, _ = normalize(cube_raw, mode=cfg.get('normalize', 'global'),
                        train_mask=train_mask if cfg.get('normalize') == 'train' else None)
    cube, _ = apply_pca(cube, cfg['pca_components'],
                        fit_mask=train_mask if cfg.get('pca_fit') == 'train' else None)
    padded = pad_cube(cube, cfg['patch_size'])

    # ---------- 重建模型并载入权重 ----------
    ckpt = torch.load(os.path.join(exp, 'model.pth'), map_location=device, weights_only=False)
    model = HybridSN(num_classes=gt.max(), pca_components=cfg['pca_components'],
                     patch_size=cfg['patch_size'], dropout=cfg.get('dropout', 0.4),
                     variant=cfg.get('variant', 'full')).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"载入模型: variant={cfg.get('variant', 'full')}  "
          f"参数量 {model.count_parameters():,}")

    # ---------- 预测 ----------
    coords = build_all_coords(gt) if args.scope == 'all' else test_coords
    print(f'预测范围: {args.scope}   像元数: {len(coords):,}')
    ds = TorchPatchDS(PatchDataset(padded, coords, gt, cfg['patch_size']))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    preds = []
    with torch.no_grad():
        for i, xb in enumerate(dl, 1):
            out = model(xb.to(device)).argmax(1).cpu().numpy()
            preds.append(out)
            if i % 25 == 0:
                print(f'  {i * args.batch_size:,} / {len(coords):,}')
    y_pred_all = np.concatenate(preds).astype(np.int64)

    # ---------- 整幅标签图 ----------
    cmap = np.zeros(gt.shape, dtype=np.int64)
    cmap[coords[:, 0], coords[:, 1]] = y_pred_all + 1
    maps_dir = os.path.join(exp, 'maps')
    ensure_dirs(maps_dir)
    np.save(os.path.join(maps_dir, 'cmap.npy'), cmap)

    # ---------- 在测试集上算指标，并与训练时记录的值核对 ----------
    if args.scope == 'all':
        key = {(int(r), int(c)): i for i, (r, c) in enumerate(coords)}
        idx = np.array([key[(int(r), int(c))] for r, c in test_coords])
        y_pred_test = y_pred_all[idx]
    else:
        y_pred_test = y_pred_all
    y_true_test = np.array([gt[r, c] - 1 for r, c in test_coords])

    res = compute_all(y_true_test, y_pred_test, int(gt.max()), PAVIA_CLASSES)
    print('\n' + format_report(res, title=f'复算（应与训练时一致）'))
    print(f"\n训练时记录: OA={train_metrics['OA'] * 100:.2f}%  "
          f"AA={train_metrics['AA'] * 100:.2f}%  Kappa={train_metrics['Kappa']:.4f}")
    print(f"本次复算  : OA={res['OA'] * 100:.2f}%  "
          f"AA={res['AA'] * 100:.2f}%  Kappa={res['Kappa']:.4f}")
    np.save(os.path.join(maps_dir, 'pred_labeled.npy'), y_pred_test)
    save_json({k: v for k, v in res.items() if k != 'confusion_matrix'},
              os.path.join(maps_dir, 'metrics_recomputed.json'))

    # ---------- 出图：任务 4 要求的伪彩色 / GT / 分类结果 + 错误分布 ----------
    try:
        plt = setup_matplotlib()
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches

        # 伪彩色：挑三个波段做 RGB
        bands = sorted({5, 30, 60}, key=lambda b: b)
        bands = [b for b in bands if b < cube_raw.shape[2]] or [10, 30, 50]
        rgb = cube_raw[:, :, bands].astype(np.float32)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-9)

        cmap_colors = plt.get_cmap('tab10', int(gt.max()))
        norm = mcolors.BoundaryNorm(np.arange(-0.5, gt.max() + 0.5, 1), cmap_colors.N)

        fig, axes = plt.subplots(2, 2, figsize=(13, 11))

        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title(f'(a) 伪彩色影像  波段 {bands}')

        axes[0, 1].imshow(gt, cmap=cmap_colors, norm=norm)
        axes[0, 1].set_title('(b) Ground Truth')

        axes[1, 0].imshow(cmap, cmap=cmap_colors, norm=norm)
        axes[1, 0].set_title(f'(c) HybridSN 分类结果\nOA={res["OA"] * 100:.2f}%  '
                             f'AA={res["AA"] * 100:.2f}%  Kappa={res["Kappa"]:.4f}')

        # 错误分布：只画测试像元中被分错的
        err = np.zeros(gt.shape, dtype=np.float32)
        mis = y_pred_test != y_true_test
        err[test_coords[mis, 0], test_coords[mis, 1]] = 1
        axes[1, 1].imshow(rgb * 0.85)
        axes[1, 1].imshow(np.ma.masked_where(err == 0, err), cmap='autumn', vmin=0, vmax=1)
        axes[1, 1].set_title(f'(d) 测试集错误分布（红点，共 {int(mis.sum())} 个）')

        for ax in axes.ravel():
            ax.set_xticks([]); ax.set_yticks([])

        handles = [mpatches.Patch(color=cmap_colors(i), label=f'{i + 1}. {n}')
                   for i, n in enumerate(PAVIA_CLASSES)]
        fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f'Pavia University 分类结果  (variant={cfg.get("variant", "full")}, '
                     f'{int(cfg["train_ratio"] * 100)}% 训练)', fontsize=13)
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        fig.savefig(os.path.join(maps_dir, 'panel.png'))
        plt.close(fig)
        print(f'\n四联图已保存: {os.path.join(maps_dir, "panel.png")}')
    except Exception as exc:
        print(f'\n[警告] 出图失败（不影响数据）：{exc}')
        import traceback
        traceback.print_exc()

    print(f'\n产出目录: {os.path.abspath(maps_dir)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
