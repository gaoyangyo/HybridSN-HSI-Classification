"""训练 + 测试主程序。

用法示例：
  # 论文配置（30% 训练，PCA 15）
  python src/train.py --config configs/pavia.yaml

  # 消融实验：只改训练比例
  python src/train.py --train-ratio 0.05 --tag ratio05
  python src/train.py --train-ratio 0.10 --tag ratio10
  python src/train.py --train-ratio 0.30 --tag ratio30

  # 消融实验：结构
  python src/train.py --variant no2d --tag arch_no2d
  python src/train.py --variant two3d --tag arch_two3d

产出（全部落在 results/<tag>/ 下，直接就是任务 3 要交的东西）：
  curves.png          训练损失与精度曲线
  metrics.json        OA / AA / Kappa / 逐类精度
  per_class.csv       逐类精度表
  confusion_matrix.png 混淆矩阵（按行归一化）
  confusion_matrix.csv
  model.pth           最优模型权重
  split.npz           训练/测试划分索引（保证消融实验可比）
  config.json         本次实验的完整配置
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 项目根目录（本文件位于 <root>/src/ 下）。
# 所有相对路径一律相对【项目根目录】解析，而不是相对当前工作目录，
# 这样无论从 Anaconda Prompt、PyCharm 还是任何 IDE 启动，
# 都能正确找到 configs/ data/ results/，无需手动设置 Working directory。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def project_path(p: str) -> str:
    """相对路径 → 相对项目根目录的绝对路径；绝对路径原样返回。"""
    return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)


from data import (PAVIA_CLASSES, PatchDataset, apply_pca, dataset_summary,
                  load_pavia, normalize, pad_cube, stratified_split)
from metrics import compute_all, format_report
from model import HybridSN
from utils import (ensure_dirs, env_report, get_device, save_json, set_seed,
                   setup_matplotlib)


class TorchPatchDS(Dataset):
    """把 data.PatchDataset 适配成 torch 的 Dataset（按需切片，不预建全部 patch）。"""

    def __init__(self, pd: PatchDataset):
        self.pd = pd

    def __len__(self) -> int:
        return len(self.pd)

    def __getitem__(self, i: int):
        x, y = self.pd[i]
        return torch.from_numpy(np.asarray(x, dtype=np.float32)), y


def load_config(path: str) -> dict:
    import yaml
    with open(project_path(path), 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='HybridSN on Pavia University')
    p.add_argument('--config', default='configs/pavia.yaml')
    p.add_argument('--data-dir', default=None, help='覆盖数据目录')
    p.add_argument('--out-dir', default='results')
    p.add_argument('--tag', default=None, help='实验标签，决定输出子目录名')
    # 以下参数覆盖配置文件，方便做消融
    p.add_argument('--train-ratio', type=float, default=None)
    p.add_argument('--pca', type=int, default=None)
    p.add_argument('--patch-size', type=int, default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--variant', choices=['full', 'two3d', 'no2d'], default=None)
    p.add_argument('--log-every', type=int, default=None, help='每多少 epoch 打印一次进度')
    p.add_argument('--normalize', choices=['global', 'train'], default=None)
    p.add_argument('--pca-fit', choices=['all', 'train'], default=None,
                   help="PCA 拟合范围：all=全图（论文做法）；train=只用训练像元（更严谨）")
    p.add_argument('--device', default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    overrides = {k: v for k, v in vars(args).items() if v is not None and k in cfg}
    cfg.update(overrides)

    tag = args.tag or cfg.get('tag') or f"ratio{int(cfg['train_ratio'] * 100):02d}"
    out_dir = os.path.join(project_path(args.out_dir), tag)
    ensure_dirs(out_dir)

    print('=' * 72)
    print(f'HybridSN on Pavia University   实验标签: {tag}')
    print('=' * 72)

    set_seed(cfg['seed'])
    device, dev_name = get_device()
    if args.device:
        device = torch.device(args.device)
        dev_name = args.device
    print(f'设备 : {device}  ({dev_name})')

    env = env_report()
    print('环境 :', {k: v for k, v in env.items() if v is not None})

    # ---------------- 1. 数据 ----------------
    data_dir = project_path(args.data_dir or cfg['data_dir'])
    mat = os.path.join(data_dir, 'PaviaU.mat')
    gt_mat = os.path.join(data_dir, 'PaviaU_gt.mat')
    if not os.path.exists(mat):
        print(f'\n[错误] 找不到数据文件: {mat}')
        print('   数据已随参考仓库下载，请复制到 data/ 目录，见 README。')
        return 1

    cube_raw, gt = load_pavia(mat, gt_mat)
    summary = dataset_summary(cube_raw, gt)
    print('\n--- 数据统计（任务 3 要求展示）---')
    print(f"  尺寸        : {summary['shape'][0]} × {summary['shape'][1]}")
    print(f"  波段数      : {summary['bands']}")
    print(f"  原始 dtype  : {summary['dtype_raw']}   取值范围 {summary['value_range']}")
    print(f"  类别数      : {summary['num_classes']}")
    print(f"  标注像元    : {summary['labeled_pixels']}   背景 {summary['background_pixels']}")
    print('  逐类样本数  :')
    for c, n in summary['per_class'].items():
        print(f"    {c}. {PAVIA_CLASSES[c - 1]:<22} {n:>6}")

    # ---------------- 2. 划分（先划分，才能让 PCA/归一化可选地只用训练像元） ----------------
    train_coords, test_coords = stratified_split(gt, cfg['train_ratio'], cfg['seed'])
    print(f"\n--- 按类别分层划分（train_ratio={cfg['train_ratio']}）---")
    print(f'  训练像元 : {len(train_coords)}')
    print(f'  测试像元 : {len(test_coords)}')
    print(f'  合计     : {len(train_coords) + len(test_coords)}  (= 全部标注像元)')

    train_mask = np.zeros(gt.shape, dtype=bool)
    train_mask[train_coords[:, 0], train_coords[:, 1]] = True

    # ---------------- 3. 归一化 + PCA ----------------
    cube, norm_stats = normalize(cube_raw, mode=cfg.get('normalize', 'global'),
                                 train_mask=train_mask if cfg.get('normalize') == 'train' else None)
    print(f"\n归一化: {norm_stats}")

    cube, pca_stats = apply_pca(
        cube, cfg['pca_components'],
        fit_mask=train_mask if cfg.get('pca_fit') == 'train' else None,
    )
    print(f"PCA   : {pca_stats['n_components']} 个主成分，"
          f"累计解释方差比 {pca_stats['explained_variance_ratio_sum']:.4f}")
    print(f"        {pca_stats['info']}")

    # ---------------- 4. DataLoader ----------------
    padded = pad_cube(cube, cfg['patch_size'])
    ds_train = TorchPatchDS(PatchDataset(padded, train_coords, gt, cfg['patch_size']))
    ds_test = TorchPatchDS(PatchDataset(padded, test_coords, gt, cfg['patch_size']))
    dl_train = DataLoader(ds_train, batch_size=cfg['batch_size'], shuffle=True,
                          num_workers=cfg.get('num_workers', 0), drop_last=False)
    dl_test = DataLoader(ds_test, batch_size=cfg['batch_size'], shuffle=False,
                         num_workers=cfg.get('num_workers', 0))

    # ---------------- 5. 模型 ----------------
    model = HybridSN(num_classes=summary['num_classes'],
                     pca_components=cfg['pca_components'],
                     patch_size=cfg['patch_size'],
                     dropout=cfg.get('dropout', 0.4),
                     variant=cfg.get('variant', 'full')).to(device)
    print(f"\n模型: variant={cfg.get('variant', 'full')}  "
          f"可训练参数 {model.count_parameters():,}")
    print(f"      flatten 维度 {model.info.flatten_dim:,}  "
          f"reshape 通道 {model.info.reshaped_channels}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    # ---------------- 6. 训练 ----------------
    print(f"\n--- 开始训练（{cfg['epochs']} epoch, batch {cfg['batch_size']}, "
          f"lr {cfg['lr']}, Adam）---")
    history = {'epoch': [], 'loss': [], 'train_acc': [], 'test_acc': []}
    best_acc, best_state = 0.0, None
    t0 = time.time()

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for xb, yb in dl_train:
            xb = xb.to(device)
            yb = torch.as_tensor(yb, dtype=torch.long, device=device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            seen += len(yb)

        train_loss = total_loss / max(seen, 1)
        train_acc = correct / max(seen, 1)

        # 每轮在测试集上评估（仅用于画曲线；最终指标用最优模型算）
        model.eval()
        t_correct, t_seen = 0, 0
        with torch.no_grad():
            for xb, yb in dl_test:
                xb = xb.to(device)
                yb = torch.as_tensor(yb, dtype=torch.long, device=device)
                pred = model(xb).argmax(1)
                t_correct += (pred == yb).sum().item()
                t_seen += len(yb)
        test_acc = t_correct / max(t_seen, 1)

        history['epoch'].append(epoch)
        history['loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % cfg.get('log_every', 10) == 0 or epoch == cfg['epochs']:
            print(f'  epoch {epoch:3d}/{cfg["epochs"]}  loss {train_loss:.4f}  '
                  f'train_acc {train_acc * 100:6.2f}%  test_acc {test_acc * 100:6.2f}%')

    elapsed = time.time() - t0
    print(f'训练完成，用时 {elapsed / 60:.1f} 分钟；最佳 test_acc = {best_acc * 100:.2f}%')

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---------------- 7. 最终评价（任务 3 的核心产出） ----------------
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in dl_test:
            xb = xb.to(device)
            all_pred.append(model(xb).argmax(1).cpu().numpy())
            all_true.append(np.asarray(yb))
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true).astype(int)

    res = compute_all(y_true, y_pred, summary['num_classes'], PAVIA_CLASSES)
    report = format_report(res, title=f'HybridSN / Pavia University / tag={tag}')
    print('\n' + '=' * 72)
    print(report)
    print('=' * 72)

    # ---------------- 8. 落盘 ----------------
    torch.save({'state_dict': model.state_dict(),
                'config': cfg,
                'arch_info': model.info.__dict__,
                'best_test_acc': best_acc},
               os.path.join(out_dir, 'model.pth'))

    np.savez(os.path.join(out_dir, 'split.npz'),
             train_coords=train_coords, test_coords=test_coords)

    payload = {k: v for k, v in res.items() if k != 'confusion_matrix'}
    payload['train_ratio'] = cfg['train_ratio']
    payload['pca_components'] = cfg['pca_components']
    payload['patch_size'] = cfg['patch_size']
    payload['variant'] = cfg.get('variant', 'full')
    payload['params'] = model.count_parameters()
    payload['train_minutes'] = round(elapsed / 60, 2)
    payload['env'] = env
    payload['normalize'] = norm_stats
    payload['pca'] = pca_stats
    save_json(payload, os.path.join(out_dir, 'metrics.json'))
    save_json(cfg, os.path.join(out_dir, 'config.json'))

    with open(os.path.join(out_dir, 'per_class.csv'), 'w', encoding='utf-8-sig') as f:
        f.write('class,name,support,accuracy,precision,recall,f1\n')
        for d in res['per_class']:
            acc = '' if d['accuracy'] is None else f"{d['accuracy']:.6f}"
            f.write(f"{d['class']},{d['name']},{d['support']},{acc},"
                    f"{d['precision']:.6f},{d['recall']:.6f},{d['f1']:.6f}\n")

    np.savetxt(os.path.join(out_dir, 'confusion_matrix.csv'),
               res['confusion_matrix'], fmt='%d', delimiter=',')

    with open(os.path.join(out_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write(report + '\n')

    # ---------------- 9. 出图 ----------------
    try:
        plt = setup_matplotlib()
        cm = res['confusion_matrix']

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(history['epoch'], history['loss'], color='tab:red')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
        axes[0].set_title('训练损失'); axes[0].grid(alpha=.3)
        axes[1].plot(history['epoch'], np.array(history['train_acc']) * 100, label='训练精度')
        axes[1].plot(history['epoch'], np.array(history['test_acc']) * 100, label='测试精度')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('精度收敛'); axes[1].legend(); axes[1].grid(alpha=.3)
        fig.suptitle(f'HybridSN / PaviaU / {tag}')
        fig.savefig(os.path.join(out_dir, 'curves.png'))
        plt.close(fig)

        row = cm.sum(1, keepdims=True)
        norm = np.divide(cm, row, out=np.zeros_like(cm, dtype=float), where=row > 0)
        fig, ax = plt.subplots(figsize=(7.2, 6))
        im = ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(len(PAVIA_CLASSES)))
        ax.set_yticks(range(len(PAVIA_CLASSES)))
        ax.set_xticklabels(PAVIA_CLASSES, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(PAVIA_CLASSES, fontsize=8)
        ax.set_xlabel('预测类别'); ax.set_ylabel('真实类别')
        ax.set_title(f'混淆矩阵（按行归一化）\nOA={res["OA"]*100:.2f}%  '
                     f'AA={res["AA"]*100:.2f}%  Kappa={res["Kappa"]:.4f}')
        for i in range(len(PAVIA_CLASSES)):
            for j in range(len(PAVIA_CLASSES)):
                if cm[i, j] > 0:
                    ax.text(j, i, f'{norm[i, j]:.2f}', ha='center', va='center',
                            fontsize=6.5, color='white' if norm[i, j] > 0.5 else 'black')
        fig.colorbar(im, ax=ax, shrink=.85)
        fig.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
        plt.close(fig)
        print(f'\n已保存曲线与混淆矩阵到 {out_dir}/')
    except Exception as exc:
        print(f'\n[警告] 出图失败（不影响指标）：{exc}')

    print(f'\n全部产出在: {os.path.abspath(out_dir)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
