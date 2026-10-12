"""汇总 results/ 下所有实验，自动生成对比表（任务 5 的产出）。

用法：
  python src/summarize.py                      # 扫描 results/
  python src/summarize.py --out results        # 指定目录

产出：
  results/compare.md     Markdown 对比表（可直接贴进报告）
  results/compare.csv    CSV（可导入 Excel 画图）
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os


def collect(results_dir: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, '*', 'metrics.json'))):
        tag = os.path.basename(os.path.dirname(path))
        try:
            with open(path, 'r', encoding='utf-8') as f:
                m = json.load(f)
        except Exception as exc:
            print(f'跳过 {path}: {exc}')
            continue
        rows.append({
            'tag': tag,
            'variant': m.get('variant', 'full'),
            'train_ratio': m.get('train_ratio'),
            'pca': m.get('pca_components'),
            'patch': m.get('patch_size'),
            'params': m.get('params'),
            'OA': m.get('OA'),
            'AA': m.get('AA'),
            'Kappa': m.get('Kappa'),
            'minutes': m.get('train_minutes'),
        })
    return rows


def fmt_pct(value) -> str:
    return '—' if value is None else f'{value * 100:.2f}'


def fmt_num(value, digits: int = 4) -> str:
    return '—' if value is None else f'{value:.{digits}f}'


def fmt_ratio(value) -> str:
    return '—' if value is None else f'{value:.2f}'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results')
    args = ap.parse_args()

    rows = collect(args.out)
    if not rows:
        print(f'{args.out}/ 下还没有任何实验结果，先跑： python src/train.py')
        return 1

    rows.sort(key=lambda r: (r['variant'], r['train_ratio'] if r['train_ratio'] is not None else 0))

    os.makedirs(args.out, exist_ok=True)

    # ---- CSV ----
    csv_path = os.path.join(args.out, 'compare.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if v is None else v) for k, v in r.items()})

    # ---- Markdown ----
    header = ('| 实验标签 | 变体 | 训练比例 | PCA | Patch | 参数量 '
              '| OA (%) | AA (%) | Kappa | 训练(分) |')
    sep = '|---|---|---|---|---|---|---|---|---|---|'
    lines = ['# 实验结果对比', '', header, sep]
    for r in rows:
        params = '—' if r['params'] is None else f"{r['params']:,}"
        lines.append(
            '| {tag} | {variant} | {ratio} | {pca} | {patch} | {params} | {oa} | {aa} | {kappa} | {mins} |'.format(
                tag=r['tag'], variant=r['variant'], ratio=fmt_ratio(r['train_ratio']),
                pca=r['pca'], patch=r['patch'], params=params,
                oa=fmt_pct(r['OA']), aa=fmt_pct(r['AA']),
                kappa=fmt_num(r['Kappa']), mins=r['minutes'],
            )
        )

    md_path = os.path.join(args.out, 'compare.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print('\n'.join(lines))
    print(f'\n已保存: {md_path}')
    print(f'已保存: {csv_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
