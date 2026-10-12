"""评价指标：OA / AA / Kappa / 逐类精度 / 混淆矩阵。

论文对三项指标的定义（任务 1 第 5 问）：
  OA    = 分对的像元 / 总测试像元           —— 会被大类主导
  AA    = 先算每类精度再对类别取平均         —— 每类等权，能暴露小类被牺牲
  Kappa = (Po - Pe) / (1 - Pe)             —— 扣除随机一致，衡量一致性
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def overall_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def average_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                     num_classes: int | None = None) -> float:
    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max())) + 1
    accs = []
    for c in range(num_classes):
        m = (y_true == c)
        if m.sum() > 0:
            accs.append(float((y_pred[m] == c).mean()))
    return float(np.mean(accs)) if accs else 0.0


def kappa_coefficient(y_true: np.ndarray, y_pred: np.ndarray,
                      num_classes: int | None = None) -> float:
    """普通 Kappa（论文用的就是这个，不是二次加权 Kappa）。"""
    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max())) + 1
    cm = confusion_matrix(y_true, y_pred, num_classes)
    n = cm.sum()
    if n == 0:
        return 0.0
    po = np.trace(cm) / n
    pe = float((cm.sum(0) * cm.sum(1)).sum()) / (n * n)
    return float((po - pe) / (1 - pe)) if pe != 1 else 0.0


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                       num_classes: int | None = None) -> list[float | None]:
    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max())) + 1
    out: list[float | None] = []
    for c in range(num_classes):
        m = (y_true == c)
        out.append(float((y_pred[m] == c).mean()) if m.sum() > 0 else None)
    return out


def compute_all(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int,
                class_names: Sequence[str] | None = None) -> dict:
    """一次性算出全部指标，返回可直接落盘/打印的字典。"""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, num_classes)
    pc = per_class_accuracy(y_true, y_pred, num_classes)

    # Precision / Recall / F1（补充材料表 5 就是这个口径）
    per_class_detail = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class_detail.append({
            'class': c + 1,
            'name': class_names[c] if class_names and c < len(class_names) else f'class_{c + 1}',
            'support': int((y_true == c).sum()),
            'accuracy': pc[c],
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
        })

    return {
        'OA': overall_accuracy(y_true, y_pred),
        'AA': average_accuracy(y_true, y_pred, num_classes),
        'Kappa': kappa_coefficient(y_true, y_pred, num_classes),
        'n_test': int(len(y_true)),
        'confusion_matrix': cm,
        'per_class': per_class_detail,
    }


def format_report(res: dict, title: str = '') -> str:
    """把指标渲染成可直接贴进报告的文本表。"""
    lines = []
    if title:
        lines.append(title)
    lines.append(f"样本数 : {res['n_test']}")
    lines.append(f"OA     : {res['OA'] * 100:.2f} %")
    lines.append(f"AA     : {res['AA'] * 100:.2f} %")
    lines.append(f"Kappa  : {res['Kappa']:.4f}")
    lines.append('')
    lines.append(f"{'类别':<4}{'名称':<24}{'样本':>7}{'精度%':>9}{'精确率':>9}{'召回率':>9}{'F1':>8}")
    lines.append('-' * 72)
    for d in res['per_class']:
        acc = '  n/a  ' if d['accuracy'] is None else f"{d['accuracy'] * 100:6.2f}"
        lines.append(
            f"{d['class']:<4}{d['name']:<24}{d['support']:>7}{acc:>9}"
            f"{d['precision']:>9.4f}{d['recall']:>9.4f}{d['f1']:>8.4f}"
        )
    return '\n'.join(lines)
