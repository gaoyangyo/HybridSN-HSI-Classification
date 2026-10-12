# -*- coding: utf-8 -*-
"""MATLAB .mat 文件查看器 —— 打印结构 + 可视化

用法：
    python view_mat.py                          # 查看 data/ 下所有 .mat
    python view_mat.py 某个文件.mat              # 查看指定文件（可多个）
    python view_mat.py --no-plot 某文件.mat      # 只看结构，不出图

说明：.mat 是 MATLAB 的二进制数据格式，高光谱数据集常用它。
      Windows 把它显示成 "Microsoft Access Table" 是错误关联，不用管。
"""
import os
import sys
import glob

import numpy as np
import scipy.io as sio

NO_PLOT = '--no-plot' in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith('--')]

if args:
    files = args
else:
    here = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(here, 'data', '*.mat')))
    if not files:
        files = sorted(glob.glob('*.mat'))

if not files:
    print('没有找到 .mat 文件。用法： python view_mat.py 文件名.mat')
    sys.exit(1)


def human(n):
    for u in ['B', 'KB', 'MB', 'GB']:
        if n < 1024:
            return '{:.1f} {}'.format(n, u)
        n /= 1024
    return '{:.1f} TB'.format(n)


def describe(path):
    print('=' * 74)
    print('文件 : ' + path)
    print('大小 : ' + human(os.path.getsize(path)))
    print('=' * 74)

    md = sio.loadmat(path)
    keys = [k for k in md if not k.startswith('__')]
    print('包含 {} 个变量: {}'.format(len(keys), ', '.join(keys)))
    print('')

    arrays = {}
    for k in keys:
        v = md[k]
        if not isinstance(v, np.ndarray):
            print('  {:<28} 类型={}  值={}'.format(k, type(v).__name__, v))
            continue
        arrays[k] = v
        print('  变量名 : {}'.format(k))
        print('    形状   : {}'.format(v.shape))
        print('    数据类型: {}'.format(v.dtype))
        print('    元素个数: {:,}'.format(v.size))
        print('    数值范围: {} ~ {}'.format(v.min(), v.max()))
        if v.ndim == 2 and v.dtype.kind in 'iu' and 0 < v.max() < 300:
            vals, cnt = np.unique(v, return_counts=True)
            labeled = int(cnt[vals > 0].sum()) if (vals == 0).any() else int(v.size)
            print('    唯一值个数: {}   (0 通常表示背景/未标注)'.format(len(vals)))
            print('    非零像元数: {:,}'.format(labeled))
            if len(vals) <= 25:
                print('    逐类像元数:')
                for val, c in zip(vals, cnt):
                    tag = '背景' if val == 0 else '类别 {}'.format(val)
                    print('      {:<10} {:>9,}'.format(tag, c))
        print('')
    return arrays


def visualize(path, arrays):
    try:
        os.environ.setdefault('MPLCONFIGDIR',
                              os.path.join(os.path.dirname(os.path.abspath(path)), '.mplconfig'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as exc:
        print('[警告] 无法加载 matplotlib，跳过出图: {}'.format(exc))
        return

    cube = None
    labels = None
    for k, v in arrays.items():
        if v.ndim == 3 and cube is None:
            cube = (k, v)
        elif v.ndim == 2 and v.dtype.kind in 'iu' and labels is None:
            labels = (k, v)

    panels = []
    if cube is not None:
        k, v = cube
        d = v.shape[2]
        bands = [b for b in (int(d * 0.15), int(d * 0.5), int(d * 0.85)) if 0 <= b < d] or [0, 1, 2]
        rgb = v[:, :, bands].astype('float32')
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-9)
        panels.append(('伪彩色影像\n{}  波段 {}'.format(k, bands), rgb, None))
    if labels is not None:
        k, v = labels
        panels.append(('真值标签\n{}  共 {} 类'.format(k, int(v.max())), v, 'tab20'))

    if not panels:
        print('[提示] 没有可可视化的三维数据或标签图')
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.2))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, img, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, interpolation='nearest')
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    out = os.path.splitext(os.path.basename(path))[0] + '_预览.png'
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print('已出图: ' + os.path.abspath(out))


for f in files:
    if not os.path.exists(f):
        print('文件不存在: ' + f)
        continue
    arrays = describe(f)
    if not NO_PLOT:
        visualize(f, arrays)
    print('')
