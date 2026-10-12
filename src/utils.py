"""通用工具：随机种子、设备选择、目录管理、绘图风格。"""
from __future__ import annotations

import json
import os
import random
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# ★ 必须在任何模块导入 matplotlib 之前设定 MPLCONFIGDIR。
#   matplotlib 在 import 的瞬间就会读取该变量并据此决定缓存目录；若此时尚未设置，
#   它会尝试写入用户 AppData，在受限环境（如只读沙箱）中会被拒绝，于是退回临时目录
#   并打印警告。注意 env_report() 里也会 __import__('matplotlib')，因此这一设定
#   只能放在模块级，不能放在 setup_matplotlib() 内部。
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.environ.get('MPLCONFIGDIR'):
    os.environ['MPLCONFIGDIR'] = os.path.join(_PROJECT_ROOT, '.mplconfig')
try:
    os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
except OSError as _exc:      # 只警告，不因为画不了图就让整个实验失败
    print(f'[警告] 无法创建 matplotlib 缓存目录 {os.environ["MPLCONFIGDIR"]}: {_exc}')


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """把三个随机源全部固定，保证实验可复现。

    任务书要求"可重复运行的代码环境"，这一步是硬性要求：
    不固定种子，你的消融实验之间就没有可比性。
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device():
    """有 GPU 用 GPU，没有就用 CPU（CPU 也能跑完，只是慢）。"""
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda'), f'CUDA / {torch.cuda.get_device_name(0)}'
    return torch.device('cpu'), 'CPU'


def env_report() -> dict:
    """任务 2 要求记录的完整环境信息。"""
    import platform

    info = {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'processor': platform.processor(),
    }
    try:
        import torch
        info['torch'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        info['cuda_version'] = torch.version.cuda
        if torch.cuda.is_available():
            info['gpu'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
    except ImportError:
        info['torch'] = None
    for m in ('numpy', 'scipy', 'sklearn', 'matplotlib'):
        try:
            info[m] = __import__(m).__version__
        except ImportError:
            info[m] = None
    return info


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime('%Y%m%d-%H%M%S')


def save_json(obj: dict, path: str) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=default)


def setup_matplotlib():
    """统一绘图风格；并解决中文与负号显示问题。

    MPLCONFIGDIR 已在本模块导入时设定（见文件顶部），此处不再重复。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 130
    plt.rcParams['savefig.bbox'] = 'tight'
    return plt
