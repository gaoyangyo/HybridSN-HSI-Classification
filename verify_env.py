# -*- coding: utf-8 -*-
"""HybridSN 复现环境验证程序

逐项检查环境是否配置完整，并输出一张清晰的报告。
不修改任何文件，只读取和计算。

用法：
    D:\\Users\\conda\\envs\\hybridsn\\python.exe verify_env.py
或双击桌面上的「验证HybridSN环境.bat」
"""
import os
import sys
import time
import platform

# 项目根目录（本文件所在目录），不依赖当前工作目录
PROJ = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJ)

# matplotlib 的缓存目录必须在**导入 matplotlib 之前**设定，
# 否则它会去写用户 AppData，在受限环境下会被拒绝并退回临时目录。
os.environ.setdefault('MPLCONFIGDIR', os.path.join(PROJ, '.mplconfig'))

PASS, FAIL, WARN = '[OK]  ', '[FAIL]', '[警告]'
results = []          # (是否关键项, 通过与否, 说明)


def check(name, ok, detail='', critical=True):
    tag = PASS if ok else (FAIL if critical else WARN)
    line = '  {0} {1:<22}{2}'.format(tag, name, detail)
    print(line)
    results.append((critical, ok, name))
    return ok


def section(title):
    print('')
    print('-' * 74)
    print(' ' + title)
    print('-' * 74)


# ======================================================================
print('=' * 74)
print('           HybridSN 复现环境验证程序')
print('=' * 74)
print('  运行时间    : ' + time.strftime('%Y-%m-%d %H:%M:%S'))
print('  Python 路径 : ' + sys.executable)
print('  项目目录    : ' + PROJ)

# ======================================================================
section('一、解释器与依赖库')
pyver = sys.version.split()[0]
check('Python 版本', pyver.startswith('3.12') or pyver.startswith('3.13') or pyver.startswith('3.14'),
      pyver + '   (3.12 为推荐版本)')

mods = {}
for name, attr in [('torch', '__version__'), ('numpy', '__version__'), ('scipy', '__version__'),
                   ('sklearn', '__version__'), ('matplotlib', '__version__')]:
    try:
        m = __import__(name)
        mods[name] = getattr(m, attr, '?')
        check(name, True, mods[name])
    except Exception as exc:
        check(name, False, '导入失败: ' + str(exc)[:40])
try:
    import yaml
    check('PyYAML', True, yaml.__version__)
except Exception as exc:
    check('PyYAML', False, '导入失败: ' + str(exc)[:40])

# ======================================================================
section('二、GPU 与 CUDA')
torch = None
try:
    import torch as _t
    torch = _t
except Exception:
    pass

gpu_ok = False
if torch is None:
    check('PyTorch 可用', False, '未安装，后续 GPU 检查跳过')
else:
    cuda = torch.cuda.is_available()
    gpu_ok = cuda
    check('CUDA 可用', cuda, str(cuda))
    if cuda:
        check('CUDA 版本', True, str(torch.version.cuda))
        check('GPU 型号', True, torch.cuda.get_device_name(0))
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        check('显存', vram >= 2, '{:.1f} GB'.format(vram))
        # 真实算一次，证明不是"认出来但不能算"
        try:
            x = torch.randn(2000, 2000, device='cuda')
            t0 = time.time()
            for _ in range(20):
                torch.mm(x, x)
            torch.cuda.synchronize()
            dt = time.time() - t0
            check('GPU 实算测试', True, '2000x2000 矩阵乘 20 次 = {:.2f} 秒'.format(dt))
        except Exception as exc:
            check('GPU 实算测试', False, str(exc)[:40])
        del x
        torch.cuda.empty_cache()
    else:
        check('GPU 可用', False, '将退化为 CPU 训练（很慢）')

# ======================================================================
section('三、数据文件')
data_dir = os.path.join(PROJ, 'data')
mat = os.path.join(data_dir, 'PaviaU.mat')
gtf = os.path.join(data_dir, 'PaviaU_gt.mat')
check('data/PaviaU.mat', os.path.exists(mat),
      '{:.1f} MB'.format(os.path.getsize(mat) / 1024 / 1024) if os.path.exists(mat) else '缺失')
check('data/PaviaU_gt.mat', os.path.exists(gtf),
      '{:.2f} MB'.format(os.path.getsize(gtf) / 1024 / 1024) if os.path.exists(gtf) else '缺失')

gt = None
if os.path.exists(mat) and os.path.exists(gtf):
    try:
        import scipy.io as sio
        cube = sio.loadmat(mat)['paviaU']
        gt = sio.loadmat(gtf)['paviaU_gt'].astype('int64')
        check('数据立方体尺寸', cube.shape == (610, 340, 103), str(cube.shape))
        check('原始数据类型', str(cube.dtype) == 'uint16', str(cube.dtype) + '  范围 %d~%d' % (cube.min(), cube.max()))
        check('类别数', int(gt.max()) == 9, str(gt.max()))
        check('标注像元总数', int((gt > 0).sum()) == 42776, str(int((gt > 0).sum())))
    except Exception as exc:
        check('数据读取', False, str(exc)[:50])

# ======================================================================
section('四、代码文件')
for f in ['src/model.py', 'src/data.py', 'src/train.py', 'src/predict.py',
          'src/metrics.py', 'src/utils.py', 'src/summarize.py',
          'configs/pavia.yaml', 'requirements.txt', 'README.md', 'quick_test.py']:
    p = os.path.join(PROJ, f.replace('/', os.sep))
    check(f, os.path.exists(p), '{:,} 字节'.format(os.path.getsize(p)) if os.path.exists(p) else '缺失',
          critical=False)

# ======================================================================
section('五、功能自检（真正跑一遍网络）')
model_ok = False
if torch is not None:
    try:
        import torch.nn as nn

        class HybridSN(nn.Module):
            def __init__(self, n_cls, bands, patch, dropout=0.4):
                super().__init__()
                # Conv3d 输入是 (N,C,D,H,W)，D 为光谱维 -> 核顺序 (光谱深, 空间高, 空间宽)
                self.c3 = nn.ModuleList([nn.Conv3d(1, 8, (7, 3, 3)),
                                         nn.Conv3d(8, 16, (5, 3, 3)),
                                         nn.Conv3d(16, 32, (3, 3, 3))])
                sp, bd = patch - 6, bands - 12
                self.sp, self.ch = sp, 32 * bd
                self.c2 = nn.Conv2d(self.ch, 64, 3)
                flat = (sp - 2) ** 2 * 64
                self.fc1, self.do1 = nn.Linear(flat, 256), nn.Dropout(dropout)
                self.fc2, self.do2 = nn.Linear(256, 128), nn.Dropout(dropout)
                self.fc3 = nn.Linear(128, n_cls)

            def forward(self, x):
                for c in self.c3:
                    x = torch.relu(c(x))
                n = x.size(0)
                x = x.permute(0, 2, 1, 3, 4).reshape(n, self.ch, self.sp, self.sp)
                x = torch.relu(self.c2(x))
                return self.fc3(self.do2(torch.relu(self.fc2(self.do1(torch.relu(self.fc1(x.reshape(n, -1))))))))

        dev = torch.device('cuda' if gpu_ok else 'cpu')
        net = HybridSN(9, 15, 25).to(dev)
        n_par = sum(p.numel() for p in net.parameters() if p.requires_grad)
        check('网络构建', n_par == 4844793, '可训练参数 {:,}  (论文配置值 4,844,793)'.format(n_par))
        model_ok = (n_par == 4844793)
        with torch.no_grad():
            probe = torch.randn(2, 1, 15, 25, 25, device=dev)
            shape = tuple(net(probe).shape)
        check('前向传播', shape == (2, 9), '(2,1,15,25,25) -> ' + str(shape))
    except Exception as exc:
        check('模型自检', False, str(exc)[:60])

    # 真的训练几批，证明能收敛
    if model_ok and gt is not None:
        try:
            import numpy as np
            from sklearn.decomposition import PCA
            import scipy.io as sio

            t0 = time.time()
            cube = sio.loadmat(mat)['paviaU'].astype('float32') / 8000.0
            M, N, D = cube.shape
            data = PCA(n_components=15).fit_transform(
                cube.reshape(-1, D)).reshape(M, N, 15).astype('float32')

            rows, cols = np.where(gt > 0)
            idx = np.random.default_rng(42).choice(len(rows), size=min(3000, len(rows)), replace=False)
            rows, cols = rows[idx], cols[idx]
            pad = 12
            padded = np.pad(data, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
            X = np.stack([padded[r:r + 25, c:c + 25, :].transpose(2, 0, 1)[None] for r, c in zip(rows, cols)])
            y = np.array([gt[r, c] - 1 for r, c in zip(rows, cols)])
            print('  [..]   取 {} 个样本做一次迷你训练测试，请稍候...'.format(len(X)))

            net = HybridSN(9, 15, 25).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            crit = nn.CrossEntropyLoss()
            Xt = torch.from_numpy(X).float()
            yt = torch.from_numpy(y).long()
            losses = []
            net.train()
            for ep in range(2):
                perm = torch.randperm(len(Xt))
                tot = 0.0
                for i in range(0, len(perm), 256):
                    b = perm[i:i + 256]
                    opt.zero_grad()
                    out = net(Xt[b].to(dev))
                    loss = crit(out, yt[b].to(dev))
                    loss.backward()
                    opt.step()
                    tot += loss.item() * len(b)
                losses.append(tot / len(perm))
            ok = losses[1] < losses[0]
            check('迷你训练测试', ok,
                  'loss {:.4f} -> {:.4f}（下降说明能训练）  用时 {:.1f} 秒'.format(
                      losses[0], losses[1], time.time() - t0))
        except Exception as exc:
            check('迷你训练测试', False, str(exc)[:60])

# ======================================================================
print('')
print('=' * 74)
crit_fail = [r for r in results if r[0] and not r[1]]
warn_fail = [r for r in results if not r[0] and not r[1]]
total = len(results)
passed = len([r for r in results if r[1]])
print('  检查项合计: {}   通过: {}   失败: {}'.format(total, passed, total - passed))
if not crit_fail and not warn_fail:
    print('')
    print('  结论: [OK] 环境配置完整，可以开始实验')
elif not crit_fail:
    print('')
    print('  结论: [OK] 关键项全部通过（有 {} 个非关键文件缺失，不影响运行）'.format(len(warn_fail)))
else:
    print('')
    print('  结论: [FAIL] 有 {} 个关键项未通过:'.format(len(crit_fail)))
    for _, _, name in crit_fail:
        print('        - ' + name)
print('=' * 74)
print('')
print('  下一步：在 Anaconda Prompt 里执行')
print('')
print('      conda activate hybridsn')
print('      cd /d ' + PROJ)
print('      python src/train.py --config configs/pavia.yaml')
print('')
print('  （3 轮快速试跑： python src/train.py --epochs 3 --tag 试跑 ）')
print('')
