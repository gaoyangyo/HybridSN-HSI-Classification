"""HybridSN：光谱–空间 3-D-CNN → 空间 2-D-CNN。

严格对应论文表 I（Roy et al., IEEE GRSL 17(2), 2020）的逐层结构。

★ 自检：variant='full', num_classes=16, pca_components=30, patch_size=25 时，
  可训练参数总量必须恰好等于论文报告的 5,122,176。直接运行本文件即可验证。

结构消融（任务 5 用）：
  variant='full'  : 完整 HybridSN（3 个 3-D 卷积 + 1 个 2-D 卷积）—— 论文结构
  variant='two3d' : 去掉第 3 个 3-D 卷积，验证"3-D 堆叠次数"的作用
  variant='no2d'  : 去掉 2-D 卷积，直接展平 —— 验证"为什么还要接 2-D CNN"
                    注意：这一变体参数量会暴涨（2-D 卷积正是用来压参数量的）

与官方 Keras 代码的两点差异（论文正文未明确，属实现细节）：
  1. Dropout 比例：论文只列了 Dropout 层、未给比例，此处沿用官方 Keras 代码的 0.4。
  2. 张量重排：Keras 是 channels_last，Reshape 把 (光谱维, 通道) 合并为 D*C；
     PyTorch 是 channels_first，因此用 permute 保证合并顺序与 Keras 一致。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

# 三个 3-D 卷积的输出通道与光谱维核深（论文：8×3×3×7、16×3×3×5、32×3×3×3）
CONV3D_SPECS = [(8, 7), (16, 5), (32, 3)]


@dataclass
class ArchInfo:
    """推导出来的中间形状，便于打印与自检。"""
    n_conv3d: int
    use_conv2d: bool
    spatial_after_3d: int
    bands_after_3d: int
    reshaped_channels: int
    flatten_dim: int


class HybridSN(nn.Module):
    def __init__(self, num_classes: int, pca_components: int = 15, patch_size: int = 25,
                 dropout: float = 0.4, variant: str = 'full'):
        super().__init__()
        if variant not in ('full', 'two3d', 'no2d'):
            raise ValueError(f"variant 必须是 full/two3d/no2d，收到 {variant}")

        self.num_classes = num_classes
        self.pca_components = pca_components
        self.patch_size = patch_size
        self.variant = variant
        self.n_conv3d = 2 if variant == 'two3d' else 3
        self.use_conv2d = (variant != 'no2d')

        # ---------- 3-D 卷积：卷积核在（空间, 空间, 光谱）三维滑动 ----------
        specs = CONV3D_SPECS[:self.n_conv3d]
        self.conv3d_layers = nn.ModuleList()
        in_ch = 1
        for out_ch, depth in specs:
            # ★★ 关键：PyTorch 的 Conv3d 输入布局是 (N, C, D, H, W)，D 是光谱维，
            #    因此 kernel_size 三元组的顺序是 (光谱深度, 空间高, 空间宽)，
            #    光谱深度必须放在**第一位**。
            #    若误写成 (3, 3, depth)，参数量看起来完全一样
            #    （3*3*7 == 7*3*3），但卷积会沿错误的方向压缩维度：
            #    光谱维只减 2，空间宽却减 6 —— 只验参数量根本发现不了，
            #    必须同时验证前向传播的输出形状。
            self.conv3d_layers.append(nn.Conv3d(in_ch, out_ch, kernel_size=(depth, 3, 3)))
            in_ch = out_ch
        self.last_conv3d_channels = specs[-1][0]

        # ---------- 推导中间形状（全部是 valid 卷积，无 padding） ----------
        spatial = patch_size - 2 * self.n_conv3d
        bands = pca_components - sum(d - 1 for _, d in specs)
        if bands < 1:
            raise ValueError(
                f'PCA 主成分数 {pca_components} 对 variant={variant} 太小：'
                f'3-D 卷积之后只剩 {bands} 个光谱通道。请提高 pca_components。'
            )
        self.info = ArchInfo(
            n_conv3d=self.n_conv3d,
            use_conv2d=self.use_conv2d,
            spatial_after_3d=spatial,
            bands_after_3d=bands,
            reshaped_channels=self.last_conv3d_channels * bands,
            flatten_dim=0,   # 下面算
        )

        # ---------- 2-D 卷积：在空间维进一步抽象，并大幅压缩参数量 ----------
        if self.use_conv2d:
            self.conv2d_1 = nn.Conv2d(self.info.reshaped_channels, 64, kernel_size=3)
            flat = (spatial - 2) ** 2 * 64
        else:
            self.conv2d_1 = None
            flat = spatial * spatial * self.info.reshaped_channels
        self.info.flatten_dim = flat

        # ---------- 3 层全连接 ----------
        self.fc1 = nn.Linear(flat, 256)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(128, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """论文：'所有权重均随机初始化'（未指定具体方案，此处用 He 初始化）。"""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 1, B, S, S) -> logits (N, num_classes)

        Conv3d 的输入布局是 (N, C, D, H, W)，这里 D = 光谱维（PCA 后的 B），
        (H, W) = 空间窗口。
        """
        for conv in self.conv3d_layers:
            x = torch.relu(conv(x))                    # (N, C', B', S', S')

        n = x.size(0)
        if self.use_conv2d:
            # 与 Keras channels_last 的 Reshape 对齐：合并顺序为 (光谱维, 通道)
            x = x.permute(0, 2, 1, 3, 4)               # (N, B', C', S', S')
            x = x.reshape(n, self.info.reshaped_channels,
                          self.info.spatial_after_3d, self.info.spatial_after_3d)
            x = torch.relu(self.conv2d_1(x))           # (N, 64, S'-2, S'-2)

        x = x.reshape(n, -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)                             # 不做 softmax，交给 CrossEntropyLoss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    print('=' * 68)
    print('自检 1：复现论文表 I（IP 数据集配置）')
    print('=' * 68)
    expected_ip = 5_122_176
    net = HybridSN(num_classes=16, pca_components=30, patch_size=25)
    total = net.count_parameters()
    print(f'  可训练参数 : {total:,}')
    print(f'  论文报告   : {expected_ip:,}')
    assert total == expected_ip, f'参数量不符！{total} != {expected_ip}'
    print('  [OK] 与论文表 I 完全一致')

    print('\n  逐层参数量：')
    def show(name, mod):
        print(f'    {name:12s} {sum(p.numel() for p in mod.parameters()):>12,}')
    for i, c in enumerate(net.conv3d_layers, 1):
        show(f'conv3d_{i}', c)
    show('conv2d_1', net.conv2d_1)
    show('fc1', net.fc1)
    show('fc2', net.fc2)
    show('fc3', net.fc3)

    print()
    print('=' * 68)
    print('自检 2：你实际要用的 Pavia University 配置')
    print('=' * 68)
    for variant in ('full', 'two3d', 'no2d'):
        m = HybridSN(num_classes=9, pca_components=15, patch_size=25, variant=variant)
        print(f'  variant={variant:6s} 参数量 {m.count_parameters():>12,}   '
              f'flatten={m.info.flatten_dim:>8,}   reshape通道={m.info.reshaped_channels}')

    print()
    print('  ↑ 注意 no2d 的参数量暴涨：这正是论文里"2-D 卷积用来降低模型复杂度"的直接证据，')
    print('    可以直接写进你的消融实验分析。')

    print()
    print('=' * 68)
    print('自检 3：前向传播形状')
    print('=' * 68)
    for variant in ('full', 'two3d', 'no2d'):
        m = HybridSN(num_classes=9, pca_components=15, patch_size=25, variant=variant)
        x = torch.randn(4, 1, 15, 25, 25)
        y = m(x)
        assert y.shape == (4, 9), y.shape
        print(f'  variant={variant:6s}  {tuple(x.shape)} -> {tuple(y.shape)}  [OK]')
