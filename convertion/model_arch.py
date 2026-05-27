"""
YOSO (You Only Segment Once) - ResNet-50 / COCO
Architecture reconstructed from checkpoint weight shapes.

Architecture overview
─────────────────────
Backbone  : ResNet-50  (Detectron2 naming: stem + res2..res5)
Neck      : YOSONeck   (Deformable-Conv FPN + location projection)
Head      : YOSOHead   (100 learned kernels, 2 decoder stages with
                        hash-based attention + self-attention + FFN)

Inputs  : [B, 3, H, W]  (H,W must be multiples of 32)
Outputs : (logits [B, 100, 134], masks [B, 100, H/4, W/4])
          134 = 80 thing + 53 stuff + 1 background  (COCO panoptic)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


# ────────────────────────────────────────────────────────────────
# 1.  BACKBONE  (ResNet-50, Detectron2 weight naming)
# ────────────────────────────────────────────────────────────────

class _BN(nn.Module):
    """Tiny wrapper: BatchNorm2d whose parameters are stored as
    weight / bias / running_mean / running_var (no 'running_num_batches')"""
    def __init__(self, c):
        super().__init__()
        self.weight       = nn.Parameter(torch.ones(c))
        self.bias         = nn.Parameter(torch.zeros(c))
        self.register_buffer('running_mean', torch.zeros(c))
        self.register_buffer('running_var',  torch.ones(c))

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var,
                            self.weight, self.bias, False, 0.1, 1e-5)


class _ConvBN(nn.Module):
    def __init__(self, in_c, out_c, k=1, s=1, p=0):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_c, in_c, k, k))
        self.norm   = _BN(out_c)

    def forward(self, x, stride=1, padding=0):
        # stride/padding taken from the constructor; kept as args for clarity
        return self.norm(F.conv2d(x, self.weight,
                                  stride=self._stride, padding=self._padding))


class Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = _ConvBN(3, 64, 7, 2, 3)   # stored as conv1/norm
        self.conv1._stride  = 2
        self.conv1._padding = 3

    def forward(self, x):
        x = F.relu(self.conv1.norm(F.conv2d(x, self.conv1.weight, stride=2, padding=3)))
        return F.max_pool2d(x, 3, stride=2, padding=1)


class BottleneckBlock(nn.Module):
    """Standard ResNet bottleneck.  Uses Detectron2 key naming."""
    def __init__(self, in_c, mid_c, out_c, stride=1, has_shortcut=False):
        super().__init__()
        self.stride      = stride
        self.has_shortcut = has_shortcut

        # 1×1 → 3×3 → 1×1
        self.conv1 = nn.Parameter(torch.zeros(mid_c,  in_c,  1, 1))
        self.norm1 = _BN(mid_c)
        self.conv2 = nn.Parameter(torch.zeros(mid_c,  mid_c, 3, 3))
        self.norm2 = _BN(mid_c)
        self.conv3 = nn.Parameter(torch.zeros(out_c,  mid_c, 1, 1))
        self.norm3 = _BN(out_c)

        if has_shortcut:
            self.shortcut      = nn.Parameter(torch.zeros(out_c, in_c, 1, 1))
            self.shortcut_norm = _BN(out_c)

    # Checkpoint key mapping: conv1/norm1 → conv1 / conv1.norm, etc.
    def _load(self, sd, prefix):
        """Load weights from state-dict using Detectron2 naming."""
        def _cp(dst: nn.Parameter, key):
            if key in sd:
                dst.data.copy_(sd[key])

        _cp(self.conv1, f'{prefix}.conv1.weight')
        self.norm1.weight.data.copy_(sd[f'{prefix}.conv1.norm.weight'])
        self.norm1.bias.data.copy_(sd[f'{prefix}.conv1.norm.bias'])
        self.norm1.running_mean.copy_(sd[f'{prefix}.conv1.norm.running_mean'])
        self.norm1.running_var.copy_(sd[f'{prefix}.conv1.norm.running_var'])

        _cp(self.conv2, f'{prefix}.conv2.weight')
        self.norm2.weight.data.copy_(sd[f'{prefix}.conv2.norm.weight'])
        self.norm2.bias.data.copy_(sd[f'{prefix}.conv2.norm.bias'])
        self.norm2.running_mean.copy_(sd[f'{prefix}.conv2.norm.running_mean'])
        self.norm2.running_var.copy_(sd[f'{prefix}.conv2.norm.running_var'])

        _cp(self.conv3, f'{prefix}.conv3.weight')
        self.norm3.weight.data.copy_(sd[f'{prefix}.conv3.norm.weight'])
        self.norm3.bias.data.copy_(sd[f'{prefix}.conv3.norm.bias'])
        self.norm3.running_mean.copy_(sd[f'{prefix}.conv3.norm.running_mean'])
        self.norm3.running_var.copy_(sd[f'{prefix}.conv3.norm.running_var'])

        if self.has_shortcut:
            _cp(self.shortcut, f'{prefix}.shortcut.weight')
            self.shortcut_norm.weight.data.copy_(sd[f'{prefix}.shortcut.norm.weight'])
            self.shortcut_norm.bias.data.copy_(sd[f'{prefix}.shortcut.norm.bias'])
            self.shortcut_norm.running_mean.copy_(sd[f'{prefix}.shortcut.norm.running_mean'])
            self.shortcut_norm.running_var.copy_(sd[f'{prefix}.shortcut.norm.running_var'])

    def forward(self, x):
        identity = x

        out = F.relu(self.norm1(F.conv2d(x,   self.conv1, stride=1,          padding=0)))
        out = F.relu(self.norm2(F.conv2d(out,  self.conv2, stride=self.stride, padding=1)))
        out =        self.norm3(F.conv2d(out,  self.conv3, stride=1,          padding=0))

        if self.has_shortcut:
            identity = self.shortcut_norm(
                F.conv2d(x, self.shortcut, stride=self.stride, padding=0))

        return F.relu(out + identity)


class ResNet50(nn.Module):
    # (in_c, mid_c, out_c, num_blocks, stride)
    STAGES = [
        (64,   64,  256, 3, 1),   # res2
        (256, 128,  512, 4, 2),   # res3
        (512, 256, 1024, 6, 2),   # res4
        (1024,512, 2048, 3, 2),   # res5
    ]

    def __init__(self):
        super().__init__()
        self.stem = Stem()
        names = ['res2', 'res3', 'res4', 'res5']
        for name, (in_c, mid_c, out_c, n, s) in zip(names, self.STAGES):
            blocks = []
            for i in range(n):
                blocks.append(BottleneckBlock(
                    in_c  if i == 0 else out_c,
                    mid_c, out_c,
                    stride      = s if i == 0 else 1,
                    has_shortcut= (i == 0)
                ))
            setattr(self, name, nn.ModuleList(blocks))

    def load_from_state_dict(self, sd):
        # Stem
        self.stem.conv1.weight.data.copy_(sd['backbone.stem.conv1.weight'])
        self.stem.conv1.norm.weight.data.copy_(sd['backbone.stem.conv1.norm.weight'])
        self.stem.conv1.norm.bias.data.copy_(sd['backbone.stem.conv1.norm.bias'])
        self.stem.conv1.norm.running_mean.copy_(sd['backbone.stem.conv1.norm.running_mean'])
        self.stem.conv1.norm.running_var.copy_(sd['backbone.stem.conv1.norm.running_var'])

        for name in ['res2', 'res3', 'res4', 'res5']:
            stage = getattr(self, name)
            for i, blk in enumerate(stage):
                blk._load(sd, f'backbone.{name}.{i}')

    def forward(self, x):
        x    = self.stem(x)          # /4
        res2 = self._run_stage(self.res2, x)   # /4
        res3 = self._run_stage(self.res3, res2)  # /8
        res4 = self._run_stage(self.res4, res3)  # /16
        res5 = self._run_stage(self.res5, res4)  # /32
        return res2, res3, res4, res5

    @staticmethod
    def _run_stage(blocks, x):
        for blk in blocks:
            x = blk(x)
        return x


# ────────────────────────────────────────────────────────────────
# 2.  NECK  (Deformable-Conv FPN + location projection)
# ────────────────────────────────────────────────────────────────

class DeformConvBlock(nn.Module):
    """One DCN stage: offset-pred → deform_conv2d → BN → ConvTranspose upsample → BN"""
    def __init__(self, in_c, out_c):
        super().__init__()
        # offset + mask prediction (modulated DCN v2: 27 = 9*(2+1))
        self.dcn_offset = nn.Conv2d(in_c, 27, 3, padding=1)
        self.dcn        = nn.Parameter(torch.zeros(out_c, in_c, 3, 3))
        self.dcn_bn     = nn.BatchNorm2d(out_c)
        # 2× bilinear-like upsample via ConvTranspose
        self.up_sample  = nn.ConvTranspose2d(out_c, out_c, 4, stride=2, padding=1)
        self.up_bn      = nn.BatchNorm2d(out_c)

    def forward(self, x):
        off_mask = self.dcn_offset(x)          # [B, 27, H, W]
        offset   = off_mask[:, :18]            # [B, 18, H, W]
        mask     = off_mask[:, 18:].sigmoid()  # [B,  9, H, W]

        out = F.relu(self.dcn_bn(
            deform_conv2d(x, offset, self.dcn,
                          mask=mask, padding=1)
        ))
        out = F.relu(self.up_bn(self.up_sample(out)))
        return out


class YOSONeck(nn.Module):
    def __init__(self):
        super().__init__()
        d = nn.Module()
        d.lateral_conv0 = nn.Conv2d(2048, 1024, 1)
        d.deform_conv1  = DeformConvBlock(1024, 512)
        d.lateral_conv1 = nn.Conv2d(1024, 512, 1)
        d.deform_conv2  = DeformConvBlock(512, 256)
        d.lateral_conv2 = nn.Conv2d(512, 256, 1)
        d.deform_conv3  = DeformConvBlock(256, 128)
        d.lateral_conv3 = nn.Conv2d(256, 128, 1)
        d.output_conv   = nn.Conv2d(128, 128, 3, padding=1)
        # attention-projection convs (sum multi-scale → 128 ch)
        d.conv_a5 = nn.Conv2d(1024, 128, 1, bias=False)
        d.conv_a4 = nn.Conv2d(512,  128, 1, bias=False)
        d.conv_a3 = nn.Conv2d(256,  128, 1, bias=False)
        d.conv_a2 = nn.Conv2d(128,  128, 1, bias=False)
        d.bias    = nn.Parameter(torch.zeros(1, 128, 1, 1))
        self.deconv = d

        # location projection: 128 features + 2 coordinate channels → 256
        self.loc_conv = nn.Conv2d(130, 256, 1)

    def _load_dcn_block(self, blk: DeformConvBlock, sd, prefix):
        blk.dcn_offset.weight.data.copy_(sd[f'{prefix}.dcn_offset.weight'])
        blk.dcn_offset.bias.data.copy_(sd[f'{prefix}.dcn_offset.bias'])
        blk.dcn.data.copy_(sd[f'{prefix}.dcn.weight'])
        blk.dcn_bn.weight.data.copy_(sd[f'{prefix}.dcn_bn.weight'])
        blk.dcn_bn.bias.data.copy_(sd[f'{prefix}.dcn_bn.bias'])
        blk.dcn_bn.running_mean.copy_(sd[f'{prefix}.dcn_bn.running_mean'])
        blk.dcn_bn.running_var.copy_(sd[f'{prefix}.dcn_bn.running_var'])
        blk.up_sample.weight.data.copy_(sd[f'{prefix}.up_sample.weight'])
        blk.up_bn.weight.data.copy_(sd[f'{prefix}.up_bn.weight'])
        blk.up_bn.bias.data.copy_(sd[f'{prefix}.up_bn.bias'])
        blk.up_bn.running_mean.copy_(sd[f'{prefix}.up_bn.running_mean'])
        blk.up_bn.running_var.copy_(sd[f'{prefix}.up_bn.running_var'])

    def load_from_state_dict(self, sd):
        d = self.deconv
        p = 'yoso_neck.deconv'
        for attr, key in [
            ('lateral_conv0', f'{p}.lateral_conv0'),
            ('lateral_conv1', f'{p}.lateral_conv1'),
            ('lateral_conv2', f'{p}.lateral_conv2'),
            ('lateral_conv3', f'{p}.lateral_conv3'),
            ('output_conv',   f'{p}.output_conv'),
        ]:
            conv = getattr(d, attr)
            conv.weight.data.copy_(sd[f'{key}.weight'])
            if conv.bias is not None:
                conv.bias.data.copy_(sd[f'{key}.bias'])

        for attr, key in [
            ('conv_a5', f'{p}.conv_a5'),
            ('conv_a4', f'{p}.conv_a4'),
            ('conv_a3', f'{p}.conv_a3'),
            ('conv_a2', f'{p}.conv_a2'),
        ]:
            getattr(d, attr).weight.data.copy_(sd[f'{key}.weight'])

        d.bias.data.copy_(sd[f'{p}.bias'])

        self._load_dcn_block(d.deform_conv1, sd, f'{p}.deform_conv1')
        self._load_dcn_block(d.deform_conv2, sd, f'{p}.deform_conv2')
        self._load_dcn_block(d.deform_conv3, sd, f'{p}.deform_conv3')

        self.loc_conv.weight.data.copy_(sd['yoso_neck.loc_conv.weight'])
        self.loc_conv.bias.data.copy_(sd['yoso_neck.loc_conv.bias'])

    @staticmethod
    def _coord_grid(h, w, device):
        """Create normalized (x, y) grid in [-1, 1], shape [1, 2, H, W]."""
        ys = torch.linspace(-1, 1, h, device=device)
        xs = torch.linspace(-1, 1, w, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        return torch.stack([grid_x, grid_y], 0).unsqueeze(0)  # [1, 2, H, W]

    def forward(self, res2, res3, res4, res5):
        d = self.deconv

        # FPN top-down path – keep intermediate features for multi-scale aggregation
        feat5 = d.lateral_conv0(res5)                      # [B,1024,H/32,W/32]
        p4    = d.deform_conv1(feat5) + d.lateral_conv1(res4)   # [B,512,H/16,W/16]
        p3    = d.deform_conv2(p4)    + d.lateral_conv2(res3)   # [B,256,H/8, W/8 ]
        p2    = d.deform_conv3(p3)    + d.lateral_conv3(res2)   # [B,128,H/4, W/4 ]
        feat  = F.relu(d.output_conv(p2))                  # [B,128,H/4, W/4 ]

        # Multi-scale aggregation: project each FPN level to 128-ch, upsample, sum
        H, W = feat.shape[2:]
        a5 = F.interpolate(d.conv_a5(feat5), (H, W), mode='bilinear', align_corners=False)
        a4 = F.interpolate(d.conv_a4(p4),    (H, W), mode='bilinear', align_corners=False)
        a3 = F.interpolate(d.conv_a3(p3),    (H, W), mode='bilinear', align_corners=False)
        a2 = d.conv_a2(feat)
        feat = feat + a5 + a4 + a3 + a2 + d.bias

        # Append coordinate channels then project to 256-d location features
        coord = self._coord_grid(H, W, feat.device).expand(feat.size(0), -1, -1, -1)
        loc_feat = F.relu(self.loc_conv(torch.cat([feat, coord], dim=1)))  # [B, 256, H/4, W/4]
        return feat, loc_feat   # (128-ch raw, 256-ch location-aware)


# ────────────────────────────────────────────────────────────────
# 3.  HEAD  (YOSO decoder)
# ────────────────────────────────────────────────────────────────

class HashAttention(nn.Module):
    """
    Learned hash-based attention (YOSO fast attention).
    Queries and keys are projected to a low-dim hash space;
    attention weights are computed as a scaled dot-product
    in that space.  Compatible with ONNX export.
    """
    def __init__(self, embed_dim, hash_dim):
        super().__init__()
        self.weight_linear = nn.Linear(embed_dim, hash_dim)
        self.norm           = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value):
        """
        query : [B, Nq, C]
        key   : [B, Nk, C]
        value : [B, Nk, C]
        """
        q_hash = self.weight_linear(query)   # [B, Nq, H]
        k_hash = self.weight_linear(key)     # [B, Nk, H]
        scale  = q_hash.size(-1) ** -0.5
        attn   = torch.softmax(q_hash @ k_hash.transpose(-1, -2) * scale, dim=-1)  # [B, Nq, Nk]
        out    = attn @ value                # [B, Nq, C]
        return self.norm(out + query)


class FFN(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=2048):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, hidden_dim)),
            nn.Linear(hidden_dim, embed_dim),
        ])
        self.norm   = nn.LayerNorm(embed_dim)

    def forward(self, x):
        out = F.relu(self.layers[0][0](x))
        out = self.layers[1](out)
        return self.norm(out + x)


class _ManualMHA(nn.Module):
    """
    Drop-in for nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    implemented via standard matmul + softmax → ONNX-exportable.

    Stores weights under the same names as nn.MultiheadAttention so that
    the state-dict loader works without changes.
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        # Match nn.MultiheadAttention weight names exactly
        self.in_proj_weight = nn.Parameter(torch.zeros(3 * embed_dim, embed_dim))
        self.in_proj_bias   = nn.Parameter(torch.zeros(3 * embed_dim))
        self.out_proj       = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=False, attn_mask=None):
        B, N, C = query.shape
        H, D    = self.num_heads, self.head_dim

        # QKV projection
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)   # [B,N,3C]
        q, k, v = qkv.chunk(3, dim=-1)                                   # each [B,N,C]

        # Split heads
        def split_heads(x):
            return x.reshape(B, N, H, D).transpose(1, 2)  # [B,H,N,D]

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Scaled dot-product attention
        scale = D ** -0.5
        attn  = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)  # [B,H,N,N]
        out   = (attn @ v).transpose(1, 2).reshape(B, N, C)               # [B,N,C]

        return self.out_proj(out), None   # match nn.MHA return signature


class MaskDecoderStage(nn.Module):
    """One YOSO decoder stage."""
    def __init__(self, embed_dim=256, hash_dim=103, num_classes=134):
        super().__init__()
        self.f_atten      = HashAttention(embed_dim, hash_dim)
        self.f_atten_norm = nn.LayerNorm(embed_dim)
        self.k_atten      = HashAttention(embed_dim, hash_dim)
        self.k_atten_norm = nn.LayerNorm(embed_dim)
        self.s_atten      = _ManualMHA(embed_dim, 8)   # ONNX-exportable
        self.s_atten_norm = nn.LayerNorm(embed_dim)
        self.ffn          = FFN(embed_dim)
        self.ffn_norm     = nn.LayerNorm(embed_dim)   # already inside FFN; kept for weight load

        # Classification branch
        self.cls_fcs  = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.fc_cls   = nn.Linear(embed_dim, num_classes)

        # Mask branch
        self.mask_fcs = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.fc_mask  = nn.Linear(embed_dim, embed_dim)

    def _load(self, sd, prefix):
        def copy(dst, key):
            if key in sd:
                dst.data.copy_(sd[key])

        # HashAttention – f_atten
        copy(self.f_atten.weight_linear.weight, f'{prefix}.f_atten.weight_linear.weight')
        copy(self.f_atten.weight_linear.bias,   f'{prefix}.f_atten.weight_linear.bias')
        copy(self.f_atten.norm.weight,           f'{prefix}.f_atten.norm.weight')
        copy(self.f_atten.norm.bias,             f'{prefix}.f_atten.norm.bias')
        copy(self.f_atten_norm.weight,           f'{prefix}.f_atten_norm.weight')
        copy(self.f_atten_norm.bias,             f'{prefix}.f_atten_norm.bias')

        # HashAttention – k_atten
        copy(self.k_atten.weight_linear.weight, f'{prefix}.k_atten.weight_linear.weight')
        copy(self.k_atten.weight_linear.bias,   f'{prefix}.k_atten.weight_linear.bias')
        copy(self.k_atten.norm.weight,           f'{prefix}.k_atten.norm.weight')
        copy(self.k_atten.norm.bias,             f'{prefix}.k_atten.norm.bias')
        copy(self.k_atten_norm.weight,           f'{prefix}.k_atten_norm.weight')
        copy(self.k_atten_norm.bias,             f'{prefix}.k_atten_norm.bias')

        # Self-attention
        copy(self.s_atten.in_proj_weight,        f'{prefix}.s_atten.in_proj_weight')
        copy(self.s_atten.in_proj_bias,          f'{prefix}.s_atten.in_proj_bias')
        copy(self.s_atten.out_proj.weight,       f'{prefix}.s_atten.out_proj.weight')
        copy(self.s_atten.out_proj.bias,         f'{prefix}.s_atten.out_proj.bias')
        copy(self.s_atten_norm.weight,           f'{prefix}.s_atten_norm.weight')
        copy(self.s_atten_norm.bias,             f'{prefix}.s_atten_norm.bias')

        # FFN  (layers.0.0 = first Linear; layers.1 = second Linear)
        copy(self.ffn.layers[0][0].weight,  f'{prefix}.ffn.layers.0.0.weight')
        copy(self.ffn.layers[0][0].bias,    f'{prefix}.ffn.layers.0.0.bias')
        copy(self.ffn.layers[1].weight,     f'{prefix}.ffn.layers.1.weight')
        copy(self.ffn.layers[1].bias,       f'{prefix}.ffn.layers.1.bias')
        copy(self.ffn.norm.weight,          f'{prefix}.ffn_norm.weight')
        copy(self.ffn.norm.bias,            f'{prefix}.ffn_norm.bias')

        # Classification
        copy(self.cls_fcs[0].weight,  f'{prefix}.cls_fcs.0.weight')
        copy(self.cls_fcs[1].weight,  f'{prefix}.cls_fcs.1.weight')
        copy(self.cls_fcs[1].bias,    f'{prefix}.cls_fcs.1.bias')
        copy(self.fc_cls.weight,      f'{prefix}.fc_cls.weight')
        copy(self.fc_cls.bias,        f'{prefix}.fc_cls.bias')

        # Mask
        copy(self.mask_fcs[0].weight, f'{prefix}.mask_fcs.0.weight')
        copy(self.mask_fcs[1].weight, f'{prefix}.mask_fcs.1.weight')
        copy(self.mask_fcs[1].bias,   f'{prefix}.mask_fcs.1.bias')
        copy(self.fc_mask.weight,     f'{prefix}.fc_mask.weight')
        copy(self.fc_mask.bias,       f'{prefix}.fc_mask.bias')

    def forward(self, kernels, loc_feat):
        """
        kernels  : [B, Nk=100, 256]
        loc_feat : [B, 256, H, W]  location-aware feature map
        Returns  : logits [B, 100, 134], mask_weights [B, 100, 256]
        """
        B, C, H, W = loc_feat.shape
        # Flatten spatial → tokens:  [B, H*W, C]
        pixels = loc_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Feature attention: kernels attend to pixels
        k2 = self.f_atten(kernels, pixels, pixels)
        k2 = self.f_atten_norm(k2 + kernels)

        # Kernel attention: pixels attend to kernels
        p2 = self.k_atten(pixels, kernels, kernels)
        p2 = self.k_atten_norm(p2 + pixels)

        # Self-attention among kernels
        k3, _ = self.s_atten(k2, k2, k2)
        k3     = self.s_atten_norm(k3 + k2)

        # FFN
        k4 = self.ffn(k3)

        # Classification head
        cls_feat = self.cls_fcs(k4)
        logits   = self.fc_cls(cls_feat)                 # [B, 100, 134]

        # Mask-kernel head
        mask_feat    = self.mask_fcs(k4)
        mask_weights = self.fc_mask(mask_feat)           # [B, 100, 256]

        return logits, mask_weights, k4   # k4 passed to next stage


class YOSOHead(nn.Module):
    def __init__(self, num_kernels=100, embed_dim=256,
                 hash_dim=103, num_classes=134, num_stages=2):
        super().__init__()
        self.kernels    = nn.Conv2d(embed_dim, num_kernels, 1)  # [100, 256, 1, 1]
        self.mask_heads = nn.ModuleList([
            MaskDecoderStage(embed_dim, hash_dim, num_classes)
            for _ in range(num_stages)
        ])

    def load_from_state_dict(self, sd):
        self.kernels.weight.data.copy_(sd['yoso_head.kernels.weight'])
        self.kernels.bias.data.copy_(sd['yoso_head.kernels.bias'])
        for i, stage in enumerate(self.mask_heads):
            stage._load(sd, f'yoso_head.mask_heads.{i}')

    def forward(self, loc_feat):
        """
        loc_feat : [B, 256, H, W]
        Returns  : list of (logits, mask_weights) per stage,
                   plus final instance_masks [B, 100, H, W]
        """
        B, C, H, W = loc_feat.shape

        # The stored Conv2d weight [100, 256, 1, 1] is the learnable kernel bank.
        # Expand to [B, 100, 256] as initial object-query features.
        kernels = self.kernels.weight.view(self.kernels.weight.size(0), -1)  # [100, 256]
        kernels = kernels.unsqueeze(0).expand(B, -1, -1)                     # [B, 100, 256]

        outputs = []
        k_state = kernels
        for stage in self.mask_heads:
            logits, mask_w, k_state = stage(k_state, loc_feat)
            outputs.append((logits, mask_w))

        # Generate final masks: dot each kernel with spatial features
        # mask_w : [B, 100, 256];  loc_feat : [B, 256, H, W]
        final_mask_w  = outputs[-1][1]                        # [B, 100, 256]
        spatial       = loc_feat.view(B, C, H * W)           # [B, 256, H*W]
        masks         = torch.bmm(final_mask_w, spatial)     # [B, 100, H*W]
        masks         = masks.view(B, 100, H, W)

        return outputs[-1][0], masks   # logits [B,100,134],  masks [B,100,H,W]


# ────────────────────────────────────────────────────────────────
# 4.  FULL MODEL
# ────────────────────────────────────────────────────────────────

class YOSOModel(nn.Module):
    """
    Full YOSO instance/panoptic segmentation model.

    Inputs
    ------
    x : torch.Tensor  [B, 3, H, W]  – normalised image (ImageNet mean/std)

    Outputs
    -------
    logits : [B, 100, 134]  – per-kernel class logits
    masks  : [B, 100, H/4, W/4]  – per-kernel mask logits (pre-sigmoid)
    """

    def __init__(self):
        super().__init__()
        self.backbone = ResNet50()
        self.neck     = YOSONeck()
        self.head     = YOSOHead()

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, map_location='cpu') -> 'YOSOModel':
        model = cls()
        ckpt  = torch.load(ckpt_path, map_location=map_location)
        sd    = ckpt['model'] if 'model' in ckpt else ckpt
        model.backbone.load_from_state_dict(sd)
        model.neck.load_from_state_dict(sd)
        model.head.load_from_state_dict(sd)
        model.eval()
        print(f"[YOSOModel] Loaded from '{ckpt_path}'  "
              f"({sum(p.numel() for p in model.parameters()):,} params)")
        return model

    def forward(self, x):
        res2, res3, res4, res5 = self.backbone(x)
        _feat, loc_feat        = self.neck(res2, res3, res4, res5)
        logits, masks          = self.head(loc_feat)
        return logits, masks


# ────────────────────────────────────────────────────────────────
# 5.  POST-PROCESSING HELPERS
# ────────────────────────────────────────────────────────────────

COCO_THING_CLASSES = 80   # instance classes
NUM_CLASSES        = 134  # 80 thing + 53 stuff + 1 background

def postprocess(logits, masks, score_threshold=0.5, mask_threshold=0.5,
                orig_h=None, orig_w=None):
    """
    logits : [1, 100, 134]
    masks  : [1, 100, H, W]  (pre-sigmoid)
    Returns list of dicts with 'class_id', 'score', 'mask'
    """
    probs     = logits[0].softmax(-1)                     # [100, 134]
    scores, labels = probs[:, :-1].max(-1)                # exclude background
    keep      = scores > score_threshold

    det_scores = scores[keep].detach().cpu()
    det_labels = labels[keep].detach().cpu()
    det_masks  = masks[0][keep].sigmoid()                 # [N, H, W]

    if orig_h and orig_w:
        det_masks = F.interpolate(det_masks.unsqueeze(0),
                                  size=(orig_h, orig_w),
                                  mode='bilinear', align_corners=False)[0]

    results = []
    for s, l, m in zip(det_scores, det_labels, det_masks):
        results.append({
            'class_id': int(l),
            'score'   : float(s),
            'mask'    : (m > mask_threshold).bool().cpu().numpy(),
        })
    return results


# ────────────────────────────────────────────────────────────────
# Quick sanity check
# ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else 'yoso_res50_coco.pth'
    model = YOSOModel.from_checkpoint(ckpt)
    dummy = torch.zeros(1, 3, 512, 512)
    with torch.no_grad():
        logits, masks = model(dummy)
    print(f"logits: {logits.shape}   masks: {masks.shape}")
