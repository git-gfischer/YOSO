"""
export_yoso_detectron2_onnx.py – Export the *real* Detectron2 YOSO model to ONNX.

The legacy convertion/model_arch.py is an approximate reconstruction and produces
incorrect panoptic outputs.  This script loads the official YOSO implementation
from vendor/yoso/projects/YOSO and exports logits + mask logits for TRT/ORT inference.

Usage
-----
    python convertion/export_yoso_detectron2_onnx.py \
        --checkpoint models/yoso_res50_coco.pth \
        --config     vendor/yoso/projects/YOSO/configs/coco/panoptic-segmentation/YOSO-R50.yaml \
        --output     models/yoso_res50.onnx \
        --height 480 --width 640
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
YOSO_VENDOR_ROOT = os.path.join(REPO_ROOT, "vendor", "yoso")
YOSO_ROOT = os.path.join(YOSO_VENDOR_ROOT, "projects", "YOSO")
for p in (YOSO_VENDOR_ROOT, YOSO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib
import types

# Register yoso package without executing yoso/__init__.py (avoids cv2 dataset mappers).
if "yoso" not in sys.modules:
    _pkg = types.ModuleType("yoso")
    _pkg.__path__ = [os.path.join(YOSO_ROOT, "yoso")]
    sys.modules["yoso"] = _pkg

_yoso_config = importlib.import_module("yoso.config")
add_yoso_config = _yoso_config.add_yoso_config
importlib.import_module("yoso.segmentator")  # register META_ARCH

from detectron2.config import get_cfg
from detectron2.modeling import build_model


class DeformLayerGridSample(nn.Module):
    """Modulated DCN reimplemented with grid_sample for ONNX/TRT export."""

    def __init__(self, src: nn.Module):
        super().__init__()
        self.dcn_offset = src.dcn_offset
        self.dcn_w = src.dcn.weight
        self.dcn_bn = src.dcn_bn
        self.up_sample = src.up_sample
        self.up_bn = src.up_bn
        self.relu = src.relu
        self.C_out = self.dcn_w.shape[0]

    def forward(self, x):
        B, _, H, W = x.shape
        off_mask = self.dcn_offset(x)
        offset_x, offset_y, mask = torch.chunk(off_mask, 3, dim=1)
        offset = torch.cat((offset_x, offset_y), dim=1)
        mask = mask.sigmoid()

        ys = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        ky = torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1], device=x.device, dtype=x.dtype) / H
        kx = torch.tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1], device=x.device, dtype=x.dtype) / W

        out = torch.zeros(B, self.C_out, H, W, device=x.device, dtype=x.dtype)
        w = self.dcn_w
        for k in range(9):
            dy = offset[:, 2 * k] / H
            dx = offset[:, 2 * k + 1] / W
            sy = (grid_y.unsqueeze(0) + ky[k] + dy).unsqueeze(-1)
            sx = (grid_x.unsqueeze(0) + kx[k] + dx).unsqueeze(-1)
            grid_k = torch.cat([sx, sy], dim=-1)
            samp_k = F.grid_sample(x, grid_k, align_corners=True, padding_mode="zeros", mode="bilinear")
            samp_k = samp_k * mask[:, k : k + 1]
            ky_idx, kx_idx = k // 3, k % 3
            w_k = w[:, :, ky_idx, kx_idx].unsqueeze(-1).unsqueeze(-1)
            out = out + F.conv2d(samp_k, w_k)

        out = self.relu(self.dcn_bn(out))
        out = self.relu(self.up_bn(self.up_sample(out)))
        return out


def replace_dcn_with_gridsampl(yoso_model: nn.Module) -> nn.Module:
    deconv = yoso_model.yoso_neck.deconv
    for name in ("deform_conv1", "deform_conv2", "deform_conv3"):
        setattr(deconv, name, DeformLayerGridSample(getattr(deconv, name)))
    return yoso_model


class YOSOExportWrapper(nn.Module):
    """Inference-only forward: normalized NCHW tensor -> (logits, masks)."""

    def __init__(self, yoso_model: nn.Module):
        super().__init__()
        self.model = yoso_model

    def forward(self, image: torch.Tensor):
        # image: [B, 3, H, W] already Detectron2-normalized RGB
        backbone_feats = self.model.backbone(image)
        feats = [backbone_feats[f] for f in self.model.in_features]
        neck_feats = self.model.yoso_neck(feats)

        # Mirror YOSOHead eval path (targets=None, final stage only).
        head = self.model.yoso_head
        object_kernels = None
        mask_preds = None
        cls_scores = None
        for stage in range(head.num_stages + 1):
            if stage == 0:
                mask_preds = head.kernels(neck_feats)
                proposal_kernels = head.kernels.weight.clone()
                object_kernels = proposal_kernels[None].expand(
                    neck_feats.shape[0], *proposal_kernels.size()
                )
            elif stage == head.num_stages:
                mask_head = head.mask_heads[stage - 1]
                cls_scores, mask_preds, proposal_kernels = mask_head(
                    neck_feats, object_kernels, mask_preds, True
                )
            else:
                mask_head = head.mask_heads[stage - 1]
                cls_scores, mask_preds, proposal_kernels = mask_head(
                    neck_feats, object_kernels, mask_preds, False
                )
                object_kernels = proposal_kernels

            if cls_scores is not None:
                pass  # temperature applied in C++ postprocess (YOSO_TEMPERATURE)

        return cls_scores, mask_preds


def parse_args():
    p = argparse.ArgumentParser(description="Export real Detectron2 YOSO to ONNX")
    p.add_argument("--checkpoint", default="models/yoso_res50_coco.pth")
    p.add_argument(
        "--config",
        default="vendor/yoso/projects/YOSO/configs/coco/panoptic-segmentation/YOSO-R50.yaml",
    )
    p.add_argument("--output", default="models/yoso_res50.onnx")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--device", default="cpu",
                   help="cuda if available; cpu works with --dcn-mode gridsampl")
    p.add_argument("--dcn-mode", choices=["native", "gridsampl"], default="gridsampl",
                   help="gridsampl replaces DCN with grid_sample (recommended for TRT)")
    p.add_argument("--simplify", action="store_true")
    return p.parse_args()


def build_cfg(config_path: str, checkpoint: str, device: str):
    cfg = get_cfg()
    add_yoso_config(cfg)
    cfg.merge_from_file(os.path.join(REPO_ROOT, config_path))
    cfg.MODEL.WEIGHTS = os.path.join(REPO_ROOT, checkpoint)
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    return cfg


def export(args):
    if args.dcn_mode == "gridsampl":
        if args.device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA required for --dcn-mode native (DCN has no CPU kernel)")

    H, W = args.height, args.width
    assert H % 32 == 0 and W % 32 == 0

    print(f"[1/5] Building Detectron2 YOSO (dcn-mode={args.dcn_mode}, device={device}) …")
    cfg = build_cfg(args.config, args.checkpoint, str(device))
    base = build_model(cfg)
    ckpt = torch.load(os.path.join(REPO_ROOT, args.checkpoint), map_location="cpu")
    sd = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = base.load_state_dict(sd, strict=False)
    if missing:
        print(f"    [WARN] missing keys: {len(missing)}")
    if unexpected:
        print(f"    [WARN] unexpected keys: {len(unexpected)}")
    base.eval()

    if args.dcn_mode == "gridsampl":
        print("[2/5] Replacing DCN with grid_sample ops …")
        replace_dcn_with_gridsampl(base)

    model = YOSOExportWrapper(base).to(device).eval()

    # Detectron2 pixel normalization (RGB, 0..255 scale).
    mean = torch.tensor(cfg.MODEL.PIXEL_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg.MODEL.PIXEL_STD, device=device).view(1, 3, 1, 1)
    dummy_rgb = torch.rand(1, 3, H, W, device=device) * 255.0
    dummy = (dummy_rgb - mean) / std

    print(f"[3/5] Dry-run forward {tuple(dummy.shape)} …")
    with torch.no_grad():
        logits, masks = model(dummy)
    print(f"    logits={tuple(logits.shape)}  masks={tuple(masks.shape)}")

    print(f"[4/5] torch.onnx.export (opset={args.opset}) …")
    os.makedirs(os.path.dirname(os.path.join(REPO_ROOT, args.output)), exist_ok=True)
    out_path = os.path.join(REPO_ROOT, args.output)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            dummy,
            out_path,
            dynamo=False,
            opset_version=args.opset,
            input_names=["image"],
            output_names=["logits", "masks"],
            export_params=True,
            do_constant_folding=True,
            verbose=False,
        )

    print("[5/5] Validate ONNX …")
    m = onnx.load(out_path)
    onnx.checker.check_model(m)
    if args.simplify:
        try:
            from onnxsim import simplify as onnxsim
            simp, ok = onnxsim(m)
            if ok:
                onnx.save(simp, out_path)
                print("    onnxsim OK")
        except ImportError:
            print("    onnxsim not installed")

    print(f"\n[✓] Exported real YOSO ONNX → {out_path}")
    for n in m.graph.output:
        dims = [d.dim_value or d.dim_param for d in n.type.tensor_type.shape.dim]
        print(f"  {n.name:10s} : {dims}")


if __name__ == "__main__":
    export(parse_args())
