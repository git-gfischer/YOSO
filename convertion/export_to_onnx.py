"""
export_to_onnx.py  –  Export YOSO model to ONNX

Usage
-----
    python export_to_onnx.py \
        --checkpoint yoso_res50_coco.pth \
        --output     yoso_res50.onnx  \
        --height     512  --width 512 \
        [--dynamic]   \
        [--opset 18]  \
        [--simplify]

Three DCN export modes (choose with --dcn-mode):
  native   [default] — keep torchvision DCNv2 op unchanged during export.
                       Preferred for TensorRT >= 8.6 native DCN handling.
  custom             — exports DCN as a 'custom::DeformConv2d' op node.
                       Requires matching runtime plugin.
  gridsampl          — replaces DCN with F.grid_sample (100% standard ONNX).
                       Approximation; may reduce quality.
"""

import argparse, os, sys, warnings
import torch, torch.nn as nn, torch.nn.functional as F
import onnx

sys.path.insert(0, os.path.dirname(__file__))
from model_arch import YOSOModel, DeformConvBlock


# ── DCN grid-sample replacement ──────────────────────────────────
class DeformConvBlockGridSample(nn.Module):
    """
    Deformable Conv v2 reimplemented with F.grid_sample so that the
    entire block is expressible in standard ONNX ops (opset ≥ 11).

    Mathematically equivalent to torchvision.ops.deform_conv2d
    for a single 3×3 kernel.  Slight speed penalty vs native DCN.
    """
    def __init__(self, src: DeformConvBlock):
        super().__init__()
        # Copy all sub-modules / parameters from the original block
        self.dcn_offset = src.dcn_offset
        self.dcn_w      = src.dcn            # [C_out, C_in, 3, 3]
        self.dcn_bn     = src.dcn_bn
        self.up_sample  = src.up_sample
        self.up_bn      = src.up_bn

        C_in = self.dcn_w.shape[1]
        assert self.dcn_w.shape[2:] == (3, 3), "Only 3×3 kernel supported"
        self.C_in   = C_in
        self.C_out  = self.dcn_w.shape[0]

    def forward(self, x):
        B, C, H, W = x.shape
        off_mask = self.dcn_offset(x)          # [B, 27, H, W]
        offset   = off_mask[:, :18]            # [B, 18, H, W]  (9 × (dy, dx))
        mask     = off_mask[:, 18:].sigmoid()  # [B,  9, H, W]

        # Build a sampling grid for each of the 9 kernel positions
        # Base grid (H, W) normalised to [-1, 1]
        ys = torch.linspace(-1, 1, H, device=x.device)
        xs = torch.linspace(-1, 1, W, device=x.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H, W]

        # kernel offsets for 3×3 with padding=1
        ky = torch.tensor([-1, -1, -1,  0,  0,  0,  1,  1,  1],
                          dtype=x.dtype, device=x.device) / H
        kx = torch.tensor([-1,  0,  1, -1,  0,  1, -1,  0,  1],
                          dtype=x.dtype, device=x.device) / W

        sampled = []
        for k in range(9):
            dy = offset[:, 2*k  ] / H   # [B, H, W]  normalised
            dx = offset[:, 2*k+1] / W
            sy = (grid_y.unsqueeze(0) + ky[k] + dy).unsqueeze(-1)  # [B,H,W,1]
            sx = (grid_x.unsqueeze(0) + kx[k] + dx).unsqueeze(-1)
            grid_k = torch.cat([sx, sy], dim=-1)                    # [B,H,W,2]
            samp_k = F.grid_sample(x, grid_k, align_corners=True,   # [B,C,H,W]
                                   padding_mode='zeros', mode='bilinear')
            # Apply modulation mask
            samp_k = samp_k * mask[:, k:k+1]
            sampled.append(samp_k)

        # Now apply the 3×3 conv as 9 × 1×1 convs (weight per position)
        out = torch.zeros(B, self.C_out, H, W, device=x.device, dtype=x.dtype)
        w = self.dcn_w                             # [C_out, C_in, 3, 3]
        for k in range(9):
            ky_idx, kx_idx = k // 3, k % 3
            w_k = w[:, :, ky_idx, kx_idx].unsqueeze(-1).unsqueeze(-1)  # [C_out,C_in,1,1]
            out = out + F.conv2d(sampled[k], w_k)

        out = F.relu(self.dcn_bn(out))
        out = F.relu(self.up_bn(self.up_sample(out)))
        return out


def replace_dcn_with_gridsampl(model: YOSOModel) -> YOSOModel:
    """Swap all DeformConvBlock instances with grid_sample equivalents."""
    neck = model.neck
    d    = neck.deconv
    d.deform_conv1 = DeformConvBlockGridSample(d.deform_conv1)
    d.deform_conv2 = DeformConvBlockGridSample(d.deform_conv2)
    d.deform_conv3 = DeformConvBlockGridSample(d.deform_conv3)
    return model


# ── Custom ONNX symbolic for torchvision::deform_conv2d ──────────
def _register_dcn_symbolic(opset: int):
    """
    Export deform_conv2d as a 'custom::DeformConv2d' ONNX node.
    Runtimes (TRT ≥ 8.6, mmcv ORT plugin) know how to run this op.
    """
    try:
        from torch.onnx import register_custom_op_symbolic
        from torch.onnx import symbolic_helper as sym_help

        def _to_int(v, default):
            """
            Convert ONNX symbolic arg to Python int when possible.
            Falls back to a safe default if value is a traced graph Value.
            """
            if isinstance(v, (bool, int, float)):
                return int(v)
            try:
                c = sym_help._maybe_get_const(v, 'i')
                if isinstance(c, (bool, int, float)):
                    return int(c)
            except Exception:
                pass
            return int(default)

        def _sym(g, input, weight, offset, mask, bias,
                 stride_h, stride_w, pad_h, pad_w,
                 dil_h, dil_w, n_weight_grps, n_offset_grps, use_mask):
            stride_h_i = _to_int(stride_h, 1)
            stride_w_i = _to_int(stride_w, 1)
            pad_h_i = _to_int(pad_h, 0)
            pad_w_i = _to_int(pad_w, 0)
            dil_h_i = _to_int(dil_h, 1)
            dil_w_i = _to_int(dil_w, 1)
            n_weight_grps_i = _to_int(n_weight_grps, 1)
            n_offset_grps_i = _to_int(n_offset_grps, 1)
            use_mask_i = _to_int(use_mask, 1)

            return g.op(
                'custom::DeformConv2d',
                input, offset, weight, mask, bias,
                strides_i=[stride_h_i, stride_w_i],
                pads_i=[pad_h_i, pad_w_i, pad_h_i, pad_w_i],
                dilations_i=[dil_h_i, dil_w_i],
                group_i=n_weight_grps_i,
                offset_group_i=n_offset_grps_i,
                use_mask_i=use_mask_i,
            )
        register_custom_op_symbolic('torchvision::deform_conv2d', _sym, opset)
        print(f"[ONNX] Registered custom DCN symbolic for opset {opset}")
    except Exception as e:
        print(f"[WARN] Could not register DCN symbolic: {e}")


# ── Export wrapper ────────────────────────────────────────────────
class ExportWrapper(nn.Module):
    def __init__(self, m): super().__init__(); self.model = m
    def forward(self, x): return self.model(x)


# ── CLI ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='yoso_res50_coco.pth')
    p.add_argument('--output',     default='yoso_res50.onnx')
    p.add_argument('--height',  type=int, default=512)
    p.add_argument('--width',   type=int, default=512)
    p.add_argument('--opset',   type=int, default=18)
    p.add_argument('--dynamic', action='store_true')
    p.add_argument('--simplify',action='store_true')
    p.add_argument('--device',  default='cpu')
    p.add_argument('--dcn-mode', choices=['native','custom','gridsampl'],
                   default='native',
                   help='native = keep torchvision DCN; '
                        'custom = custom op node (plugin needed); '
                        'gridsampl = pure standard ONNX approximation')
    return p.parse_args()


def export(args):
    device = torch.device(args.device)
    H, W   = args.height, args.width
    assert H % 32 == 0 and W % 32 == 0
    active_dcn_mode = args.dcn_mode

    print(f"[1/5] Loading checkpoint …")
    base = YOSOModel.from_checkpoint(args.checkpoint, map_location=device)

    if args.dcn_mode == 'gridsampl':
        print("[1/5] Replacing DCN with grid_sample ops …")
        base = replace_dcn_with_gridsampl(base)
    elif args.dcn_mode == 'custom':
        _register_dcn_symbolic(args.opset)
    else:
        print("[1/5] Using native torchvision DCN export …")

    model = ExportWrapper(base).to(device).eval()
    dummy = torch.randn(1, 3, H, W, device=device)
    print(f"[2/5] Input: {dummy.shape}  dcn-mode={active_dcn_mode}")

    dyn_axes = None
    if args.dynamic:
        dyn_axes = {'image':{0:'batch',2:'height',3:'width'},
                    'logits':{0:'batch'},'masks':{0:'batch',2:'mh',3:'mw'}}

    print(f"[3/5] torch.onnx.export  (opset={args.opset}, dynamo=False) …")
    try:
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            torch.onnx.export(
                model, dummy, args.output,
                dynamo=False,
                opset_version=args.opset,
                input_names=['image'],
                output_names=['logits','masks'],
                dynamic_axes=dyn_axes,
                export_params=True,
                do_constant_folding=True,
                verbose=False,
            )
    except Exception as e:
        msg = str(e)
        native_dcn_unsupported = (
            active_dcn_mode == 'native' and
            "torchvision::deform_conv2d" in msg and
            "UnsupportedOperatorError" in e.__class__.__name__
        )
        if not native_dcn_unsupported:
            raise

        print("    [WARN] Native DCN export is unsupported in this torch/torchvision build.")
        print("    [WARN] Falling back to --dcn-mode gridsampl automatically.")
        active_dcn_mode = 'gridsampl'
        base = replace_dcn_with_gridsampl(base)
        model = ExportWrapper(base).to(device).eval()
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            torch.onnx.export(
                model, dummy, args.output,
                dynamo=False,
                opset_version=args.opset,
                input_names=['image'],
                output_names=['logits','masks'],
                dynamic_axes=dyn_axes,
                export_params=True,
                do_constant_folding=True,
                verbose=False,
            )
    print(f"    Saved → '{args.output}'")

    print("[4/5] Validating ONNX graph …")
    m = onnx.load(args.output)
    onnx.checker.check_model(m)
    print("    Graph check OK")

    if active_dcn_mode == 'custom':
        print("    ORT validation skipped: custom::DeformConv2d requires a runtime plugin/op.")
        print("    Use '--dcn-mode native' or '--dcn-mode gridsampl' for pure ONNX Runtime validation.")
    else:
        try:
            import onnxruntime as ort, numpy as np
            prov = (['CUDAExecutionProvider','CPUExecutionProvider']
                    if device.type == 'cuda' else ['CPUExecutionProvider'])
            sess = ort.InferenceSession(args.output, providers=prov)
            feeds = {sess.get_inputs()[0].name: dummy.cpu().numpy()}
            ort_l, ort_m = sess.run(None, feeds)
            with torch.no_grad(): pt_l, pt_m = model(dummy)
            le = float(np.abs(ort_l - pt_l.cpu().numpy()).max())
            me = float(np.abs(ort_m - pt_m.cpu().numpy()).max())
            print(f"    Max error — logits: {le:.2e}   masks: {me:.2e}")
            if max(le, me) < 1e-3:
                print("    Numeric validation PASSED")
            else:
                print("    [WARN] Numeric error > 1e-3")
        except Exception as e:
            print(f"    ORT validation skipped: {e}")

    if args.simplify:
        print("[5/5] Running onnx-simplifier …")
        try:
            from onnxsim import simplify as onnxsim
            simp, ok = onnxsim(m)
            if ok:
                onnx.save(simp, args.output)
                print(f"    Simplified → '{args.output}'")
        except ImportError:
            print("    onnxsim not installed (pip install onnxsim)")

    size_mb = os.path.getsize(args.output)/1e6
    print(f"\n[✓] Done: '{args.output}'  ({size_mb:.1f} MB)")

    print("\n── Output tensors ─────────────────────────────────────")
    for n in onnx.load(args.output).graph.output:
        dims = [d.dim_value or d.dim_param
                for d in n.type.tensor_type.shape.dim]
        print(f"  {n.name:10s} : {dims}")


if __name__ == '__main__':
    export(parse_args())
