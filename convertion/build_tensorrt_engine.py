# -*- coding: utf-8 -*-
"""
build_tensorrt_engine.py  –  Convert YOSO ONNX model → TensorRT engine

Workflow
--------
    1. python export_to_onnx.py --output yoso_res50.onnx [--dynamic]
    2. python build_tensorrt_engine.py \
            --onnx  yoso_res50.onnx    \
            --engine yoso_res50.engine \
            [--fp16]                   \
            [--int8]                   \
            [--min  1,3,384,384]       \
            [--opt  1,3,512,512]       \
            [--max  1,3,768,768]       \
            [--workspace 4096]             (MB)

Requirements
------------
    TensorRT ≥ 8.6   (pip install tensorrt  or  use NVIDIA container)
    pycuda           (pip install pycuda)
    CUDA GPU

Notes on deformable convolution
--------------------------------
TensorRT ≥ 8.6 supports the standard ONNX deform_conv2d node natively.
For older TRT versions you need a custom plugin; see:
  https://github.com/NVIDIA/TensorRT/tree/main/plugin/deformableConvPlugin
"""

import argparse
import os
import sys
import struct
import time


# ── helpers ──────────────────────────────────────────────────────
def _check_trt():
    try:
        import tensorrt as trt
        print(f"[TRT] TensorRT {trt.__version__} found")
        return trt
    except ImportError:
        sys.exit("[ERROR] TensorRT not found.  Install: pip install tensorrt")


def parse_shape(s):
    """'1,3,512,512' → (1,3,512,512)"""
    return tuple(int(x) for x in s.split(','))


def parse_args():
    p = argparse.ArgumentParser(description='Build TensorRT engine from YOSO ONNX')
    p.add_argument('--onnx',      default='yoso_res50.onnx')
    p.add_argument('--engine',    default='yoso_res50.engine')
    p.add_argument('--fp16',      action='store_true', help='Enable FP16 precision')
    p.add_argument('--int8',      action='store_true', help='Enable INT8 (requires calibrator)')
    p.add_argument('--workspace', type=int, default=4096,
                   help='GPU memory workspace in MB')
    p.add_argument('--min',  default='1,3,384,384', help='Min input shape  (dynamic mode)')
    p.add_argument('--opt',  default='1,3,512,512', help='Optimal input shape')
    p.add_argument('--max',  default='1,3,768,768', help='Max input shape')
    p.add_argument('--static', action='store_true',
                   help='Static shape mode (uses --opt shape only)')
    return p.parse_args()


# ── INT8 calibrator (simple random-data calibrator for structure) ─
def _make_int8_calibrator(trt, min_shape, opt_shape, max_shape,
                          calib_data_dir=None, cache_file='calib.cache'):
    """
    Minimal entropy calibrator.  For production, replace with real
    representative data from your dataset.
    """
    import numpy as np
    import pycuda.driver as cuda
    import pycuda.autoinit

    class RandomCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, n_batches=100):
            super().__init__()
            self.n_batches   = n_batches
            self.batch_idx   = 0
            self.shape       = opt_shape
            self.device_input = cuda.mem_alloc(
                int(np.prod(self.shape)) * np.dtype(np.float32).itemsize)
            self.cache_file  = cache_file

        def get_batch_size(self):
            return self.shape[0]

        def get_batch(self, names):
            if self.batch_idx >= self.n_batches:
                return None
            batch = np.random.randn(*self.shape).astype(np.float32)
            cuda.memcpy_htod(self.device_input, batch)
            self.batch_idx += 1
            return [int(self.device_input)]

        def read_calibration_cache(self):
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    return f.read()

        def write_calibration_cache(self, cache):
            with open(self.cache_file, 'wb') as f:
                f.write(cache)

    return RandomCalibrator()


# ── main build function ───────────────────────────────────────────
def build_engine(args):
    trt = _check_trt()
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    print(f"[1/4] Reading ONNX model: '{args.onnx}'")
    if not os.path.exists(args.onnx):
        sys.exit(f"[ERROR] ONNX file not found: {args.onnx}")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(args.onnx, 'rb') as f:
        if not parser.parse(f.read()):
            plugin_missing = False
            for i in range(parser.num_errors):
                err = str(parser.get_error(i))
                print(f"  [PARSE ERROR {i}] {err}")
                if ("DeformConv2d" in err and
                        ("Plugin not found" in err or "getPluginCreator" in err)):
                    plugin_missing = True
            if plugin_missing:
                print("[HINT] ONNX uses custom DeformConv2d plugin nodes.")
                print("[HINT] Re-export with '--dcn-mode gridsampl' for plugin-free TensorRT parsing.")
            sys.exit("[ERROR] ONNX parse failed")

    print(f"    Network inputs : {network.num_inputs}")
    print(f"    Network outputs: {network.num_outputs}")

    # ── builder config ───────────────────────────────────────────
    print("[2/4] Configuring builder …")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, args.workspace * (1 << 20))

    if args.fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("    FP16 enabled")
        else:
            print("    [WARN] Platform does not support fast FP16; ignoring --fp16")

    if args.int8:
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            min_s = parse_shape(args.min)
            opt_s = parse_shape(args.opt)
            max_s = parse_shape(args.max)
            config.int8_calibrator = _make_int8_calibrator(trt, min_s, opt_s, max_s)
            print("    INT8 enabled (random calibration – replace with real data)")
        else:
            print("    [WARN] Platform does not support fast INT8; ignoring --int8")

    # ── optimization profile (dynamic shapes) ───────────────────
    if not args.static:
        profile = builder.create_optimization_profile()
        inp     = network.get_input(0)
        min_s   = parse_shape(args.min)
        opt_s   = parse_shape(args.opt)
        max_s   = parse_shape(args.max)
        profile.set_shape(inp.name, min_s, opt_s, max_s)
        config.add_optimization_profile(profile)
        print(f"    Dynamic shape profile:")
        print(f"      min={min_s}  opt={opt_s}  max={max_s}")
    else:
        opt_s = parse_shape(args.opt)
        print(f"    Static shape: {opt_s}")

    # ── build ────────────────────────────────────────────────────
    print("[3/4] Building TensorRT engine (this may take a few minutes) …")
    t0            = time.time()
    serialized    = builder.build_serialized_network(network, config)

    if serialized is None:
        sys.exit("[ERROR] Engine build failed")

    elapsed = time.time() - t0
    print(f"    Build time: {elapsed:.1f}s")

    # ── save ─────────────────────────────────────────────────────
    print(f"[4/4] Saving engine → '{args.engine}' …")
    with open(args.engine, 'wb') as f:
        f.write(serialized)

    size_mb = os.path.getsize(args.engine) / 1e6
    print(f"\n[✓] TensorRT engine saved: '{args.engine}'  ({size_mb:.1f} MB)")
    print("""
── Next step: C++ inference ────────────────────────────────────────────
  Compile:
    cd cpp && mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)

  Run:
    ./yoso_infer --engine ../../yoso_res50.engine --image /path/to/image.jpg
────────────────────────────────────────────────────────────────────────
""")


if __name__ == '__main__':
    build_engine(parse_args())
