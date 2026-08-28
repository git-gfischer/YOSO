import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load YOSO / Detectron2 model and run a dummy forward pass."
    )
    parser.add_argument(
        "--yoso-path",
        type=str,
        default=os.environ.get("YOSO_PROJECTS_PATH", "/root/YOSO/projects/YOSO"),
        help="Directory prepended to sys.path so `yoso` can be imported. "
        "Default: $YOSO_PROJECTS_PATH or /root/YOSO/projects/YOSO.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the model config YAML.",
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        required=True,
        help="Path to the checkpoint weights (.pth).",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    sys.path.insert(0, args.yoso_path)

    import torch
    from detectron2.config import get_cfg
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model

    from yoso import add_yoso_config

    def setup_cfg(config_file, weights_file):
        cfg = get_cfg()
        add_yoso_config(cfg)
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = weights_file
        cfg.MODEL.DEVICE = "cpu"
        cfg.freeze()
        return cfg

    cfg = setup_cfg(args.config, args.weights)

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    dummy = torch.randn(3, 480, 640)

    with torch.no_grad():
        out = model([{"image": dummy}])

    print(type(out))
    print(out)

if __name__ == "__main__":
    main()
