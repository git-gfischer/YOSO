# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import sys
import tempfile
import time
import warnings
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from config import add_yoso_config
from projects.YOSO.yoso.segmentator import YOSO

# constants
WINDOW_NAME = "COCO detections"


def _update_fps_ema(prev_ema, dt_sec, alpha=0.2):
    """EWMA of FPS where dt_sec is seconds per frame (e.g. model inference time); inst = 1/dt_sec."""
    if dt_sec <= 0:
        return prev_ema
    inst = 1.0 / dt_sec
    if prev_ema is None:
        return inst
    return (1.0 - alpha) * prev_ema + alpha * inst


def _draw_fps_overlay(bgr_frame, fps, label="Infer FPS"):
    """Draw FPS in the top-left corner (BGR image, modified in place)."""
    cv2.putText(
        bgr_frame,
        "{}: {:.1f}".format(label, fps),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_yoso_config(cfg)
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.YOSO.TEST.OVERLAP_THRESHOLD = args.overlap_threshold
    cfg.MODEL.YOSO.TEST.OBJECT_MASK_THRESHOLD = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index for --webcam (ignored if --camera-device is set).",
    )
    parser.add_argument(
        "--camera-device",
        default=None,
        metavar="PATH",
        help="Optional V4L2 device path (e.g. /dev/video0). If not set, uses --camera-index with cv2.VideoCapture.",
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--no-fps-overlay",
        action="store_true",
        help="Do not draw FPS on webcam / video preview windows.",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=512,
        metavar="W",
        help="Resize width for webcam / video frames before inference (default 512).",
    )
    parser.add_argument(
        "--input-height",
        type=int,
        default=256,
        metavar="H",
        help="Resize height for webcam / video frames before inference (default 256).",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.98,
        help="overlap threshold",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.8,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


def open_webcam_cv2(camera_device, camera_index, logger):
    """Open webcam using only cv2.VideoCapture (optionally CAP_V4L2 on Linux for /dev/video*)."""
    if camera_device:
        path = os.path.expanduser(camera_device)
        if not os.path.exists(path):
            raise FileNotFoundError(
                "Camera device not found: {!r}. Check the path and Docker --device.".format(path)
            )
        if sys.platform.startswith("linux") and hasattr(cv2, "CAP_V4L2"):
            cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(path)
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            "Could not open webcam (camera_index={!r}, camera_device={!r}). "
            "Try another index or --camera-device from `v4l2-ctl --list-devices`.".format(
                camera_index, camera_device
            )
        )
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    logger.info("Opened webcam with cv2.VideoCapture")
    return cap



def _args_for_logging(args_ns):
    """Return an argparse.Namespace clone safe to stringify (avoid dumping huge --input lists)."""
    out = argparse.Namespace(**vars(args_ns))
    inp = getattr(out, "input", None)
    if isinstance(inp, (list, tuple)) and len(inp) > 1:
        out.input = "[<{} input paths>]".format(len(inp))
    return out


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: {}".format(_args_for_logging(args)))

    cfg = setup_cfg(args)

    if args.input_width < 1 or args.input_height < 1:
        raise ValueError("--input-width and --input-height must be positive.")

    demo = VisualizationDemo(
        cfg,
        parallel=False,
        input_size=(args.input_width, args.input_height),
    )

    if args.input:
        if len(args.input) == 1:
            input_pattern = os.path.expanduser(args.input[0])
            args.input = glob.glob(input_pattern)
            assert args.input, (
                "No files matched {!r} (empty glob or path does not exist). "
                "If your images are not *.jpg, set IMAGE_GLOB when using dataset_inference.sh."
            ).format(input_pattern)
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            img = cv2.resize(
                img,
                (args.input_width, args.input_height),
                interpolation=cv2.INTER_LINEAR,
            )
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = open_webcam_cv2(args.camera_device, args.camera_index, logger)
        fps_ema = None
        for vis, infer_sec in tqdm.tqdm(
            demo.run_on_video(cam, yield_inference_time=True), desc="webcam"
        ):
            fps_ema = _update_fps_ema(fps_ema, infer_sec)
            if not args.no_fps_overlay and fps_ema is not None:
                _draw_fps_overlay(vis, fps_ema)
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = args.input_width
        height = args.input_height
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        fps_ema = None
        for vis_frame, infer_sec in tqdm.tqdm(
            demo.run_on_video(video, yield_inference_time=True), total=num_frames, desc="video"
        ):
            if args.output:
                output_file.write(vis_frame)
            else:
                fps_ema = _update_fps_ema(fps_ema, infer_sec)
                if not args.no_fps_overlay and fps_ema is not None:
                    _draw_fps_overlay(vis_frame, fps_ema)
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
