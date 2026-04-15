# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import re
import shutil
import subprocess
import threading
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


def _update_fps_ema(prev_ema, dt_sec, alpha=0.15):
    if dt_sec <= 0:
        return prev_ema
    inst = 1.0 / dt_sec
    if prev_ema is None:
        return inst
    return (1.0 - alpha) * prev_ema + alpha * inst


def _draw_fps_overlay(bgr_frame, fps):
    """Draw FPS in the top-left corner (BGR image, modified in place)."""
    cv2.putText(
        bgr_frame,
        "FPS: {:.1f}".format(fps),
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
        help="V4L2 device path (e.g. /dev/video4). Prefer this in Docker or when --camera-index fails; "
        "usually pairs with: docker --device /dev/video4:/dev/video4",
    )
    parser.add_argument(
        "--webcam-use-ffmpeg",
        action="store_true",
        help="Capture via ffmpeg v4l2 instead of OpenCV (recommended in Docker if OpenCV cannot open the device).",
    )
    parser.add_argument(
        "--webcam-size",
        default="640x480",
        metavar="WxH",
        help="Resolution for ffmpeg capture (e.g. 640x480 or 1280x720). Tried first, then other common sizes.",
    )
    parser.add_argument(
        "--webcam-fps",
        type=int,
        default=30,
        help="Framerate for ffmpeg webcam capture.",
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


def _try_open_capture(src, backend, logger, label):
    """Try VideoCapture; optionally verify we can read at least one frame."""
    cap = cv2.VideoCapture(src, backend) if backend is not None else cv2.VideoCapture(src)
    if not cap.isOpened():
        cap.release()
        return None
    ok, frame = False, None
    for _ in range(20):
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap
        time.sleep(0.05)
    logger.warning("%s: opened but no frames read from %r; trying next.", label, src)
    cap.release()
    return None


def _parse_webcam_size(s):
    s = s.strip().lower().replace("*", "x")
    if "x" not in s:
        raise ValueError("--webcam-size must be like 640x480")
    a, b = s.split("x", 1)
    return int(a), int(b)


def _webcam_size_tries(first_wh):
    """User preference first, then common modes."""
    rest = [(640, 480), (1280, 720), (320, 240), (800, 600), (640, 360), (1920, 1080)]
    out = [first_wh]
    for r in rest:
        if r not in out:
            out.append(r)
    return out


class FFmpegV4l2Capture:
    def __init__(self, proc, width, height):
        self._proc = proc
        self.width = width
        self.height = height
        self._frame_size = width * height * 3

    def isOpened(self):
        return self._proc is not None and self._proc.poll() is None

    def read(self):
        if self._proc is None:
            return False, None
        raw = self._proc.stdout.read(self._frame_size)
        if len(raw) != self._frame_size:
            return False, None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
        return True, frame

    def release(self):
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        try:
            self._proc.stdout.close()
        except Exception:
            pass
        try:
            self._proc.stderr.close()
        except Exception:
            pass
        self._proc = None


def _try_ffmpeg_v4l2(device, logger, size_tries, fps):
    """
    Open /dev/video* via ffmpeg's v4l2 input. Works in many Docker setups where OpenCV's
    VideoIO does not.
    """
    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg not found in PATH; cannot use ffmpeg webcam capture.")
        return None
    if not os.path.exists(device):
        return None
    for w, h in size_tries:
        for input_format in (None, "mjpeg", "yuyv422"):
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "v4l2"]
            if input_format:
                cmd += ["-input_format", input_format]
            cmd += [
                "-framerate",
                str(fps),
                "-video_size",
                "{}x{}".format(w, h),
                "-i",
                device,
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "-",
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
            )
            time.sleep(0.45)
            if proc.poll() is not None:
                err = proc.stderr.read().decode("utf-8", errors="replace")
                logger.info(
                    "ffmpeg failed %dx%d fmt=%s: %s",
                    w,
                    h,
                    input_format,
                    err[:300].replace("\n", " "),
                )
                continue
            frame_size = w * h * 3
            raw = proc.stdout.read(frame_size)
            if len(raw) == frame_size:
                logger.info(
                    "Using ffmpeg v4l2 capture on %r at %dx%d (input_format=%s)",
                    device,
                    w,
                    h,
                    input_format,
                )
                # Avoid stderr PIPE filling and stalling ffmpeg on long runs.
                threading.Thread(target=lambda: proc.stderr.read(), daemon=True).start()
                return FFmpegV4l2Capture(proc, w, h)
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
    return None


def open_webcam_capture(
    camera_device,
    camera_index,
    logger,
    use_ffmpeg_only=False,
    webcam_size="640x480",
    webcam_fps=30,
):
    """
    Open a camera for --webcam. On Linux, prefer the V4L2 backend and try both device path
    and numeric index. If OpenCV fails (common in Docker), fall back to ffmpeg v4l2.
    """
    first_wh = _parse_webcam_size(webcam_size)
    size_tries = _webcam_size_tries(first_wh)

    def _v4l_device_path():
        if camera_device:
            p = os.path.expanduser(camera_device)
            if not os.path.exists(p):
                raise FileNotFoundError(
                    "Camera device not found: {!r}. Check the path and Docker --device.".format(p)
                )
            return p
        return "/dev/video{}".format(camera_index)

    vdev = _v4l_device_path()

    if use_ffmpeg_only:
        cap = _try_ffmpeg_v4l2(vdev, logger, size_tries, webcam_fps)
        if cap is not None:
            return cap
        raise RuntimeError(
            "ffmpeg could not open {!r}. Install ffmpeg, check v4l2-ctl --list-devices for the "
            "capture node, try --webcam-size 1280x720, and ensure the device is readable in the "
            "container.".format(vdev)
        )

    if camera_device:
        src_path = os.path.expanduser(camera_device)
        attempts = []
        m = re.match(r".*/video(\d+)$", src_path)
        idx = int(m.group(1)) if m else None
        if sys.platform.startswith("linux") and hasattr(cv2, "CAP_V4L2"):
            attempts.append((src_path, cv2.CAP_V4L2, "V4L2 device path"))
            if idx is not None:
                attempts.append((idx, cv2.CAP_V4L2, "V4L2 numeric index"))
        attempts.append((src_path, None, "default backend, device path"))
        if idx is not None:
            attempts.append((idx, None, "default backend, numeric index"))
    else:
        attempts = []
        if sys.platform.startswith("linux") and hasattr(cv2, "CAP_V4L2"):
            attempts.append((camera_index, cv2.CAP_V4L2, "V4L2 index"))
        attempts.append((camera_index, None, "default index"))

    for src, backend, label in attempts:
        cap = _try_open_capture(src, backend, logger, label)
        if cap is not None:
            logger.info("Using camera %s (src=%r)", label, src)
            return cap

    logger.warning(
        "OpenCV could not capture from the camera; trying ffmpeg v4l2 fallback (same as --webcam-use-ffmpeg)."
    )
    cap = _try_ffmpeg_v4l2(vdev, logger, size_tries, webcam_fps)
    if cap is not None:
        return cap

    raise RuntimeError(
        "Could not open camera with OpenCV or ffmpeg. Tried OpenCV: {}. "
        "For v4l2: run `v4l2-ctl --list-devices` and use the /dev/video* line for **Video Capture** "
        "(not metadata). Try: --webcam-use-ffmpeg --webcam-size 1280x720 --camera-device /dev/videoN. "
        "Ensure ffmpeg is installed and the device is passed into Docker (e.g. --device /dev/videoN)."
        .format(", ".join([a[2] for a in attempts]))
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg, parallel=False)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
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
        cam = open_webcam_capture(
            args.camera_device,
            args.camera_index,
            logger,
            use_ffmpeg_only=args.webcam_use_ffmpeg,
            webcam_size=args.webcam_size,
            webcam_fps=args.webcam_fps,
        )
        fps_ema = None
        prev_t = time.perf_counter()
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            now = time.perf_counter()
            fps_ema = _update_fps_ema(fps_ema, now - prev_t)
            prev_t = now
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
        width = 512 #int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = 256 #int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(width, height)

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
        prev_t = time.perf_counter()
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            now = time.perf_counter()
            fps_ema = _update_fps_ema(fps_ema, now - prev_t)
            prev_t = now
            if args.output:
                output_file.write(vis_frame)
            else:
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
