"""Face detection via FaceFusion-compatible YOLO Face 8n ONNX.

Outputs FaceHit with 5-point landmarks (eyes, nose, mouth corners) used by ArcFace / inswapper,
plus bbox and confidence. No MediaPipe.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("fsp")

DEFAULT_MIN_CONF = 0.50
DEFAULT_MIN_FACE_FRAC = 0.025
DEFAULT_DETECTOR_SIZE = 640  # FaceFusion default for yolo_face


@dataclass
class DetectOpts:
    min_confidence: float = DEFAULT_MIN_CONF
    min_face_frac: float = DEFAULT_MIN_FACE_FRAC
    max_faces: int = 8
    detector_size: int = DEFAULT_DETECTOR_SIZE


@dataclass
class FaceHit:
    """One accepted face with 5-point landmarks."""
    kps: np.ndarray              # (5, 2) float32 — left_eye, right_eye, nose, left_mouth, right_mouth
    confidence: float
    gender: Optional[str] = None
    age: Optional[int] = None
    bbox: tuple = (0.0, 0.0, 0.0, 0.0)
    pts: np.ndarray = field(default=None, repr=False)  # alias for tracking (Nx2); defaults to kps

    def __post_init__(self):
        self.kps = np.asarray(self.kps, np.float32).reshape(5, 2)
        if self.pts is None:
            self.pts = self.kps.copy()
        if self.bbox == (0.0, 0.0, 0.0, 0.0):
            x0, y0 = self.kps.min(0); x1, y1 = self.kps.max(0)
            # expand bbox from kps
            w, h = x1 - x0, y1 - y0
            self.bbox = (float(x0 - 0.3 * w), float(y0 - 0.45 * h), float(x1 + 0.3 * w), float(y1 + 0.25 * h))

    @property
    def pts468(self) -> np.ndarray:
        """Compatibility shim: return kps (callers that expected MediaPipe pts use kps5())."""
        return self.kps

    @property
    def pts5(self) -> np.ndarray:
        return self.kps


def _nms(boxes, scores, thr=0.4):
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, np.float32)
    scores = np.asarray(scores, np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0]); keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= thr]
    return keep


def as_bgr_uint8(img):
    """Reject empty / weird frames before they reach ONNX (a bad shape can abort DirectML)."""
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("Empty frame — nothing to scan for faces.")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"Unexpected image shape {getattr(img, 'shape', None)}.")
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    h, w = int(img.shape[0]), int(img.shape[1])
    if h < 16 or w < 16:
        raise ValueError(f"Image is too small to detect faces ({w}×{h}).")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def _fit_square(img_bgr, size):
    """Scale so the long side is <= size. Never produce a side of size+1 (round-up) or 0."""
    h0, w0 = img_bgr.shape[:2]
    s = min(1.0, float(size) / float(max(h0, w0)))
    tw = max(1, min(int(size), int(round(w0 * s))))
    th = max(1, min(int(size), int(round(h0 * s))))
    if (tw, th) != (w0, h0):
        temp = cv2.resize(img_bgr, (tw, th), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    else:
        temp = img_bgr
    return temp, w0 / float(tw), h0 / float(th)


def _letterbox(img, size):
    """Resize keeping aspect, pad to size×size (FaceFusion prepare_detect_frame style)."""
    h, w = img.shape[:2]
    s = min(size / h, size / w)
    nh, nw = int(round(h * s)), int(round(w * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:nh, :nw] = resized
    return canvas, s, nh, nw


class FaceDetector:
    """ONNX face detector (YOLO Face or RetinaFace) run through the shared Engine lock."""

    def __init__(self, engine, model_name="yoloface_8n", size=640):
        self.engine = engine
        self.model_name = model_name
        self.size = size
        self._session_ready = False

    def _ensure(self):
        if not self._session_ready:
            self.engine.session(self.model_name)
            self._session_ready = True

    def detect(self, img_bgr, opts: DetectOpts | None = None) -> list[FaceHit]:
        opts = opts or DetectOpts()
        img_bgr = as_bgr_uint8(img_bgr)
        self._ensure()
        size = int(opts.detector_size or self.size)
        if size < 32 or size > 2048:
            size = DEFAULT_DETECTOR_SIZE
        try:
            return self._detect_yolo(img_bgr, opts, size)
        except ValueError:
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("face detector failed")
            raise RuntimeError(
                f"Face detection failed ({type(e).__name__}: {e}). "
                "If this keeps happening, set Processor to CPU in Options. "
                "A log was written to %LOCALAPPDATA%\\FaceSwapPhoto\\crash.log."
            ) from e

    def _detect_yolo(self, img_bgr, opts, size):
        # FaceFusion detect_with_yolo_face
        h0, w0 = img_bgr.shape[:2]
        temp, ratio_w, ratio_h = _fit_square(img_bgr, size)
        th, tw = temp.shape[:2]
        canvas = np.zeros((size, size, 3), np.float32)
        canvas[:th, :tw] = temp.astype(np.float32)
        inp = (canvas.transpose(2, 0, 1)[None] / 255.0).astype(np.float32)
        out = self.engine.run(self.model_name, {"input": inp})[0]
        det = np.squeeze(np.asarray(out))
        if det.ndim != 2:
            log.warning("unexpected yolo output shape %s", getattr(out, "shape", None))
            return []
        # Channel axis is the small one (20 = box4 + score + 15 landmark values), anchors are the long one.
        if det.shape[0] <= 64 and det.shape[1] > det.shape[0]:
            det = det.T
        elif det.shape[1] <= 64 and det.shape[0] > det.shape[1]:
            pass
        else:
            det = det.T
        if det.ndim != 2 or det.shape[1] < 5:
            return []
        boxes_raw = det[:, :4]
        scores = det[:, 4]
        lms_raw = det[:, 5:5 + 15] if det.shape[1] >= 20 else None
        keep = np.where(scores > opts.min_confidence)[0]
        if keep.size == 0:
            return []
        boxes, sc, lms = [], [], []
        for i in keep:
            cx, cy, bw, bh = boxes_raw[i]
            x0 = (cx - bw / 2) * ratio_w
            y0 = (cy - bh / 2) * ratio_h
            x1 = (cx + bw / 2) * ratio_w
            y1 = (cy + bh / 2) * ratio_h
            boxes.append([x0, y0, x1, y1])
            sc.append(float(scores[i]))
            if lms_raw is not None and lms_raw[i].size % 3 == 0:
                lm = lms_raw[i].reshape(-1, 3)[:, :2].copy()
                if lm.shape[0] < 5 or not np.isfinite(lm).all():
                    lm = None
                else:
                    lm[:, 0] *= ratio_w
                    lm[:, 1] *= ratio_h
                    lms.append(lm.astype(np.float32))
            else:
                lm = None
            if lms_raw is None or lm is None:
                # synthesize 5 pts from bbox
                lms.append(np.array([
                    [x0 + 0.3 * (x1 - x0), y0 + 0.35 * (y1 - y0)],
                    [x0 + 0.7 * (x1 - x0), y0 + 0.35 * (y1 - y0)],
                    [x0 + 0.5 * (x1 - x0), y0 + 0.55 * (y1 - y0)],
                    [x0 + 0.35 * (x1 - x0), y0 + 0.75 * (y1 - y0)],
                    [x0 + 0.65 * (x1 - x0), y0 + 0.75 * (y1 - y0)],
                ], np.float32))
        idx = _nms(boxes, sc, 0.4)
        short = min(h0, w0)
        hits = []
        for i in idx[:opts.max_faces]:
            b = boxes[i]
            fw, fh = b[2] - b[0], b[3] - b[1]
            if not np.isfinite(b).all() or not np.isfinite(lms[i]).all():
                continue
            if min(fw, fh) / short < opts.min_face_frac:
                continue
            hits.append(FaceHit(kps=lms[i], confidence=sc[i], bbox=(float(b[0]), float(b[1]), float(b[2]), float(b[3]))))
        # left-to-right
        hits.sort(key=lambda h: (h.bbox[0] + h.bbox[2]) / 2)
        return hits


# Process-global default detector (shared across worker threads; Engine.run is lock-serialised)
_default_detector: FaceDetector | None = None
_default_lock = threading.Lock()


def set_default_engine(engine, model_name="yoloface_8n"):
    global _default_detector
    with _default_lock:
        _default_detector = FaceDetector(engine, model_name)


def detect_faces(img_bgr, opts: DetectOpts | None = None, engine=None) -> list[FaceHit]:
    """Detect faces. Uses process-global detector or a one-shot engine= override."""
    opts = opts or DetectOpts()
    if engine is not None:
        return FaceDetector(engine, "yoloface_8n", opts.detector_size).detect(img_bgr, opts)
    det = _default_detector
    if det is None:
        raise RuntimeError("Face detector not initialised — call set_default_engine or pass engine=")
    return det.detect(img_bgr, opts)


def load_image(path) -> np.ndarray:
    """Decode at FULL resolution (EXIF orientation applied by OpenCV). Never downscales."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def detect_photo(img, opts: DetectOpts | None = None, engine=None) -> list[FaceHit]:
    return detect_faces(img, opts, engine=engine)
