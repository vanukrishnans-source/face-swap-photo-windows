"""Photo-only face swap pipeline (no video code at all).

Everything is computed at the target photo's FULL original resolution: the photo is never downscaled.
Detection runs on a 640 px letterboxed copy (that is how YOLO Face works), then, for big photos, each
face is re-detected on a tight crop so the 5 landmarks are accurate at full resolution.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import core, detect
from .detect import DetectOpts, FaceHit
from .engine import Engine
from .models import ENHANCERS, ModelStore

log = logging.getLogger("fsp")

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".jfif"}
ENHANCE_LABEL = {None: "Off", "gpen256": "Light (GPEN 256)", "gpen512": "HQ (GPEN 512)"}


class Cancelled(Exception):
    pass


@dataclass
class Photo:
    path: str
    img: np.ndarray                     # full-resolution BGR uint8
    faces: list = field(default_factory=list)   # FaceHit, left -> right
    latents: list = field(default_factory=list)  # one per face (only filled for source photos)

    @property
    def size(self):
        return self.img.shape[1], self.img.shape[0]


@dataclass
class SwapOptions:
    enhance: Optional[str] = "gpen512"   # None | gpen256 | gpen512
    enhance_blend: float = 0.8
    color_match: bool = True             # face-only Reinhard LAB (never touches body / skin outside the face)
    seamless: bool = False               # Poisson edge (slower)
    detail: bool = True                  # sharpen + grain-match big faces so they aren't soft/plastic
    device: str = "auto"


def face_thumb(img, hit: FaceHit, size=160, mark=False):
    x0, y0, x1, y1 = hit.bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = max(x1 - x0, y1 - y0) * 0.75
    a, b = int(max(0, cx - r)), int(max(0, cy - r))
    c, d = int(min(img.shape[1], cx + r)), int(min(img.shape[0], cy + r))
    if d <= b or c <= a:
        return np.zeros((size, size, 3), np.uint8)
    crop = img[b:d, a:c]
    if mark:  # outline this person's face (couple photos: the neighbour's face is often in the crop too)
        crop = crop.copy()
        t = max(2, int(max(crop.shape[:2]) / 70))
        cv2.rectangle(crop, (int(x0 - a), int(y0 - b)), (int(x1 - a), int(y1 - b)), (169, 184, 0), t, cv2.LINE_AA)
    hh, ww = crop.shape[:2]
    s = size / max(hh, ww)
    crop = cv2.resize(crop, (max(1, int(ww * s)), max(1, int(hh * s))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 22, np.uint8)
    oy, ox = (size - crop.shape[0]) // 2, (size - crop.shape[1]) // 2
    canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    return canvas


def default_pictures_dir() -> Path:
    if os.name == "nt":
        try:
            import ctypes
            import uuid
            fid = uuid.UUID("{33E28130-4E1E-4676-835A-98395C3BC3BB}")  # FOLDERID_Pictures
            guid = (ctypes.c_byte * 16).from_buffer_copy(fid.bytes_le)
            pth = ctypes.c_wchar_p()
            if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(pth)) == 0:
                base = Path(pth.value); ctypes.windll.ole32.CoTaskMemFree(pth)
                return base / "FaceSwapPhoto"
        except Exception:  # noqa: BLE001
            pass
    return Path.home() / "Pictures" / "FaceSwapPhoto"


def save_image(img, path, fmt=None, jpeg_quality=97) -> Path:
    """Save at the image's full resolution. PNG = lossless; JPEG = quality 97, 4:4:4 chroma (no colour smear)."""
    path = Path(path)
    fmt = (fmt or path.suffix.lstrip(".") or "jpg").lower()
    if fmt in ("jpeg", "jpg"):
        ext = ".jpg"
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality), int(cv2.IMWRITE_JPEG_OPTIMIZE), 1]
        if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR") and hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444"):
            params += [int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444)]
    elif fmt == "png":
        ext = ".png"; params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    else:
        raise ValueError(f"unsupported format {fmt}")
    if path.suffix.lower() not in ((".jpg", ".jpeg") if ext == ".jpg" else (".png",)):
        path = path.with_suffix(ext)
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError(f"{ext} encode failed")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    buf.tofile(str(tmp))
    os.replace(tmp, path)
    return path


def default_mapping(n_tgt: int, n_src: int, tgt_faces=None) -> list:
    """Target faces are ordered left -> right; source faces too (across photos in the order added).

    * 2+ source faces: target i gets source i (leftmost person gets the leftmost face).
    * 1 source face: it goes on the biggest target face only (tap a face to add more).
    """
    if n_tgt <= 0 or n_src <= 0:
        return [-1] * max(n_tgt, 0)
    if n_src == 1:
        m = [-1] * n_tgt
        if tgt_faces:
            big = max(range(n_tgt), key=lambda i: (tgt_faces[i].bbox[2] - tgt_faces[i].bbox[0]) *
                      (tgt_faces[i].bbox[3] - tgt_faces[i].bbox[1]))
        else:
            big = 0
        m[big] = 0
        return m
    return [i if i < n_src else -1 for i in range(n_tgt)]


def flip_mapping(mapping: list) -> list:
    """Mirror who-becomes-who left <-> right (for a couple: swap the two faces)."""
    return list(reversed(mapping))


class PhotoSwapper:
    def __init__(self, store: ModelStore, device: str = "auto"):
        self.store = store
        self.device = device
        self.engine: Optional[Engine] = None
        self.opts = DetectOpts(min_confidence=0.5, min_face_frac=0.02, max_faces=12)

    # ------------------------------------------------------------- engine
    def get_engine(self, device=None) -> Engine:
        device = device or self.device
        if self.engine is None or self.engine.device != device:
            if self.engine is not None:
                self.engine.close()
            self.engine = Engine(self.store, device)
            self.engine.prepare(None)
            detect.set_default_engine(self.engine)
            self.device = device
        return self.engine

    def close(self):
        if self.engine is not None:
            self.engine.close(); self.engine = None

    # ------------------------------------------------------------- faces
    def find_faces(self, img) -> list:
        eng = self.get_engine()
        hits = detect.detect_faces(img, self.opts, engine=eng)
        h, w = img.shape[:2]
        if max(h, w) > 1600:
            hits = [self._refine(eng, img, hit) for hit in hits]
        hits.sort(key=lambda x: (x.bbox[0] + x.bbox[2]) / 2)
        return hits

    def _refine(self, eng, img, hit: FaceHit) -> FaceHit:
        """Re-detect on a tight crop so landmarks are precise at full resolution."""
        try:
            x0, y0, x1, y1 = hit.bbox
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            r = max(x1 - x0, y1 - y0) * 1.1
            h, w = img.shape[:2]
            a, b = int(max(0, cx - r)), int(max(0, cy - r))
            c, d = int(min(w, cx + r)), int(min(h, cy + r))
            if c - a < 64 or d - b < 64 or max(c - a, d - b) < 700:
                return hit
            crop = img[b:d, a:c]
            sub = detect.detect_faces(crop, DetectOpts(min_confidence=0.35, min_face_frac=0.1, max_faces=4), engine=eng)
            if not sub:
                return hit
            ccx, ccy = cx - a, cy - b
            best = min(sub, key=lambda s: ((s.bbox[0] + s.bbox[2]) / 2 - ccx) ** 2 + ((s.bbox[1] + s.bbox[3]) / 2 - ccy) ** 2)
            bx = (best.bbox[0] + best.bbox[2]) / 2 - ccx; by = (best.bbox[1] + best.bbox[3]) / 2 - ccy
            if (bx * bx + by * by) ** 0.5 > 0.35 * max(x1 - x0, y1 - y0):
                return hit
            kps = best.kps + np.array([a, b], np.float32)
            bb = (best.bbox[0] + a, best.bbox[1] + b, best.bbox[2] + a, best.bbox[3] + b)
            return FaceHit(kps=kps, confidence=max(hit.confidence, best.confidence), bbox=tuple(float(v) for v in bb))
        except Exception as e:  # noqa: BLE001
            log.warning("refine failed: %s", e)
            return hit

    def load_target(self, path) -> Photo:
        img = detect.load_image(path)
        return Photo(str(path), img, self.find_faces(img))

    def load_source(self, path) -> Photo:
        img = detect.load_image(path)
        ph = Photo(str(path), img, self.find_faces(img))
        eng = self.get_engine()
        ph.latents = [core.latent_for(eng, core.embedding(eng, img, f.kps)) for f in ph.faces]
        return ph

    # ------------------------------------------------------------- swap
    def swap(self, target: Photo, source_latents: list, mapping: list, opts: SwapOptions,
             progress=None, cancel=None) -> dict:
        """mapping[i] = index into source_latents for target face i, or -1 to keep that face."""
        if not target.faces:
            raise ValueError("No face found in the target photo. Try a clearer, front-facing picture.")
        if not source_latents:
            raise ValueError("No face found in the faces photo(s).")
        if opts.enhance and not self.store.is_installed(ENHANCERS[opts.enhance]):
            raise ValueError(f"The {ENHANCE_LABEL[opts.enhance]} face enhancer isn't downloaded yet.")
        t0 = time.perf_counter()
        eng = self.get_engine(opts.device)
        if opts.enhance:
            eng.prepare(opts.enhance)
        pairs = [(i, target.faces[i].kps.astype(np.float32), source_latents[s])
                 for i, s in enumerate(mapping) if 0 <= s < len(source_latents) and i < len(target.faces)]
        if not pairs:
            raise ValueError("No faces are set to swap. Tap a face under 'Who becomes who' to pick one.")
        out = target.img.copy()
        for k, (ti, kps, lat) in enumerate(pairs):
            others = [f.kps for j, f in enumerate(target.faces) if j != ti]
            if cancel and cancel():
                raise Cancelled()
            if progress:
                progress(dict(done=k, total=len(pairs), detail=f"Swapping face {k + 1} of {len(pairs)}…"))
            ref = target.img if opts.detail else None
            core.swap_face(eng, out, kps, lat, inplace=True, color_match=opts.color_match, seamless=opts.seamless,
                           detail_ref=None if opts.enhance else ref, exclude=others)
            if opts.enhance:
                if progress:
                    progress(dict(done=k + 0.5, total=len(pairs), detail=f"Enhancing face {k + 1} of {len(pairs)}…"))
                core.enhance(eng, out, kps, opts.enhance, opts.enhance_blend, inplace=True, color_match=opts.color_match,
                             detail_ref=ref, exclude=others)
        if progress:
            progress(dict(done=len(pairs), total=len(pairs), detail="Done"))
        assert out.shape == target.img.shape, "output must keep the original resolution"
        info = eng.info
        return dict(img=out, W=out.shape[1], H=out.shape[0], faces=len(pairs), secs=round(time.perf_counter() - t0, 2),
                    device=info.label(), device_active=info.active, enhance=ENHANCE_LABEL.get(opts.enhance),
                    mapping=list(mapping))


def before_after(before, after, max_w=3000, gap=16, labels=True):
    """Side-by-side comparison image for sharing/report (downscaled to max_w total width)."""
    h, w = before.shape[:2]
    s = min(1.0, (max_w - gap) / (2.0 * w))
    nw, nh = int(w * s), int(h * s)
    b = cv2.resize(before, (nw, nh), interpolation=cv2.INTER_AREA)
    a = cv2.resize(after, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((nh, nw * 2 + gap, 3), 255, np.uint8)
    canvas[:, :nw] = b; canvas[:, nw + gap:] = a
    if labels:
        fs = max(0.8, nw / 900); th = max(2, int(fs * 2))
        for x, text in ((12, "BEFORE"), (nw + gap + 12, "AFTER")):
            (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            cv2.rectangle(canvas, (x - 6, 10), (x + tw + 8, 22 + tht), (20, 20, 20), -1)
            cv2.putText(canvas, text, (x, 16 + tht), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
    return canvas
