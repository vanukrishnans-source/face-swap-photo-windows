"""Model manifest + SHA-256 verified store for Face Swap Photo.

Lookup order for every model file:
  1. %LOCALAPPDATA%\\FaceSwapPhoto\\models          (this app's own folder)
  2. %LOCALAPPDATA%\\FaceFusionStudio\\models       (shared with Face Fusion Studio — used in place, never copied)
  3. %LOCALAPPDATA%\\GifFaceSwap\\models            (shared with GIF Face Swap — used in place)
A shared file is accepted only when its size matches and its SHA-256 matches (a FaceFusionStudio ".ok" marker
with the right digest is trusted, otherwise the file is hashed once and the result remembered here).

Downloads (only for files not found above) come from vanukrishnans-source model releases first, then the
public FaceFusion assets / Hugging Face mirror:
  * arcface / inswapper / GPEN  → vanukrishnans-source/collage-video  models-v1
  * yoloface_8n                → vanukrishnans-source/face-swap-photo-windows  models-v1

InsightFace weights (ArcFace, inswapper): personal / non-commercial only. YOLO Face (derronqi): GPL-3.0.
GPEN-BFR (Alibaba DAMO): research / personal.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

VK_COLLAGE = "https://github.com/vanukrishnans-source/collage-video/releases/download/models-v1/"
VK_PHOTO = "https://github.com/vanukrishnans-source/face-swap-photo-windows/releases/download/models-v1/"
FF_GH = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/"
FF_HF = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/"


@dataclass(frozen=True)
class ModelSpec:
    file: str
    label: str
    bytes: int
    sha256: str
    primary: str = VK_COLLAGE

    @property
    def urls(self) -> list[str]:
        return [self.primary + self.file, FF_GH + self.file, FF_HF + self.file]


YOLOFACE = ModelSpec("yoloface_8n.onnx", "Face finder (YOLO Face 8n)", 12_659_761,
                     "821cdbb1e65fbbabdde7dd0933f754797a343e56fd962729c61ffcefcd135929", VK_PHOTO)
ARCFACE = ModelSpec("arcface_w600k_r50.onnx", "Face identity (ArcFace)", 174_388_474,
                    "f1f79dc3b0b79a69f94799af1fffebff09fbd78fd96a275fd8f0cbbea23270d1")
SWAPPER = ModelSpec("inswapper_128_fp16.onnx", "Face swap (inswapper 128 fp16)", 277_680_829,
                    "c4eccca86ad177586c85c28bf1a64a9d9ed237e283a15818d831f7facfd3f420")
ENHANCER_LIGHT = ModelSpec("gpen_bfr_256.onnx", "Light enhancer (GPEN 256)", 75_792_988,
                           "bad8bf0426873828df2dbf4e3b3d9ababba9da7965b8b72426569486f7ae5c25")
ENHANCER_HQ = ModelSpec("gpen_bfr_512.onnx", "HQ enhancer (GPEN 512)", 284_340_240,
                        "d5f066b9068a8b74217f9712e28e875a6144629b108a6f7355acbdb3a2832c54")

REQUIRED = [YOLOFACE, ARCFACE, SWAPPER]
REQUIRED_BYTES = sum(s.bytes for s in REQUIRED)
ENHANCERS = {"gpen256": ENHANCER_LIGHT, "gpen512": ENHANCER_HQ}
ALL = REQUIRED + [ENHANCER_LIGHT, ENHANCER_HQ]


def _local_appdata() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))


def default_models_dir() -> Path:
    env = os.environ.get("FSP_MODELS")
    if env:
        return Path(env)
    return _local_appdata() / "FaceSwapPhoto" / "models"


def shared_model_dirs() -> list[Path]:
    if os.environ.get("FSP_NO_SHARED", "").strip() in ("1", "true", "yes"):
        return []
    base = _local_appdata()
    out = [base / "FaceFusionStudio" / "models", base / "GifFaceSwap" / "models"]
    extra = os.environ.get("FSP_SHARED_MODELS")
    if extra:
        out = [Path(p) for p in extra.split(os.pathsep) if p] + out
    return out


class ModelStore:
    """Resolves models across own + shared folders; downloads missing ones with resume + SHA-256 check."""

    def __init__(self, dir: Path | None = None, shared: Optional[list] = None):
        self.dir = Path(dir or default_models_dir())
        self.dir.mkdir(parents=True, exist_ok=True)
        self.shared = [Path(p) for p in (shared if shared is not None else shared_model_dirs())]
        self._resolved: dict[str, Path] = {}
        self._verified_path = self.dir / "shared_verified.json"
        try:
            self._verified = json.loads(self._verified_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            self._verified = {}

    # ------------------------------------------------------------------ lookup
    def _own(self, s: ModelSpec) -> Path:
        return self.dir / s.file

    def _own_ok(self, s: ModelSpec) -> bool:
        f, m = self._own(s), self.dir / (s.file + ".ok")
        try:
            return f.is_file() and f.stat().st_size == s.bytes and m.is_file() and \
                m.read_text(encoding="utf-8").strip() == s.sha256
        except OSError:
            return False

    def _shared_ok(self, s: ModelSpec, folder: Path, allow_hash: bool) -> bool:
        f = folder / s.file
        try:
            if not f.is_file() or f.stat().st_size != s.bytes:
                return False
            marker = folder / (s.file + ".ok")
            if marker.is_file() and marker.read_text(encoding="utf-8").strip() == s.sha256:
                return True
            key = str(f.resolve())
            st = f.stat()
            rec = self._verified.get(key)
            if rec and rec.get("sha") == s.sha256 and rec.get("mtime") == int(st.st_mtime):
                return True
            if not allow_hash:
                return False
            if self._sha256(f, s.bytes, None, s.file, None) != s.sha256:
                return False
            self._verified[key] = {"sha": s.sha256, "mtime": int(st.st_mtime)}
            try:
                self._verified_path.write_text(json.dumps(self._verified, indent=1), encoding="utf-8")
            except OSError:
                pass
            return True
        except OSError:
            return False

    def locate(self, s: ModelSpec, allow_hash: bool = False) -> Optional[Path]:
        p = self._resolved.get(s.file)
        if p is not None and p.is_file():
            return p
        if self._own_ok(s):
            p = self._own(s)
        else:
            p = None
            for d in self.shared:
                if self._shared_ok(s, d, allow_hash):
                    p = d / s.file
                    break
        if p is not None:
            self._resolved[s.file] = p
        return p

    def path(self, s: ModelSpec) -> Path:
        return self.locate(s) or self._own(s)

    def is_installed(self, s: ModelSpec) -> bool:
        return self.locate(s) is not None

    def source_of(self, s: ModelSpec) -> str:
        p = self.locate(s)
        if p is None:
            return "missing"
        return "this app" if p.parent == self.dir else f"shared: {p.parent}"

    def scan_shared(self, specs: Iterable[ModelSpec] = ALL, progress=None) -> list[str]:
        """Hash-verify shared copies that have no marker (one-time). Returns files found."""
        got = []
        for s in specs:
            if self.locate(s, allow_hash=True) is not None:
                got.append(s.file)
        return got

    def missing(self, specs: Iterable[ModelSpec] = REQUIRED) -> list[ModelSpec]:
        return [s for s in specs if not self.is_installed(s)]

    def all_installed(self, specs: Iterable[ModelSpec] = REQUIRED) -> bool:
        return not self.missing(specs)

    # ------------------------------------------------------------------ download
    def ensure(self, specs: Iterable[ModelSpec], progress: Optional[Callable] = None,
               cancelled: Optional[Callable[[], bool]] = None) -> None:
        for s in specs:
            if self.locate(s, allow_hash=True) is not None:
                continue
            self._download(s, progress, cancelled)

    def _download(self, s, progress, cancelled):
        part = self.dir / (s.file + ".part")
        last_err: Exception | None = None
        for url in s.urls:
            have = part.stat().st_size if part.is_file() else 0
            if have > s.bytes:
                part.unlink(); have = 0
            try:
                self._fetch(url, part, have, s.bytes, s.file, progress, cancelled)
                break
            except Exception as e:  # noqa: BLE001
                if str(e) == "cancelled":
                    raise
                last_err = e
        else:
            raise RuntimeError(f"Failed to download {s.file}: {last_err}")
        if progress:
            progress(s.file, s.bytes, s.bytes, 0.0, True)
        digest = self._sha256(part, s.bytes, progress, s.file, cancelled)
        if digest != s.sha256:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch for {s.file}: got {digest}, expected {s.sha256}")
        dest = self._own(s)
        part.replace(dest)
        (self.dir / (s.file + ".ok")).write_text(digest + "\n", encoding="utf-8")
        self._resolved[s.file] = dest

    def _fetch(self, url, part, have, total, label, progress, cancelled):
        headers = {"User-Agent": "FaceSwapPhoto-Windows/1.0"}
        if have > 0:
            headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(url, headers=headers)
        t0 = time.perf_counter()
        got = have
        with urllib.request.urlopen(req, timeout=60) as resp:
            mode = "ab" if have and resp.status == 206 else "wb"
            if mode == "wb":
                got = 0; have = 0
            with open(part, mode) as fh:
                while True:
                    if cancelled and cancelled():
                        raise RuntimeError("cancelled")
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    fh.write(chunk)
                    got += len(chunk)
                    if progress:
                        progress(label, got, total, (got - have) / max(time.perf_counter() - t0, 1e-3), False)
        if got != total:
            raise RuntimeError(f"Incomplete download of {label}: {got}/{total}")

    @staticmethod
    def _sha256(path, expected_size, progress, label, cancelled):
        h = hashlib.sha256(); done = 0
        with open(path, "rb") as fh:
            while True:
                if cancelled and cancelled():
                    raise RuntimeError("cancelled")
                chunk = fh.read(1024 * 1024 * 4)
                if not chunk:
                    break
                h.update(chunk); done += len(chunk)
                if progress:
                    progress(label, done, expected_size, 0.0, True)
        return h.hexdigest()
