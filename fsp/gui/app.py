"""Face Swap Photo — single fixed screen, no scrolling (Ally X: 1280x720 logical at 150 %).

Layout (one QMainWindow, no QScrollArea anywhere):
  top bar      : title · device chip · status · Options · Full screen
  left column  : 1 · photo to change     2 · face photo(s) (thumbnail grid, max 8 faces)
  right column : before/after preview (Before | After | Side by side | Face zoom)
                 3 · who becomes who  (one tap-card per person + Flip)
  bottom bar   : Enhance face toggle · JPEG|PNG · progress · SWAP · SAVE · Open folder
Rarely used settings live in a fixed-size Options popup. Heavy work runs in QThreads; ONNX runs on the
Engine's own thread with a DirectML child-process probe + CPU fallback.
"""
from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QSettings, QSize, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QFontMetrics, QIcon, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractScrollArea, QApplication, QButtonGroup, QCheckBox, QDialog, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton, QSizePolicy, QStackedWidget,
    QVBoxLayout, QWidget,
)

from .. import __version__, crashlog
from ..models import ALL, ENHANCER_HQ, ENHANCER_LIGHT, ENHANCERS, REQUIRED, ModelStore
from ..photo import (
    ENHANCE_LABEL, IMAGE_EXT, Cancelled, PhotoSwapper, SwapOptions, default_mapping, default_pictures_dir,
    face_thumb, flip_mapping, save_image,
)
from .theme import QSS

log = logging.getLogger("fsp")

TEAL = (169, 184, 0)
ORANGE = (61, 138, 255)
GREY = (140, 140, 140)
LETTERS = "ABCDEFGH"
MAX_SRC_FACES = 8
MAX_MAP_CARDS = 6
PAGE_SETUP, PAGE_MAIN = 0, 1
IMG_FILTER = "Images (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff *.jfif)"


# ---------------------------------------------------------------------------- helpers
def _dpr():
    app = QApplication.instance()
    return app.devicePixelRatio() if app else 1.0


def pix(bgr, w, h):
    if bgr is None or getattr(bgr, "size", 0) == 0 or w < 2 or h < 2:
        return QPixmap()
    dpr = _dpr()
    W, H = int(w * dpr), int(h * dpr)
    ih, iw = bgr.shape[:2]
    s = min(W / iw, H / ih)
    img = cv2.resize(bgr, (max(1, int(iw * s)), max(1, int(ih * s))),
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    p = QPixmap.fromImage(QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                                 QImage.Format.Format_RGB888).copy())
    p.setDevicePixelRatio(dpr)
    return p


def shrink(img, max_side):
    h, w = img.shape[:2]
    s = min(1.0, max_side / float(max(h, w)))
    if s >= 1.0:
        return img.copy(), 1.0
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA), s


def draw_boxes(img, faces, scale, labels, colors):
    out = img.copy()
    th = max(2, out.shape[1] // 320)
    fs = max(0.6, out.shape[1] / 1100)
    for i, f in enumerate(faces):
        x0, y0, x1, y1 = [int(v * scale) for v in f.bbox]
        c = colors[i]
        cv2.rectangle(out, (x0, y0), (x1, y1), c, th, cv2.LINE_AA)
        text = labels[i]
        (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        y_top = max(0, y0 - tht - 12)
        cv2.rectangle(out, (x0, y_top), (x0 + tw + 12, y_top + tht + 12), c, -1)
        cv2.putText(out, text, (x0 + 6, y_top + tht + 5), cv2.FONT_HERSHEY_SIMPLEX, fs, (16, 16, 16), th, cv2.LINE_AA)
    return out


def tag_thumb(thumb, text, color=TEAL):
    t = thumb.copy()
    s = t.shape[0]
    fs = s / 110.0; th = max(1, int(s / 60))
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    cv2.rectangle(t, (0, 0), (tw + 10, tht + 10), color, -1)
    cv2.putText(t, text, (5, tht + 5), cv2.FONT_HERSHEY_SIMPLEX, fs, (16, 16, 16), th, cv2.LINE_AA)
    return t


def elide(widget, text, width):
    return QFontMetrics(widget.font()).elidedText(text, Qt.TextElideMode.ElideMiddle, max(20, width))


class Worker(QThread):
    progressed = Signal(object)
    done = Signal(object)
    failed = Signal(str, str)

    def __init__(self, fn, parent=None):
        super().__init__(parent); self.fn = fn; self.cancelled = False

    def run(self):
        try:
            self.done.emit(self.fn(lambda: self.cancelled, self.progressed.emit))
        except Cancelled:
            self.failed.emit("cancelled", "")
        except Exception as e:  # noqa: BLE001
            if self.cancelled or str(e) == "cancelled":
                self.failed.emit("cancelled", ""); return
            tb = traceback.format_exc()
            log.error("worker failed: %s\n%s", e, tb)
            try:
                crashlog.record_current(where="worker")
            except Exception:  # noqa: BLE001
                pass
            self.failed.emit(str(e) or type(e).__name__, tb)


def button(text, kind=None, min_w=0, h=52):
    b = QPushButton(text)
    if kind: b.setObjectName(kind)
    if min_w: b.setMinimumWidth(min_w)
    b.setMinimumHeight(h)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


def label(text="", kind=None, wrap=False):
    l = QLabel(text)
    if kind: l.setObjectName(kind)
    l.setWordWrap(wrap)
    return l


def card():
    f = QFrame(); f.setObjectName("card"); return f


class Segmented(QWidget):
    changed = Signal(object)

    def __init__(self, items, h=44):
        super().__init__()
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(6)
        self.group = QButtonGroup(self); self.group.setExclusive(True); self.buttons = {}
        for text, value in items:
            b = button(text, "seg", 0, h); b.setCheckable(True)
            self.group.addButton(b); lay.addWidget(b, 1); self.buttons[value] = b
            b.clicked.connect(lambda _=False, v=value: self.changed.emit(v))
        if items:
            self.buttons[items[0][1]].setChecked(True)

    def set(self, value):
        b = self.buttons.get(value)
        if b: b.setChecked(True)

    def value(self):
        for v, b in self.buttons.items():
            if b.isChecked(): return v
        return None


class ClickLabel(QLabel):
    clicked = Signal()
    dropped = Signal(list)

    def __init__(self, text="", kind=None):
        super().__init__(text)
        if kind: self.setObjectName(kind)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(40, 40)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setWordWrap(True)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton: self.clicked.emit()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dropEvent(self, e):
        self.dropped.emit([u.toLocalFile() for u in e.mimeData().urls()])


# ---------------------------------------------------------------------------- popups
class PickFaceDialog(QDialog):
    """'Person N becomes…' — big face buttons, fixed size, no scrolling."""

    def __init__(self, parent, person_idx, target_thumb, src_thumbs, current):
        super().__init__(parent)
        self.setWindowTitle("Who becomes who")
        self.setModal(True)
        self.choice = None
        lay = QVBoxLayout(self); lay.setContentsMargins(20, 16, 20, 16); lay.setSpacing(12)
        top = QHBoxLayout()
        tl = QLabel(); tl.setPixmap(pix(target_thumb, 88, 88)); tl.setFixedSize(92, 92); top.addWidget(tl)
        top.addWidget(label(f"Person {person_idx + 1} becomes…", "title"), 1)
        lay.addLayout(top)
        grid = QGridLayout(); grid.setSpacing(10)
        cols = 5
        items = [(i, tag_thumb(t, LETTERS[i])) for i, t in enumerate(src_thumbs)] + [(-1, None)]
        for n, (i, th) in enumerate(items):
            if i >= 0:
                b = button("", "pick", 0, 150)
                b.setIcon(QIcon(pix(th, 120, 120))); b.setIconSize(QSize(120, 120))
                b.setToolTip(f"Face {LETTERS[i]}")
            else:
                b = button("Keep\noriginal", "pick", 0, 150)
            b.setCheckable(True); b.setChecked(i == current)
            b.setMinimumWidth(150); b.setMaximumWidth(190)
            b.clicked.connect(lambda _=False, v=i: self._pick(v))
            grid.addWidget(b, n // cols, n % cols)
        lay.addLayout(grid)
        row = QHBoxLayout(); row.addStretch(1)
        c = button("Cancel", None, 160); c.clicked.connect(self.reject); row.addWidget(c)
        lay.addLayout(row)

    def _pick(self, v):
        self.choice = v; self.accept()


class OptionsDialog(QDialog):
    def __init__(self, win: "MainWindow"):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Options")
        self.setModal(True)
        self.setFixedSize(min(1180, max(900, win.width() - 60)), min(600, max(520, win.height() - 40)))
        o = win.opt
        outer = QVBoxLayout(self); outer.setContentsMargins(20, 14, 20, 14); outer.setSpacing(8)
        head = QHBoxLayout(); head.addWidget(label("Options", "title")); head.addStretch(1)
        close = button("Close", "primary", 160); close.clicked.connect(self.accept); head.addWidget(close)
        outer.addLayout(head)
        cols = QHBoxLayout(); cols.setSpacing(24); outer.addLayout(cols, 1)
        left = QVBoxLayout(); left.setSpacing(6); right = QVBoxLayout(); right.setSpacing(6)
        cols.addLayout(left, 1); cols.addLayout(right, 1)

        def seg(col, title, items, key, cur):
            col.addWidget(label(title, "section"))
            s = Segmented(items); s.set(cur)
            s.changed.connect(lambda v, k=key: win.set_opt(k, v)); col.addWidget(s)
            return s

        seg(left, "Face enhancer quality", [("Light · GPEN 256", "gpen256"), ("HQ · GPEN 512", "gpen512")],
            "enhancer", o["enhancer"])
        seg(left, "Enhance strength", [("Subtle", 0.55), ("Normal", 0.8), ("Strong", 1.0)], "blend", o["blend"])
        seg(left, "Sharpen + grain match (big faces)", [("On", True), ("Off", False)], "detail", o["detail"])
        seg(left, "Face colour match (face only)", [("On", True), ("Off", False)], "color_match", o["color_match"])
        seg(left, "Smooth edge (Poisson, slower)", [("Off", False), ("On", True)], "seamless", o["seamless"])

        seg(right, "JPEG quality", [("95", 95), ("97 (default)", 97), ("100", 100)], "jpeg_q", o["jpeg_q"])
        seg(right, "Processor", [("Auto", "auto"), ("GPU", "dml"), ("CPU", "cpu")], "device", o["device"])
        r = QHBoxLayout()
        self.btn_gpu = button("Retry GPU", None, 160); self.btn_gpu.clicked.connect(win.retry_gpu); r.addWidget(self.btn_gpu)
        self.btn_dl = button("Download enhancers", None, 220); self.btn_dl.clicked.connect(self._dl); r.addWidget(self.btn_dl)
        r.addStretch(1); right.addLayout(r)
        right.addWidget(label("Save folder", "section"))
        r2 = QHBoxLayout()
        self.out_lbl = label("", "hint"); self.out_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        r2.addWidget(self.out_lbl, 1)
        ch = button("Change…", None, 140); ch.clicked.connect(self._change_out); r2.addWidget(ch)
        right.addLayout(r2)
        self.models_lbl = label("", "hint", True); right.addWidget(self.models_lbl)
        right.addStretch(1)
        about = label(
            f"Face Swap Photo {__version__} · YOLO Face + ArcFace + inswapper 128 + optional GPEN · "
            "photo only, saved at full original resolution. InsightFace models: personal / non-commercial use. "
            "Crash log: %LOCALAPPDATA%\\FaceSwapPhoto\\crash.log", "hint", True)
        outer.addWidget(about)
        self.refresh()

    def refresh(self):
        w = self.win
        self.out_lbl.setText(elide(self.out_lbl, w.opt["out_dir"], 380))
        miss = [s for s in (ENHANCER_LIGHT, ENHANCER_HQ) if not w.store.is_installed(s)]
        self.btn_dl.setEnabled(bool(miss))
        self.btn_dl.setText("Download enhancers" if miss else "Enhancers ready")
        lines = []
        for s in ALL:
            src = w.store.source_of(s)
            lines.append(f"{s.file}: {'missing' if src == 'missing' else ('here' if src == 'this app' else 'shared')}")
        self.models_lbl.setText(" · ".join(lines))

    def _dl(self):
        self.accept()
        self.win.download_models([s for s in (ENHANCER_LIGHT, ENHANCER_HQ) if not self.win.store.is_installed(s)])

    def _change_out(self):
        d = QFileDialog.getExistingDirectory(self, "Save folder", self.win.opt["out_dir"])
        if d:
            self.win.set_opt("out_dir", d); self.refresh()


# ---------------------------------------------------------------------------- main window
class MainWindow(QMainWindow):
    def __init__(self, store: ModelStore, device="auto"):
        super().__init__()
        self.setWindowTitle(f"Face Swap Photo {__version__}")
        self.cfg = QSettings("vanu", "FaceSwapPhoto")
        self.store = store

        def b(key, default):
            v = self.cfg.value(key, default)
            return v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes")

        dev = str(self.cfg.value("device", device))
        self.opt = dict(
            enhance=b("enhance", True),
            enhancer=str(self.cfg.value("enhancer", "gpen512")),
            blend=float(self.cfg.value("blend", 0.8)),
            detail=b("detail", True),
            color_match=b("color_match", True),
            seamless=b("seamless", False),
            fmt=str(self.cfg.value("fmt", "jpg")),
            jpeg_q=int(self.cfg.value("jpeg_q", 97)),
            device=device if device != "auto" else (dev if dev in ("auto", "dml", "cpu") else "auto"),
            out_dir=str(self.cfg.value("out_dir", str(default_pictures_dir()))),
        )
        if self.opt["enhancer"] not in ENHANCERS: self.opt["enhancer"] = "gpen512"
        self.sw = PhotoSwapper(store, self.opt["device"])
        self.target = None          # Photo
        self.sources = []           # [Photo]
        self.pool = []              # [(photo_idx, face_idx)]
        self.mapping = []
        self.result = None
        self.result_key = None
        self.saved_path = None
        self.worker = None
        self.view = "before"
        self._disp_cache = {}
        self._queue = []

        root = QWidget(); rl = QVBoxLayout(root); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(0)
        rl.addWidget(self._topbar())
        self.stack = QStackedWidget(); rl.addWidget(self.stack, 1)
        self.stack.addWidget(self._page_setup())
        self.stack.addWidget(self._page_main())
        self.setCentralWidget(root)
        self.setAcceptDrops(True)
        self.setMinimumSize(1000, 580)
        QShortcut(QKeySequence(Qt.Key.Key_Return), self, activated=lambda: self.btn_swap.isEnabled() and self.swap())
        QShortcut(QKeySequence(Qt.Key.Key_F11), self, activated=self.toggle_full)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=lambda: self.btn_save.isEnabled() and self.save())
        self.refresh_setup()
        if self.store.all_installed(REQUIRED):
            self.stack.setCurrentIndex(PAGE_MAIN)
            QTimer.singleShot(150, self.warm_engine)
        else:
            self.stack.setCurrentIndex(PAGE_SETUP)
        self.update_ui()

    # ------------------------------------------------------------------ chrome
    def _topbar(self):
        bar = QFrame(); bar.setObjectName("topbar"); bar.setFixedHeight(60)
        lay = QHBoxLayout(bar); lay.setContentsMargins(16, 6, 10, 6); lay.setSpacing(10)
        lay.addWidget(label("Face Swap Photo", "apptitle"))
        self.chip = label("Starting…", "chip"); lay.addWidget(self.chip)
        self.status = label("", "status"); self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        lay.addWidget(self.status, 1)
        self.btn_opts = button("Options", None, 130, 48); self.btn_opts.clicked.connect(self.open_options); lay.addWidget(self.btn_opts)
        self.btn_full = button("Full screen", None, 150, 48); self.btn_full.clicked.connect(self.toggle_full); lay.addWidget(self.btn_full)
        return bar

    def set_status(self, text):
        self._status_text = text
        self.status.setText(elide(self.status, text, max(100, self.status.width() - 8)))
        self.status.setToolTip(text)

    def set_chip(self):
        eng = self.sw.engine
        if eng is None:
            self.chip.setText("Ready"); self.chip.setProperty("state", "")
        else:
            i = eng.info
            if i.active == "DirectML":
                self.chip.setText("GPU · DirectML"); self.chip.setProperty("state", "")
            elif i.fell_back or i.fallback_reason:
                self.chip.setText("CPU · GPU failed"); self.chip.setProperty("state", "cpu")
                self.chip.setToolTip(i.fallback_reason or "DirectML unavailable — using CPU")
            else:
                self.chip.setText("CPU"); self.chip.setProperty("state", "cpu")
        self.chip.style().unpolish(self.chip); self.chip.style().polish(self.chip)

    def toggle_full(self):
        if self.isFullScreen():
            self.showMaximized(); self.btn_full.setText("Full screen")
        else:
            self.showFullScreen(); self.btn_full.setText("Exit full")

    # ------------------------------------------------------------------ setup page
    def _page_setup(self):
        w = QWidget(); outer = QVBoxLayout(w); outer.setContentsMargins(28, 18, 28, 18); outer.setSpacing(10)
        outer.addWidget(label("One-time setup: AI models", "title"))
        outer.addWidget(label(
            "Face Swap Photo needs the face models once (they stay on this PC). Models already downloaded by "
            "Face Fusion Studio are reused automatically. InsightFace models are for personal, non-commercial use.",
            "hint", True))
        self.setup_list = label("", "status", True); outer.addWidget(self.setup_list)
        self.chk_hq = QCheckBox(f"Also get the HQ face enhancer, GPEN 512 (+{ENHANCER_HQ.bytes / 1e6:.0f} MB, recommended)")
        self.chk_hq.setChecked(True); outer.addWidget(self.chk_hq)
        self.chk_light = QCheckBox(f"Also get the light face enhancer, GPEN 256 (+{ENHANCER_LIGHT.bytes / 1e6:.0f} MB, faster)")
        outer.addWidget(self.chk_light)
        self.setup_bar = QProgressBar(); self.setup_bar.setRange(0, 1000); self.setup_bar.setTextVisible(False)
        outer.addWidget(self.setup_bar)
        self.setup_text = label("", "status"); outer.addWidget(self.setup_text)
        outer.addStretch(1)
        row = QHBoxLayout()
        self.btn_dl = button("Download models", "primary", 300, 60); self.btn_dl.clicked.connect(self._setup_download)
        row.addWidget(self.btn_dl)
        self.btn_dl_cancel = button("Cancel", "danger", 160, 60); self.btn_dl_cancel.clicked.connect(self.cancel)
        self.btn_dl_cancel.setEnabled(False); row.addWidget(self.btn_dl_cancel)
        row.addStretch(1)
        self.btn_setup_done = button("Continue", None, 180, 60); self.btn_setup_done.clicked.connect(self._setup_continue)
        row.addWidget(self.btn_setup_done)
        outer.addLayout(row)
        return w

    def refresh_setup(self):
        lines = []
        need = 0
        for s in ALL:
            src = self.store.source_of(s)
            req = s in REQUIRED
            if src == "missing":
                if req: need += s.bytes
                lines.append(f"{'•' if req else '◦'} {s.label}: {'needed' if req else 'optional'} ({s.bytes / 1e6:.0f} MB)")
            else:
                lines.append(f"✓ {s.label}: {'ready' if src == 'this app' else 'reused from ' + src.split(': ', 1)[-1]}")
        lines.append(f"Required download: {need / 1e6:.0f} MB" if need else "All required models are ready.")
        self.setup_list.setText("\n".join(lines))
        self.chk_hq.setVisible(not self.store.is_installed(ENHANCER_HQ))
        self.chk_light.setVisible(not self.store.is_installed(ENHANCER_LIGHT))
        busy = self.busy()
        self.btn_dl.setEnabled(not busy)
        self.btn_setup_done.setEnabled(self.store.all_installed(REQUIRED) and not busy)

    def _setup_download(self):
        want = list(self.store.missing(REQUIRED))
        if self.chk_hq.isChecked() and not self.store.is_installed(ENHANCER_HQ): want.append(ENHANCER_HQ)
        if self.chk_light.isChecked() and not self.store.is_installed(ENHANCER_LIGHT): want.append(ENHANCER_LIGHT)
        self.download_models(want)

    def _setup_continue(self):
        self.stack.setCurrentIndex(PAGE_MAIN); self.update_ui(); QTimer.singleShot(100, self.warm_engine)

    def download_models(self, specs):
        if self.busy(): return
        if not specs:
            self.refresh_setup(); return
        self.stack.setCurrentIndex(PAGE_SETUP)
        self.btn_dl_cancel.setEnabled(True)

        def work(cancelled, emit):
            def prog(f, done, total, bps, verifying):
                emit(dict(file=f, done=done, total=total, bps=bps, verifying=verifying))
            self.store.ensure(specs, progress=prog, cancelled=cancelled)
            return True

        def ok(_):
            self.btn_dl_cancel.setEnabled(False)
            self.setup_text.setText("Models ready.")
            self.refresh_setup()
            if self.store.all_installed(REQUIRED):
                self._setup_continue()

        def bad(msg, tb):
            self.btn_dl_cancel.setEnabled(False); self.refresh_setup()
            self.setup_text.setText("Download paused — tap Download models to resume." if msg == "cancelled"
                                    else f"Download failed: {msg[:200]}")

        self.run_worker(work, ok, bad, self._setup_progress, "Downloading models…")
        self.refresh_setup()

    def _setup_progress(self, d):
        done, total = d.get("done", 0), max(1, d.get("total", 1))
        self.setup_bar.setValue(int(1000 * done / total))
        tag = "Checking" if d.get("verifying") else "Downloading"
        speed = f" · {d.get('bps', 0) / 1e6:.1f} MB/s" if not d.get("verifying") and d.get("bps") else ""
        self.setup_text.setText(f"{tag} {d.get('file', '')} · {done / 1e6:.0f}/{total / 1e6:.0f} MB{speed}")

    # ------------------------------------------------------------------ main page
    def _page_main(self):
        w = QWidget(); outer = QVBoxLayout(w); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)
        body = QHBoxLayout(); body.setContentsMargins(10, 8, 10, 8); body.setSpacing(10)
        outer.addLayout(body, 1)

        # ---- left column
        left = QVBoxLayout(); left.setSpacing(8)
        c1 = card(); l1 = QVBoxLayout(c1); l1.setContentsMargins(10, 8, 10, 10); l1.setSpacing(6)
        h1 = QHBoxLayout(); h1.addWidget(label("1 · Photo to change", "section")); h1.addStretch(1)
        self.btn_target = button("Choose", None, 110, 44); self.btn_target.clicked.connect(self.pick_target); h1.addWidget(self.btn_target)
        l1.addLayout(h1)
        self.target_slot = ClickLabel("Tap to choose the photo\n(one or more people)", "slot")
        self.target_slot.clicked.connect(self.pick_target)
        self.target_slot.dropped.connect(lambda ps: ps and self.set_target(ps[0]))
        l1.addWidget(self.target_slot, 1)
        self.target_info = label("", "hint"); self.target_info.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        l1.addWidget(self.target_info)
        left.addWidget(c1, 11)

        c2 = card(); l2 = QVBoxLayout(c2); l2.setContentsMargins(10, 8, 10, 10); l2.setSpacing(6)
        h2 = QHBoxLayout(); h2.addWidget(label("2 · Faces", "section")); h2.addStretch(1)
        self.btn_add = button("Add", None, 80, 44); self.btn_add.clicked.connect(self.pick_sources); h2.addWidget(self.btn_add)
        self.btn_clear = button("Clear", None, 80, 44); self.btn_clear.clicked.connect(self.clear_sources); h2.addWidget(self.btn_clear)
        l2.addLayout(h2)
        self.src_area = QWidget(); self.src_area.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        sg = QGridLayout(self.src_area); sg.setContentsMargins(0, 0, 0, 0); sg.setSpacing(6)
        self.src_cells = []
        for i in range(MAX_SRC_FACES):
            c = ClickLabel("", "slot"); c.clicked.connect(self.pick_sources)
            c.dropped.connect(self.add_sources)
            sg.addWidget(c, i // 4, i % 4); self.src_cells.append(c)
        self.src_empty = ClickLabel("Tap to add the face photo(s)\n(e.g. one photo with both of you)", "slot")
        self.src_empty.clicked.connect(self.pick_sources); self.src_empty.dropped.connect(self.add_sources)
        sg.addWidget(self.src_empty, 0, 0, 2, 4)
        l2.addWidget(self.src_area, 1)
        self.src_info = label("", "hint"); self.src_info.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        l2.addWidget(self.src_info)
        left.addWidget(c2, 9)
        lw = QWidget(); lw.setLayout(left); lw.setFixedWidth(330)
        body.addWidget(lw)

        # ---- right column
        right = QVBoxLayout(); right.setSpacing(8)
        c3 = card(); l3 = QVBoxLayout(c3); l3.setContentsMargins(10, 8, 10, 10); l3.setSpacing(6)
        h3 = QHBoxLayout(); h3.addWidget(label("Preview", "section")); h3.addSpacing(10)
        self.seg_view = Segmented([("Before", "before"), ("After", "after"), ("Side by side", "side"), ("Face zoom", "zoom")])
        self.seg_view.changed.connect(self.set_view); h3.addWidget(self.seg_view, 1)
        l3.addLayout(h3)
        self.preview = ClickLabel("Choose a photo to change and a face photo, then tap SWAP.", "preview")
        self.preview.clicked.connect(self._preview_tap)
        self.preview.dropped.connect(self._drop_any)
        l3.addWidget(self.preview, 1)
        right.addWidget(c3, 1)

        c4 = card(); c4.setFixedHeight(128)
        l4 = QHBoxLayout(c4); l4.setContentsMargins(10, 6, 10, 6); l4.setSpacing(8)
        cap = QVBoxLayout(); cap.setSpacing(2)
        cap.addWidget(label("3 · Who\nbecomes\nwho", "section")); cap.addWidget(label("tap a person", "hint"))
        capw = QWidget(); capw.setLayout(cap); capw.setFixedWidth(118); l4.addWidget(capw)
        self.map_cards = []
        for i in range(MAX_MAP_CARDS):
            b = button("", "mapcard", 0, 108); b.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            b.setIconSize(QSize(150, 72)); b.clicked.connect(lambda _=False, k=i: self.pick_for(k))
            l4.addWidget(b, 1); self.map_cards.append(b)
        self.map_empty = label("Faces appear here after you pick both photos.", "hint")
        self.map_empty.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        l4.addWidget(self.map_empty, 6)
        self.btn_flip = button("⇄ Flip", None, 120, 96); self.btn_flip.clicked.connect(self.flip); l4.addWidget(self.btn_flip)
        right.addWidget(c4)
        body.addLayout(right, 1)

        # ---- bottom bar
        bar = QFrame(); bar.setObjectName("bar"); bar.setFixedHeight(76)
        bl = QHBoxLayout(bar); bl.setContentsMargins(12, 8, 12, 8); bl.setSpacing(10)
        self.btn_enh = button("", "toggle", 230, 56); self.btn_enh.setCheckable(True)
        self.btn_enh.setChecked(self.opt["enhance"]); self.btn_enh.toggled.connect(self._toggle_enhance)
        bl.addWidget(self.btn_enh)
        self.seg_fmt = Segmented([("JPEG", "jpg"), ("PNG", "png")], 56); self.seg_fmt.set(self.opt["fmt"])
        self.seg_fmt.changed.connect(lambda v: self.set_opt("fmt", v)); self.seg_fmt.setFixedWidth(200)
        bl.addWidget(self.seg_fmt)
        pv = QVBoxLayout(); pv.setSpacing(2)
        self.prog = QProgressBar(); self.prog.setRange(0, 1000); self.prog.setTextVisible(False); pv.addWidget(self.prog)
        self.prog_text = label("", "hint"); self.prog_text.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        pv.addWidget(self.prog_text)
        pw = QWidget(); pw.setLayout(pv); pw.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        bl.addWidget(pw, 1)
        self.btn_cancel = button("Cancel", "danger", 110, 56); self.btn_cancel.clicked.connect(self.cancel); bl.addWidget(self.btn_cancel)
        self.btn_swap = button("SWAP", "primary", 170, 58); self.btn_swap.clicked.connect(self.swap); bl.addWidget(self.btn_swap)
        self.btn_save = button("SAVE", "save", 150, 58); self.btn_save.clicked.connect(lambda: self.save()); bl.addWidget(self.btn_save)
        self.btn_folder = button("Open folder", None, 140, 56); self.btn_folder.clicked.connect(self.open_folder); bl.addWidget(self.btn_folder)
        outer.addWidget(bar)
        return w

    # ------------------------------------------------------------------ options
    def set_opt(self, k, v):
        self.opt[k] = v; self.cfg.setValue(k, v)
        if k == "device":
            self.sw.device = v
            if self.sw.engine is not None and not self.busy():
                self.sw.close(); QTimer.singleShot(50, self.warm_engine)
        self.update_ui()

    def open_options(self):
        OptionsDialog(self).exec()
        self.update_ui()

    def retry_gpu(self):
        from .. import dml_probe
        dml_probe.clear_status()
        os.environ["FSP_DML_REPROBE"] = "1"
        if not self.busy():
            self.sw.close(); self.warm_engine()
        QTimer.singleShot(2000, lambda: os.environ.pop("FSP_DML_REPROBE", None))

    def _toggle_enhance(self, on):
        if on and not self.store.is_installed(ENHANCERS[self.opt["enhancer"]]):
            other = "gpen256" if self.opt["enhancer"] == "gpen512" else "gpen512"
            if self.store.is_installed(ENHANCERS[other]):
                self.set_opt("enhancer", other)
            else:
                r = QMessageBox.question(self, "Face enhancer",
                                         f"The face enhancer isn't downloaded yet ({ENHANCER_HQ.bytes / 1e6:.0f} MB). Download it now?")
                self.btn_enh.blockSignals(True); self.btn_enh.setChecked(False); self.btn_enh.blockSignals(False)
                self.set_opt("enhance", False)
                if r == QMessageBox.StandardButton.Yes:
                    self.download_models([ENHANCERS[self.opt["enhancer"]]])
                return
        self.set_opt("enhance", bool(on))

    def active_enhancer(self):
        """The enhancer that will really run: the chosen one, else the other one if only that is installed."""
        if not self.opt["enhance"]:
            return None
        for k in (self.opt["enhancer"], "gpen256" if self.opt["enhancer"] == "gpen512" else "gpen512"):
            if self.store.is_installed(ENHANCERS[k]):
                return k
        return None

    def swap_options(self) -> SwapOptions:
        enh = self.active_enhancer()
        return SwapOptions(enhance=enh, enhance_blend=float(self.opt["blend"]), color_match=bool(self.opt["color_match"]),
                           seamless=bool(self.opt["seamless"]), detail=bool(self.opt["detail"]), device=self.opt["device"])

    def _key(self):
        o = self.swap_options()
        return (self.target.path if self.target else None, tuple(self.mapping), tuple(p.path for p in self.sources),
                o.enhance, o.enhance_blend, o.color_match, o.seamless, o.detail)

    # ------------------------------------------------------------------ workers
    def busy(self):
        # A flag, not QThread.isRunning(): the thread ends before its queued done/failed slot runs.
        return bool(getattr(self, "_running", False))

    def run_worker(self, fn, ok, bad=None, progress=None, text="Working…"):
        if self.busy():
            self._queue.append((fn, ok, bad, progress, text)); return
        self.prog.setRange(0, 0) if progress is None else self.prog.setRange(0, 1000)
        self.prog_text.setText(text)
        w = Worker(fn, self)
        self.worker = w
        self._running = True
        if progress: w.progressed.connect(progress)

        def _done(r):
            self._finish_worker()
            try:
                ok(r)
            finally:
                self.update_ui(); self._next()

        def _fail(msg, tb):
            self._finish_worker()
            try:
                if bad: bad(msg, tb)
                elif msg != "cancelled": self.error(msg)
            finally:
                self.update_ui(); self._next()

        w.done.connect(_done); w.failed.connect(_fail)
        w.start()
        self.update_ui()

    def _finish_worker(self):
        self._running = False
        self.prog.setRange(0, 1000); self.prog.setValue(0); self.prog_text.setText("")
        self.set_chip()

    def _next(self):
        if self._queue and not self.busy():
            self.run_worker(*self._queue.pop(0))

    def cancel(self):
        if self.busy() and self.worker is not None:
            self.worker.cancelled = True
            self.prog_text.setText("Cancelling…")

    def error(self, msg):
        self.set_status("Error: " + msg)
        QMessageBox.warning(self, "Face Swap Photo", msg[:900])

    def warm_engine(self):
        if self.sw.engine is not None or not self.store.all_installed(REQUIRED):
            self.set_chip(); return
        self.chip.setText("Checking GPU…")

        def work(c, e):
            self.store.scan_shared()   # one-time hash check of shared models that have no .ok marker
            return self.sw.get_engine()

        def ok(eng):
            self.set_chip()
            if not getattr(eng, "_gui_hooked", False):
                eng._gui_hooked = True
                eng.on_fallback(lambda r: QTimer.singleShot(0, lambda: self._gpu_fallback(r)))
            if eng.info.fell_back and self.opt["device"] != "cpu":
                self.set_status("GPU (DirectML) failed — using CPU. Still works, just slower.")

        self.run_worker(work, ok, None, None, "Loading face models…")

    def _gpu_fallback(self, reason):
        self.set_chip()
        self.set_status("GPU failed mid-run — switched to CPU: " + (reason or "")[:120])

    # ------------------------------------------------------------------ picking
    def _drop_any(self, paths):
        imgs = [p for p in paths if Path(p).suffix.lower() in IMAGE_EXT]
        if not imgs: return
        if self.target is None:
            self.set_target(imgs[0]); imgs = imgs[1:]
        if imgs: self.add_sources(imgs)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dropEvent(self, e):
        self._drop_any([u.toLocalFile() for u in e.mimeData().urls()])

    def _start_dir(self):
        return str(self.cfg.value("last_dir", str(Path.home() / "Pictures")))

    def pick_target(self):
        if self.busy(): return
        p, _ = QFileDialog.getOpenFileName(self, "Photo to change", self._start_dir(), IMG_FILTER)
        if p:
            self.cfg.setValue("last_dir", str(Path(p).parent)); self.set_target(p)

    def pick_sources(self):
        if self.busy(): return
        ps, _ = QFileDialog.getOpenFileNames(self, "Face photo(s)", self._start_dir(), IMG_FILTER)
        if ps:
            self.cfg.setValue("last_dir", str(Path(ps[0]).parent)); self.add_sources(ps)

    def set_target(self, path):
        def work(c, e):
            return self.sw.load_target(path)

        def ok(ph):
            self.target = ph; self.result = None; self.saved_path = None; self._disp_cache.clear()
            self.rebuild_mapping(); self.set_view("before")
            w, h = ph.size
            self.set_status(f"Photo: {Path(path).name} · {w}×{h} · {len(ph.faces)} face(s)" +
                            ("" if ph.faces else " — no face found, try a clearer photo"))

        self.run_worker(work, ok, None, None, "Finding faces in the photo…")

    def add_sources(self, paths):
        paths = [p for p in paths if Path(p).suffix.lower() in IMAGE_EXT]
        if not paths: return

        def work(c, e):
            out = []
            for p in paths:
                if c(): raise Cancelled()
                out.append(self.sw.load_source(p))
            return out

        def ok(phs):
            for ph in phs:
                if not ph.faces:
                    self.set_status(f"No face found in {Path(ph.path).name}")
                    continue
                self.sources.append(ph)
            self._rebuild_pool(); self.rebuild_mapping()
            n = len(self.pool)
            self.set_status(f"{n} face(s) ready to use" + (f" (max {MAX_SRC_FACES})" if n >= MAX_SRC_FACES else ""))

        self.run_worker(work, ok, None, None, "Reading the face photo(s)…")

    def clear_sources(self):
        if self.busy(): return
        self.sources = []; self._rebuild_pool(); self.rebuild_mapping(); self.result = None; self.set_view("before")

    def _rebuild_pool(self):
        self.pool = []
        for pi, ph in enumerate(self.sources):
            for fi in range(len(ph.faces)):
                if len(self.pool) < MAX_SRC_FACES:
                    self.pool.append((pi, fi))
        self._src_thumbs = [face_thumb(self.sources[pi].img, self.sources[pi].faces[fi], 160) for pi, fi in self.pool]

    def latents(self):
        return [self.sources[pi].latents[fi] for pi, fi in self.pool]

    def rebuild_mapping(self):
        n_t = len(self.target.faces) if self.target else 0
        self.mapping = default_mapping(n_t, len(self.pool), self.target.faces if self.target else None)
        self._tgt_thumbs = [face_thumb(self.target.img, f, 160, mark=True) for f in self.target.faces] if self.target else []
        self.update_ui()

    def flip(self):
        if len(self.mapping) >= 2:
            self.mapping = flip_mapping(self.mapping); self.update_ui()
            if self.result is not None and not self.busy():
                self.swap()

    def pick_for(self, k):
        if k >= len(self.mapping) or not self.pool: return
        d = PickFaceDialog(self, k, self._tgt_thumbs[k], self._src_thumbs, self.mapping[k])
        if d.exec() and d.choice is not None:
            self.mapping[k] = d.choice
            self.update_ui()

    # ------------------------------------------------------------------ swap / save
    def swap(self):
        if self.busy() or not self.target or not self.pool: return
        if not any(m >= 0 for m in self.mapping):
            self.error("No faces are set to swap. Tap a person under 'Who becomes who'."); return
        target, lat, mapping, opts = self.target, self.latents(), list(self.mapping), self.swap_options()
        key = self._key()

        def work(c, e):
            return self.sw.swap(target, lat, mapping, opts, progress=e, cancel=c)

        def prog(d):
            self.prog.setValue(int(1000 * d.get("done", 0) / max(1, d.get("total", 1))))
            self.prog_text.setText(d.get("detail", ""))

        def ok(r):
            self.result = r; self.result_key = key; self.saved_path = None; self._disp_cache.pop("after", None)
            self._disp_cache = {k: v for k, v in self._disp_cache.items() if k == "before"}
            self.set_view("side")
            self.set_status(f"Swapped {r['faces']} face(s) in {r['secs']:.1f}s on {r['device_active']} · "
                            f"{r['W']}×{r['H']} (full size) · enhancer {r['enhance']} — tap SAVE")

        self.run_worker(work, ok, None, prog, "Swapping…")

    def save(self, path=None):
        """Save the swap result at full resolution.

        QPushButton.clicked emits a checked bool — never treat that as a path
        (1.0.0 bug: Path(False) → TypeError on SAVE).
        """
        if self.result is None or self.busy(): return
        # clicked(bool) / accidental non-path args must fall back to auto path
        if not isinstance(path, (str, Path)):
            path = None
        fmt = self.opt["fmt"]
        if path is None:
            stem = "".join(ch for ch in Path(self.target.path).stem if ch.isalnum() or ch in "-_ ")[:40].strip() or "photo"
            path = Path(self.opt["out_dir"]) / f"{stem}_faceswap_{time.strftime('%Y%m%d-%H%M%S')}.{fmt}"
        else:
            path = Path(path)
        img, q = self.result["img"], int(self.opt["jpeg_q"])

        def work(c, e):
            p = save_image(img, path, fmt, q)
            return p

        def ok(p):
            self.saved_path = Path(p)
            mb = self.saved_path.stat().st_size / 1e6
            self.set_status(f"Saved {img.shape[1]}×{img.shape[0]} {fmt.upper()} ({mb:.1f} MB) → {p}")

        self.run_worker(work, ok, None, None, "Saving at full resolution…")

    def open_folder(self):
        d = self.saved_path.parent if self.saved_path else Path(self.opt["out_dir"])
        d.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))

    # ------------------------------------------------------------------ preview
    def _preview_tap(self):
        if self.target is None:
            self.pick_target(); return
        if self.result is not None:
            self.set_view("before" if self.view != "before" else "after")

    def set_view(self, v):
        if v != "before" and self.result is None:
            v = "before"
        self.view = v; self.seg_view.set(v); self.render_preview()

    def _disp(self, which, max_side=1600):
        k = which
        if k not in self._disp_cache:
            src = self.target.img if which == "before" else self.result["img"]
            self._disp_cache[k] = shrink(src, max_side)
        return self._disp_cache[k]

    def render_preview(self):
        if not hasattr(self, "preview"): return
        W, H = self.preview.width() - 8, self.preview.height() - 8
        if self.target is None or W < 20 or H < 20:
            self.preview.setPixmap(QPixmap())
            self.preview.setText("Choose a photo to change and a face photo, then tap SWAP.")
            return
        v = self.view if self.result is not None else "before"
        if v == "before":
            img, s = self._disp("before")
            labels = [f"{i + 1}" + (f"→{LETTERS[m]}" if m >= 0 else "") for i, m in enumerate(self.mapping)]
            colors = [TEAL if m >= 0 else GREY for m in self.mapping]
            show = draw_boxes(img, self.target.faces, s, labels, colors) if self.result is None else img
        elif v == "after":
            show, _ = self._disp("after")
        elif v == "side":
            b, _ = self._disp("before"); a, _ = self._disp("after")
            gap = max(6, b.shape[1] // 120)
            if b.shape[1] / b.shape[0] * 2 <= W / H * 1.35:
                show = np.full((b.shape[0], b.shape[1] * 2 + gap, 3), 14, np.uint8)
                show[:, :b.shape[1]] = b; show[:, b.shape[1] + gap:] = a
            else:
                show = np.full((b.shape[0] * 2 + gap, b.shape[1], 3), 14, np.uint8)
                show[:b.shape[0]] = b; show[b.shape[0] + gap:] = a
            fs = max(0.6, b.shape[1] / 900); th = max(1, int(fs * 2))
            for (x, y), t in (((10, 10), "BEFORE"), ((b.shape[1] + gap + 10, 10) if show.shape[1] > b.shape[1] else (10, b.shape[0] + gap + 10), "AFTER")):
                (tw, tht), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
                cv2.rectangle(show, (x, y), (x + tw + 12, y + tht + 12), (20, 20, 20), -1)
                cv2.putText(show, t, (x + 6, y + tht + 6), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
        else:  # zoom: 1:1-ish crops of the swapped faces, before | after
            show = self._zoom_view(W, H)
        self.preview.setText("")
        self.preview.setPixmap(pix(show, W, H))

    def _zoom_view(self, W, H):
        idx = [i for i, m in enumerate(self.result.get("mapping", self.mapping)) if m >= 0 and i < len(self.target.faces)]
        f = self.target.faces[idx[0] if idx else 0]
        if len(idx) >= 2:
            xs = [self.target.faces[i].bbox for i in idx]
            box = (min(b[0] for b in xs), min(b[1] for b in xs), max(b[2] for b in xs), max(b[3] for b in xs))
        else:
            box = f.bbox
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        r = max(x1 - x0, y1 - y0) * 0.65
        ih, iw = self.target.img.shape[:2]
        a, b = int(max(0, cx - r)), int(max(0, cy - r)); c, d = int(min(iw, cx + r)), int(min(ih, cy + r))
        bc = self.target.img[b:d, a:c]; ac = self.result["img"][b:d, a:c]
        gap = 8
        show = np.full((bc.shape[0], bc.shape[1] * 2 + gap, 3), 14, np.uint8)
        show[:, :bc.shape[1]] = bc; show[:, bc.shape[1] + gap:] = ac
        return show

    def resizeEvent(self, e):
        super().resizeEvent(e)
        QTimer.singleShot(0, self.render_preview)
        if hasattr(self, "_status_text"):
            QTimer.singleShot(0, lambda: self.set_status(self._status_text))
        QTimer.singleShot(0, self._render_thumbs)

    # ------------------------------------------------------------------ refresh
    def _render_thumbs(self):
        if not hasattr(self, "target_slot"): return
        if self.target is not None:
            img, s = self._disp("before")
            labels = [f"{i + 1}" for i in range(len(self.target.faces))]
            colors = [TEAL if (i < len(self.mapping) and self.mapping[i] >= 0) else GREY for i in range(len(self.target.faces))]
            self.target_slot.setText("")
            self.target_slot.setPixmap(pix(draw_boxes(img, self.target.faces, s, labels, colors),
                                           self.target_slot.width() - 6, self.target_slot.height() - 6))
        n = len(self.pool)
        self.src_empty.setVisible(n == 0)
        for i, c in enumerate(self.src_cells):
            c.setVisible(n > 0)
            if i < n:
                c.setText(""); sz = min(c.width(), c.height()) - 4
                c.setPixmap(pix(tag_thumb(self._src_thumbs[i], LETTERS[i]), sz, sz))
            else:
                c.setPixmap(QPixmap()); c.setText("+" if i == n else "")
        for i, b in enumerate(self.map_cards):
            if i < len(self.mapping) and self.target is not None and self.pool:
                m = self.mapping[i]
                tt = tag_thumb(self._tgt_thumbs[i], str(i + 1), TEAL if m >= 0 else GREY)
                st = tag_thumb(self._src_thumbs[m], LETTERS[m]) if m >= 0 else np.full_like(tt, 40)
                if m < 0:
                    cv2.putText(st, "keep", (22, 92), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 200, 200), 2, cv2.LINE_AA)
                arrow = np.full((160, 60, 3), 27, np.uint8)
                cv2.arrowedLine(arrow, (8, 80), (52, 80), (169, 184, 0), 5, cv2.LINE_AA, tipLength=0.45)
                strip = np.hstack([tt, arrow, st])
                iw = max(60, b.width() - 12); ih = min(84, int(iw * strip.shape[0] / strip.shape[1]))
                b.setIcon(QIcon(pix(strip, iw, ih))); b.setIconSize(QSize(iw, ih)); b.setText("")
                b.setToolTip(f"Person {i + 1} → " + (f"face {LETTERS[m]}" if m >= 0 else "keep original"))
                b.setVisible(True)
            else:
                b.setVisible(False)

    def update_ui(self):
        if not hasattr(self, "btn_swap"): return
        busy = self.busy()
        has_t = self.target is not None and bool(self.target.faces)
        has_s = bool(self.pool)
        self.btn_target.setEnabled(not busy); self.btn_add.setEnabled(not busy and len(self.pool) < MAX_SRC_FACES)
        self.btn_clear.setEnabled(not busy and bool(self.sources))
        self.btn_swap.setEnabled(not busy and has_t and has_s and any(m >= 0 for m in self.mapping))
        stale = self.result is not None and self.result_key != self._key()
        self.btn_swap.setText("SWAP again" if stale else "SWAP")
        self.btn_save.setEnabled(not busy and self.result is not None)
        self.btn_folder.setEnabled(self.saved_path is not None)
        self.btn_flip.setEnabled(not busy and len(self.mapping) >= 2 and has_s)
        self.btn_cancel.setVisible(busy)
        self.btn_opts.setEnabled(not busy)
        for b in self.map_cards: b.setEnabled(not busy)
        act = self.active_enhancer()
        on = act is not None
        self.btn_enh.blockSignals(True); self.btn_enh.setChecked(on); self.btn_enh.blockSignals(False)
        q = "HQ" if act == "gpen512" else "Light"
        self.btn_enh.setText(f"Enhance face: {'ON · ' + q if on else 'OFF'}")
        n_map = len(self.mapping) if (self.target and self.pool) else 0
        self.map_empty.setVisible(n_map == 0)
        if self.target is not None:
            w, h = self.target.size
            extra = f" · first {MAX_MAP_CARDS} shown below" if len(self.target.faces) > MAX_MAP_CARDS else ""
            self.target_info.setText(elide(self.target_info, f"{Path(self.target.path).name} · {w}×{h} · "
                                           f"{len(self.target.faces)} face(s){extra}", 300))
        else:
            self.target_info.setText("")
        self.src_info.setText(f"{len(self.pool)} face(s) from {len(self.sources)} photo(s)" if self.sources else "")
        if hasattr(self, "_tgt_thumbs") and hasattr(self, "_src_thumbs"):
            self._render_thumbs()
        elif self.target is not None or self.pool:
            if not hasattr(self, "_src_thumbs"): self._src_thumbs = []
            if not hasattr(self, "_tgt_thumbs"): self._tgt_thumbs = []
            self._render_thumbs()
        self.render_preview()

    # ------------------------------------------------------------------ layout check (CI / self-check)
    def layout_report(self):
        """Proves the single-screen rule: no scroll areas, every control visible inside the window."""
        scrolls = [type(w).__name__ for w in self.findChildren(QAbstractScrollArea) if w.isVisible()]
        names = dict(target=self.target_slot, sources=self.src_area, preview=self.preview, swap=self.btn_swap,
                     save=self.btn_save, flip=self.btn_flip, enhance=self.btn_enh, fmt=self.seg_fmt,
                     view=self.seg_view, options=self.btn_opts, folder=self.btn_folder)
        for i, b in enumerate(self.map_cards):
            if b.isVisible(): names[f"map{i + 1}"] = b
        win_rect = self.centralWidget().rect()
        out = {}
        for n, w in names.items():
            tl = w.mapTo(self.centralWidget(), w.rect().topLeft())
            r = w.rect().translated(tl)
            out[n] = dict(visible=w.isVisible(), x=r.x(), y=r.y(), w=r.width(), h=r.height(),
                          inside=win_rect.contains(r), min_side_ok=min(r.width(), r.height()) >= 40)
        msh = self.minimumSizeHint()
        return dict(window=[self.width(), self.height()], central=[win_rect.width(), win_rect.height()],
                    min_size_hint=[msh.width(), msh.height()], scroll_widgets=scrolls, widgets=out,
                    ok=not scrolls and all(v["visible"] and v["inside"] and v["min_side_ok"] for v in out.values()))


def make_app(argv=None):
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setApplicationName("Face Swap Photo"); app.setOrganizationName("vanu")
    app.setStyle("Fusion")
    font = QFont("Segoe UI", 11); font.setStyleHint(QFont.StyleHint.SansSerif); app.setFont(font)
    app.setStyleSheet(QSS)
    ico = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2] / "packaging")) / "icon.ico"
    if ico.is_file(): app.setWindowIcon(QIcon(str(ico)))
    return app


def spin(app, sec=0.0, until=None, timeout=900):
    t0 = time.time()
    while True:
        app.processEvents(); time.sleep(0.02)
        if until is None and time.time() - t0 >= sec: return True
        if until is not None and until(): return True
        if time.time() - t0 > timeout: return False


def autorun(app, win, args):
    """Scripted run of the REAL window (used for screenshots, CI and the box trial):
    load target + faces through the normal UI paths, SWAP, optionally SAVE, grab screenshots."""
    import json
    report = dict(ok=False, steps={})
    out = Path(args.shots_dir) if getattr(args, "shots_dir", None) else None
    if out: out.mkdir(parents=True, exist_ok=True)
    idle = lambda: not win.busy() and not win._queue

    def shot(name, size=None):
        if size:
            if win.isMaximized() or win.isFullScreen(): win.showNormal()
            win.resize(*size)
        spin(app, 0.6)
        win.render_preview(); win._render_thumbs(); spin(app, 0.3)
        if out:
            p = out / f"{name}.png"; win.grab().save(str(p)); report["steps"][f"shot_{name}"] = str(p)
            report.setdefault("layout", {})[name] = win.layout_report()

    spin(app, until=idle, timeout=600)
    report["device"] = win.chip.text()
    if out: shot("main_empty_1280x720", (1280, 720))
    if args.target:
        win.set_target(args.target); spin(app, until=idle, timeout=900)
        report["steps"]["target_faces"] = len(win.target.faces) if win.target else 0
    if args.faces:
        win.add_sources(list(args.faces)); spin(app, until=idle, timeout=900)
        report["steps"]["source_faces"] = len(win.pool)
    if getattr(args, "flip", False):
        win.mapping = flip_mapping(win.mapping); win.update_ui()
    report["steps"]["mapping"] = list(win.mapping)
    if out: shot("main_loaded_1280x720", (1280, 720))
    if getattr(args, "auto_swap", False) and win.btn_swap.isEnabled():
        win.swap(); spin(app, until=idle, timeout=1800)
        r = win.result
        report["steps"]["swap"] = None if r is None else dict(W=r["W"], H=r["H"], faces=r["faces"], secs=r["secs"],
                                                               device=r["device"], enhance=r["enhance"])
        report["steps"]["status"] = getattr(win, "_status_text", "")
    if out and win.result is not None:
        shot("main_result_1280x720", (1280, 720))
        shot("main_result_1280x800", (1280, 800))
        shot("main_result_1280x640_maximized_with_taskbar", (1280, 640))
        win.set_view("zoom"); shot("main_face_zoom_1280x720", (1280, 720)); win.set_view("side")
        # popups are fixed-size dialogs (no scrolling) — capture them too
        try:
            d = OptionsDialog(win); d.show(); spin(app, 0.5); d.grab().save(str(out / "options_popup.png")); d.close()
            if win.pool and win.mapping:
                d2 = PickFaceDialog(win, 0, win._tgt_thumbs[0], win._src_thumbs, win.mapping[0]); d2.show(); spin(app, 0.5)
                d2.grab().save(str(out / "who_becomes_who_popup.png")); d2.close()
        except Exception as e:  # noqa: BLE001
            report["steps"]["popup_error"] = str(e)
    if getattr(args, "auto_save", None) and win.result is not None:
        win.save(args.auto_save); spin(app, until=idle, timeout=600)
        report["steps"]["saved"] = str(win.saved_path) if win.saved_path else None
        if win.saved_path and win.saved_path.is_file():
            chk = cv2.imdecode(np.fromfile(str(win.saved_path), np.uint8), cv2.IMREAD_COLOR)
            report["steps"]["saved_size"] = [int(chk.shape[1]), int(chk.shape[0])]
    # Click SAVE like a user (QPushButton.clicked emits bool) for JPEG and PNG — guards the 1.0.0 path bug
    if getattr(args, "click_save", False) and win.result is not None:
        click_dir = Path(getattr(args, "click_save_dir", None) or (Path(args.shots_dir) / "click_saves" if out else Path(win.opt["out_dir"])))
        click_dir.mkdir(parents=True, exist_ok=True)
        win.set_opt("out_dir", str(click_dir))
        tsize = list(win.target.size) if win.target is not None else None
        click_report = {}
        for fmt in ("jpg", "png"):
            win.set_opt("fmt", fmt)
            win.seg_fmt.set(fmt)
            spin(app, 0.1)
            # Real button click: emits checked bool into whatever was connected — must not crash
            win.btn_save.click()
            spin(app, until=idle, timeout=600)
            p = win.saved_path
            entry = dict(path=str(p) if p else None, ok=False, size=None, fmt=fmt)
            if p is not None and p.is_file():
                chk = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
                if chk is not None:
                    entry["size"] = [int(chk.shape[1]), int(chk.shape[0])]
                    entry["ok"] = entry["size"] == tsize and p.suffix.lower() in ((".jpg", ".jpeg") if fmt == "jpg" else (".png",))
            click_report[fmt] = entry
        report["steps"]["click_save"] = click_report
        report["steps"]["click_save_ok"] = all(v.get("ok") for v in click_report.values())
    if getattr(args, "screen_grab", None):
        win.resize(1280, 720); spin(app, 0.8)
        scr = QApplication.primaryScreen()
        if scr is not None:
            scr.grabWindow(0).save(str(args.screen_grab)); report["steps"]["screen_grab"] = str(args.screen_grab)
    hold = float(getattr(args, "hold", 0) or 0)
    if hold > 0:  # keep the real window up (1280x720 logical, result shown) for an external screen capture
        win.set_view("side"); win.resize(1280, 720); win.move(0, 0); spin(app, 0.5)
        print("HOLDING", flush=True)
        spin(app, hold)
    report["device"] = win.chip.text()
    tsize = list(win.target.size) if win.target is not None else None
    sw = report["steps"].get("swap") or {}
    report["ok"] = bool(sw) and [sw.get("W"), sw.get("H")] == tsize and sw.get("faces", 0) >= 1 and \
        all(v.get("ok") for v in report.get("layout", {}).values()) and \
        (not getattr(args, "auto_save", None) or report["steps"].get("saved_size") == tsize) and \
        (not getattr(args, "click_save", False) or report["steps"].get("click_save_ok"))
    report["target_size"] = tsize
    if out:
        (out / "gui_autorun_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str), flush=True)
    return 0 if report["ok"] else 3


def run_gui(args=None) -> int:
    crashlog.install()
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("vanu.FaceSwapPhoto")
        except Exception:  # noqa: BLE001
            pass
    if args is not None and getattr(args, "shots_dir", None):
        os.environ.setdefault("QT_SCALE_FACTOR", "1.5")
    app = make_app()
    try:
        from PySide6.QtCore import QtMsgType, qInstallMessageHandler

        def _qt_msg(mode, context, message):
            if mode in (QtMsgType.QtFatalMsg, QtMsgType.QtCriticalMsg):
                crashlog.record(RuntimeError, RuntimeError(f"Qt {mode}: {message}"), None, where="qt")
        qInstallMessageHandler(_qt_msg)
    except Exception:  # noqa: BLE001
        pass
    store = ModelStore(Path(args.models) if args is not None and getattr(args, "models", None) else None)
    win = MainWindow(store, getattr(args, "device", "auto") if args is not None else "auto")
    try:
        from .gamepad import Gamepad
        win.gamepad = Gamepad(win)
    except Exception:  # noqa: BLE001
        pass
    scripted = args is not None and (getattr(args, "auto_swap", False) or getattr(args, "shots_dir", None) or getattr(args, "click_save", False))
    win.resize(1280, 720)
    scr = QApplication.primaryScreen()
    if not scripted and scr is not None and scr.availableGeometry().width() <= 1400:
        win.showMaximized()
    else:
        win.show()
    if scripted:
        return autorun(app, win, args)
    if args is not None:
        if getattr(args, "target", None):
            QTimer.singleShot(300, lambda: win.set_target(args.target))
        if getattr(args, "faces", None):
            QTimer.singleShot(400, lambda: win.add_sources(list(args.faces)))
    return app.exec()
