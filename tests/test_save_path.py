"""Unit tests for the 1.0.0 SAVE crash: path must never be a bool.

QPushButton.clicked emits checked:bool. Connecting that signal directly to
MainWindow.save(path=None) made path=False, then Path(False) / cv2 path APIs
raised: argument should be a str or an os.PathLike object where __fspath__
returns a str, not 'bool'.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fsp.photo import save_image  # noqa: E402


class TestSaveImagePath(unittest.TestCase):
    def setUp(self):
        self.img = np.zeros((64, 96, 3), dtype=np.uint8)
        self.img[:, :] = (40, 80, 160)
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_png_and_jpeg_write_full_resolution(self):
        p_png = save_image(self.img, self.dir / "out.png", "png")
        p_jpg = save_image(self.img, self.dir / "out.jpg", "jpg", 97)
        self.assertTrue(p_png.is_file())
        self.assertTrue(p_jpg.is_file())
        back_png = cv2.imdecode(np.fromfile(str(p_png), np.uint8), cv2.IMREAD_COLOR)
        back_jpg = cv2.imdecode(np.fromfile(str(p_jpg), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(list(back_png.shape), list(self.img.shape))
        self.assertEqual(list(back_jpg.shape), list(self.img.shape))
        self.assertTrue((back_png == self.img).all())

    def test_bool_path_rejected(self):
        with self.assertRaises(TypeError) as cm:
            save_image(self.img, False, "png")  # type: ignore[arg-type]
        self.assertIn("bool", str(cm.exception).lower())
        with self.assertRaises(TypeError):
            save_image(self.img, True, "jpg")  # type: ignore[arg-type]

    def test_format_is_separate_from_path(self):
        # path stem without suffix + explicit fmt must produce correct extension
        p = save_image(self.img, self.dir / "nosuffix", "png")
        self.assertEqual(p.suffix.lower(), ".png")
        p2 = save_image(self.img, self.dir / "nosuffix2", "jpg")
        self.assertEqual(p2.suffix.lower(), ".jpg")


class TestSaveIgnoresClickedBool(unittest.TestCase):
    """Exercise MainWindow.save with the exact bool Qt would pass."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_save_with_false_writes_real_png_and_jpeg(self):
        from fsp.gui.app import MainWindow
        from fsp.models import ModelStore

        store = ModelStore(Path(tempfile.mkdtemp(prefix="fsp-models-")))
        win = MainWindow(store, "cpu")
        win.update_ui = lambda: None  # isolate save path from thumb/render plumbing

        h, w = 48, 72
        img = np.full((h, w, 3), 90, dtype=np.uint8)
        win.result = dict(img=img, W=w, H=h, faces=1, secs=0.0, device="cpu", enhance=None)
        tdir = Path(tempfile.mkdtemp(prefix="fsp-tgt-"))
        tpath = tdir / "photo.jpg"
        tpath.write_bytes(b"x")
        win.target = type("T", (), {"path": str(tpath), "size": (w, h), "faces": [], "img": img})()
        win.mapping = []
        out = Path(tempfile.mkdtemp(prefix="fsp-save-"))
        win.opt["out_dir"] = str(out)

        for fmt in ("jpg", "png"):
            win.opt["fmt"] = fmt
            # Simulate QPushButton.clicked(checked=False) — the 1.0.0 bug input
            win.save(False)
            # Drain the worker
            for _ in range(100):
                self.app.processEvents()
                if win.saved_path and win.saved_path.is_file() and not win.busy():
                    if win.saved_path.suffix.lower() in ((".jpg", ".jpeg") if fmt == "jpg" else (".png",)):
                        break
                import time
                time.sleep(0.05)
            self.assertFalse(win.busy(), f"save worker hung for {fmt}")
            self.assertIsNotNone(win.saved_path)
            self.assertTrue(win.saved_path.is_file(), f"{fmt} not written at {win.saved_path}")
            chk = cv2.imdecode(np.fromfile(str(win.saved_path), np.uint8), cv2.IMREAD_COLOR)
            self.assertIsNotNone(chk)
            self.assertEqual([chk.shape[1], chk.shape[0]], [w, h])
            self.assertIn(win.saved_path.suffix.lower(), (".jpg", ".jpeg") if fmt == "jpg" else (".png",))

        # Button click path (lambda: self.save()) must also write
        win.opt["fmt"] = "png"
        win.btn_save.setEnabled(True)
        win.btn_save.click()
        for _ in range(100):
            self.app.processEvents()
            if win.saved_path and win.saved_path.is_file() and not win.busy():
                break
            import time
            time.sleep(0.05)
        self.assertTrue(win.saved_path.is_file())
        win.close()


if __name__ == "__main__":
    unittest.main()
