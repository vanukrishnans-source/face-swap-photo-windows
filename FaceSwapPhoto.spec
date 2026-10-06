# -*- mode: python ; coding: utf-8 -*-
# PyInstaller onedir build for Face Swap Photo (Windows x64; the same spec also builds on Linux for testing).
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

block_cipher = None
root = Path(SPECPATH)
datas = [
    (str(root / "packaging" / "icon.ico"), "."),
    (str(root / "THIRD_PARTY.md"), "."),
    (str(root / "LICENSE"), "."),
]
binaries = []
hidden = ["onnxruntime", "cv2", "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets"]
d, b, h = collect_all("onnxruntime")
datas += d; binaries += b; hidden += h

a = Analysis(
    [str(root / "fsp" / "__main__.py")],
    pathex=[str(root)], binaries=binaries, datas=datas, hiddenimports=hidden,
    hookspath=[], hooksconfig={}, runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "IPython", "notebook", "pytest", "onnx", "mediapipe", "scipy", "pandas",
              "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtNetwork", "PySide6.QtOpenGL",
              "PySide6.QtPdf", "PySide6.QtSvg", "PySide6.QtDBus", "PySide6.QtMultimedia", "PySide6.QtWebEngineCore"],
    cipher=block_cipher, noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
icon = str(root / "packaging" / "icon.ico")
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="FaceSwapPhoto", debug=False,
          bootloader_ignore_signals=False, strip=False, upx=False, console=False,
          disable_windowed_traceback=False, argv_emulation=False, icon=icon)
exe_console = EXE(pyz, a.scripts, [], exclude_binaries=True, name="FaceSwapPhoto_cli", debug=False,
                  bootloader_ignore_signals=False, strip=False, upx=False, console=True, icon=icon)
coll = COLLECT(exe, exe_console, a.binaries, a.zipfiles, a.datas, strip=False, upx=False, upx_exclude=[],
               name="FaceSwapPhoto")
