"""Dark, touch-first theme for the Ally X (7" 1080p @ 150 % = 1280x720 logical). Tap targets >= 52 px."""

QSS = """
* { font-family: "Segoe UI Variable", "Segoe UI", Tahoma, "Noto Sans", "DejaVu Sans", sans-serif; }
QWidget { background:#0e1117; color:#e8eaed; font-size:16px; }
QMainWindow, QDialog { background:#0e1117; }
QLabel { background:transparent; }
QLabel#apptitle { font-size:22px; font-weight:700; color:#ffffff; }
QLabel#title { font-size:24px; font-weight:700; color:#ffffff; }
QLabel#section { font-size:17px; font-weight:700; color:#ffffff; }
QLabel#hint { font-size:14px; color:#9aa0a6; }
QLabel#status { font-size:15px; color:#cfd3d8; }
QLabel#chip { font-size:14px; font-weight:600; color:#041014; background:#00b8a9; border-radius:12px; padding:4px 12px; }
QLabel#chip[state="cpu"] { background:#ffb74d; }
QLabel#slot { background:#11151c; border:2px dashed #3a4254; border-radius:12px; color:#8b929c; font-size:16px; }
QLabel#preview { background:#0a0d12; border-radius:12px; color:#8b929c; font-size:18px; }
QFrame#card { background:#161a22; border:1px solid #2a3140; border-radius:14px; }
QFrame#bar { background:#12161d; border-top:1px solid #2a3140; }
QFrame#topbar { background:#12161d; border-bottom:1px solid #2a3140; }
QPushButton {
  background:#232833; color:#e8eaed; border:2px solid #3a4252; border-radius:12px;
  padding:6px 14px; min-height:48px; font-size:16px; font-weight:600;
}
QPushButton:hover { background:#2c3340; border-color:#4a5568; }
QPushButton:pressed { background:#1a1e27; }
QPushButton:disabled { color:#5f6368; background:#161a22; border-color:#2a2f3a; }
QPushButton#primary { background:#00b8a9; color:#041014; border:none; font-weight:800; font-size:20px; }
QPushButton#primary:hover { background:#1ad1c1; }
QPushButton#primary:disabled { background:#1a4a45; color:#6a8a86; }
QPushButton#save { background:#3d7bff; color:#ffffff; border:none; font-weight:800; font-size:19px; }
QPushButton#save:hover { background:#5a8fff; }
QPushButton#save:disabled { background:#1d2c4d; color:#6b7a99; }
QPushButton#danger { background:#c5221f; color:#fff; border:none; }
QPushButton#seg { border-radius:10px; min-height:44px; padding:4px 10px; font-size:15px; }
QPushButton#seg:checked { background:#00b8a9; color:#041014; border:none; font-weight:800; }
QPushButton#toggle:checked { background:#00b8a9; color:#041014; border:none; font-weight:800; }
QPushButton#mapcard { background:#1b2029; border:2px solid #2f3747; border-radius:12px; padding:4px; }
QPushButton#mapcard:hover { border-color:#00b8a9; }
QPushButton#pick { background:#1b2029; border:2px solid #2f3747; border-radius:12px; padding:6px; font-size:16px; }
QPushButton#pick:checked { border:3px solid #00b8a9; }
QPushButton:focus { border:2px solid #ff8a3d; }
QProgressBar { background:#161a22; border:1px solid #2a3140; border-radius:8px; text-align:center;
  min-height:20px; max-height:20px; color:#e8eaed; font-size:13px; }
QProgressBar::chunk { background:#00b8a9; border-radius:7px; }
QCheckBox { spacing:12px; font-size:16px; min-height:44px; }
QCheckBox::indicator { width:28px; height:28px; }
QMessageBox QLabel { font-size:16px; }
"""
