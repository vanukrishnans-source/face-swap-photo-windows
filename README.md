# Face Swap Photo for Windows

**Photo-only** face swap for the **ASUS ROG Ally X** (Windows 11 handheld · Ryzen Z1 Extreme · Radeon 780M · 24 GB).
No video mode — pick a photo, pick the face(s), swap, compare, save at **full original resolution**.

| | |
|---|---|
| Package | Face Swap Photo **1.0.1** · portable zip · unsigned |
| Acceleration | ONNX Runtime **DirectML** on the Radeon 780M · DirectML is probed in a child process first, any failure → **CPU** (the app never closes because of a GPU driver crash) |
| Models | YOLO Face 8n · ArcFace w600k r50 · inswapper 128 fp16 · optional GPEN-BFR 512 / 256 enhancer |
| Output | Same width × height as the original photo · **PNG** (lossless) or **JPEG** q97, 4:4:4 chroma |
| Save folder | `%USERPROFILE%\Pictures\FaceSwapPhoto` (change in Options) |
| Logs | `%LOCALAPPDATA%\FaceSwapPhoto\crash.log`, `faceswapphoto.log` |

## Changelog

* **1.0.1** — Fix SAVE crash when PNG/JPEG was selected: `QPushButton.clicked` was wiring its checked `bool` into `save(path=…)`, so `Path(False)` raised `argument should be a str or an os.PathLike object … not 'bool'`. Path is always a str/Path; format is separate. CI clicks SAVE for JPEG and PNG at full resolution.
* **1.0.0** — Initial Ally X release.

## Install / run

1. Download **`FaceSwapPhoto-1.0.1-win64.zip`** from [Releases](https://github.com/vanukrishnans-source/face-swap-photo-windows/releases) and unzip anywhere (e.g. `C:\Games\FaceSwapPhoto\`).
2. Run **`FaceSwapPhoto.exe`**. SmartScreen (the build is not code-signed): **More info** → **Run anyway**.
3. Models:
   * If Face Fusion Studio is installed, its models in `%LOCALAPPDATA%\FaceFusionStudio\models` are **reused in place** (SHA-256 checked, not copied). GIF Face Swap's folder is checked too.
   * Otherwise the one-time setup screen downloads them into `%LOCALAPPDATA%\FaceSwapPhoto\models`: **~465 MB required** (YOLO Face 12.7 MB + ArcFace 174.4 MB + inswapper 277.7 MB) **+ 284 MB HQ enhancer** (GPEN 512, ticked by default) · optional Light enhancer 76 MB. Downloads resume, every file is SHA-256 verified.
   * Download sources, in order: `vanukrishnans-source/collage-video` models-v1 (ArcFace, inswapper, GPEN) and `vanukrishnans-source/face-swap-photo-windows` models-v1 (YOLO Face), then the public FaceFusion assets on GitHub / Hugging Face as fallback.

Verify the zip: `Get-FileHash .\FaceSwapPhoto-1.0.1-win64.zip -Algorithm SHA256`

## Using it (one screen, no scrolling)

Everything is on a single fixed screen sized for the Ally X at 150 % (1280 × 720 logical); it also fits 1280 × 800 and a maximised window above the taskbar (1280 × 640). There are no scroll areas anywhere — rare settings are in a small fixed **Options** popup.

1. **Photo to change** — tap the box (or Choose). Works with one or many people; faces are numbered left → right.
2. **Faces** — tap Add and pick one or more face photos (a single photo with both of you works: faces are taken left → right and lettered A, B, …; up to 8 faces).
3. **Who becomes who** — couples are mapped left → right automatically (person 1 → face A, person 2 → face B). Tap **⇄ Flip** to swap them, or tap a person card to pick any face or *Keep original*. With only one face it goes on the biggest person; tap others to add.
4. **SWAP** — then compare with **Before / After / Side by side / Face zoom** (tap the preview to flip before/after).
5. **SAVE** — full resolution, JPEG or PNG (toggle in the bottom bar). **Open folder** shows the file.

Bottom bar: **Enhance face** toggle (GPEN; HQ 512 by default, Light 256 in Options). Options: enhancer quality and strength, sharpen + grain match, face colour match, Poisson edge, JPEG quality, processor (Auto / GPU / CPU) + Retry GPU, save folder.

Controller: D-pad / stick moves focus, **A** presses, **B** closes popups, **Start / Enter** = SWAP. Keyboard: Enter = Swap, Ctrl+S = Save, F11 = full screen.

## Quality notes

* **Full resolution**: the photo is never downscaled. Detection runs on a 640 px copy (that is how YOLO Face works) and, on big photos, each face is re-detected on a tight crop so landmarks are accurate at full size. Only the face region is replaced; every other pixel is bit-identical to the original (PNG) — checked in CI.
* **Face-only colour blend**: colour matching is measured and applied only inside the soft face mask. Body, neck, arms, hair and background keep their original colours.
* **Couples cheek-to-cheek**: each person's swap is masked away from the other person's face, so glasses / beard from one source don't bleed onto the partner.
* **Big faces**: inswapper works at 128 px and GPEN at 512 px. For faces much larger than that (e.g. a 24 MP photo where a face is ~1000 px), the app adds a gentle upscale-aware sharpen and matches the photo's grain so the face isn't plastic — but at 100 % zoom the new face is still softer than the untouched parts of the photo. On the Ally X screen / phone-size viewing this is not visible.

## Command line (`FaceSwapPhoto_cli.exe`)

```powershell
# headless swap, couple photo, faces from one photo (left->right), HQ enhancer, PNG
FaceSwapPhoto_cli.exe --swap couple.jpg --faces us.jpg --out swapped.png --enhance gpen512
# flip who-becomes-who, or give an explicit map (target faces left->right; - = keep)
FaceSwapPhoto_cli.exe --swap couple.jpg --faces us.jpg --out swapped.jpg --flip
FaceSwapPhoto_cli.exe --swap group.jpg --faces a.jpg b.jpg --out out.jpg --map "1,-,0"
# end-to-end self test (what CI runs on the packaged exe)
FaceSwapPhoto_cli.exe --selftest --target photo.jpg --faces faces.jpg --out selftest-out --device cpu
FaceSwapPhoto_cli.exe --version
```

## Build from source

Windows x64 or the GitHub Actions workflow (`.github/workflows/build.yml`, `windows-latest`): builds with PyInstaller, then tests the **packaged** exes — CLI selftest on a 6000 × 4000 CC0 photo (full-res kept, faces changed, nothing outside the faces changed, PNG lossless, JPEG PSNR, Flip), a headless swap with `--device auto`, a forced DirectML failure → CPU fallback test, and the **windowed exe** running a real swap + save through the real window with a no-scroll layout check and screenshots.

```powershell
py -3.13 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements-win.txt
python scripts/prefetch_models.py .models-cache light hq
python -m fsp --selftest --target testdata\target_couple_e5.jpg --faces testdata\faces_mixkit48205.jpg --models .models-cache --device cpu
pyinstaller FaceSwapPhoto.spec --noconfirm
powershell -File packaging\make_zip.ps1
```

## Licences

* App code: MIT (`LICENSE`). Third-party notices: `THIRD_PARTY.md`.
* **InsightFace models (ArcFace, inswapper): personal / non-commercial use only.** YOLO Face: GPL-3.0 (derronqi). GPEN-BFR: research release.
* Only swap faces of people who agreed to it; don't use this to impersonate or deceive anyone.

Related apps (separate repos, untouched by this one): face-fusion-windows, face-swap-video-windows, gif-face-swap-windows, body-swap-windows.
