"""FaceSwapPhoto(.exe) entry: GUI by default; headless --swap / --selftest / --test-dml-fallback / --dml-probe;
scripted GUI runs with --auto-swap / --shots-dir (real window, real swap, screenshots)."""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import __version__

log = logging.getLogger("fsp")


def _setup_logging(verbose=True):
    from . import crashlog
    crashlog.install()
    base = crashlog.log_dir(); base.mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(base / "faceswapphoto.log", encoding="utf-8")]
    if verbose and sys.stdout is not None:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers, force=True)


def versions():
    import cv2
    import onnxruntime as ort
    return dict(app=__version__, python=sys.version.split()[0], platform=platform.platform(), machine=platform.machine(),
                cpu_count=os.cpu_count(), onnxruntime=ort.__version__, providers=ort.get_available_providers(),
                opencv=cv2.__version__, numpy=np.__version__, frozen=bool(getattr(sys, "frozen", False)))


def _store(args, want_enh=None):
    from .models import ENHANCERS, REQUIRED, ModelStore
    store = ModelStore(Path(args.models) if args.models else None)
    need = list(REQUIRED) + ([ENHANCERS[want_enh]] if want_enh else [])
    last = [0.0]

    def prog(f, done, total, bps, verifying):
        if time.time() - last[0] > 5 or done == total:
            last[0] = time.time()
            log.info("%s %s %d/%d MB", "verify" if verifying else "download", f, done // 1_000_000, total // 1_000_000)
    store.ensure(need, progress=prog)
    return store


def _parse_map(s, n):
    if not s:
        return None
    m = [int(x) if x.strip() not in ("-", "x", "") else -1 for x in s.split(",")]
    return (m + [-1] * n)[:n]


def swap_headless(args):
    from .photo import PhotoSwapper, SwapOptions, before_after, default_mapping, flip_mapping, save_image
    import cv2
    enh = None if args.enhance == "off" else args.enhance
    store = _store(args, enh)
    sw = PhotoSwapper(store, args.device)
    tgt = sw.load_target(args.swap)
    srcs = [sw.load_source(p) for p in args.faces]
    lat = [l for s in srcs for l in s.latents]
    mapping = _parse_map(args.map, len(tgt.faces)) or default_mapping(len(tgt.faces), len(lat), tgt.faces)
    if args.flip:
        mapping = flip_mapping(mapping)
    r = sw.swap(tgt, lat, mapping, SwapOptions(enhance=enh, color_match=args.color_match, seamless=args.seamless,
                                               detail=args.detail, device=args.device))
    out = save_image(r["img"], args.out, args.format, args.jpeg_quality)
    if args.before_after:
        save_image(before_after(tgt.img, r["img"]), args.before_after, "jpg", 92)
    res = {k: v for k, v in r.items() if k != "img"}
    res.update(out=str(out), target_size=list(tgt.size), target_faces=len(tgt.faces), source_faces=len(lat),
               bytes=out.stat().st_size)
    print(json.dumps(res, indent=2, default=str))
    sw.close()
    return 0


def selftest(args):
    """Packaged-exe end-to-end photo swap check (used by CI on windows-latest)."""
    import cv2
    from .photo import PhotoSwapper, SwapOptions, before_after, default_mapping, flip_mapping, save_image
    out = Path(args.out or "selftest-out"); out.mkdir(parents=True, exist_ok=True)
    report = dict(ok=False, versions=versions(), checks={})
    c = report["checks"]
    try:
        enh = None if args.enhance == "off" else args.enhance
        store = _store(args, enh)
        sw = PhotoSwapper(store, args.device)
        t0 = time.perf_counter()
        tgt = sw.load_target(args.target)
        srcs = [sw.load_source(p) for p in args.faces]
        lat = [l for s in srcs for l in s.latents]
        report["target"] = dict(path=args.target, size=list(tgt.size), faces=len(tgt.faces))
        report["sources"] = [dict(path=s.path, size=list(s.size), faces=len(s.faces)) for s in srcs]
        c["target_faces_found"] = len(tgt.faces) >= 1
        c["source_faces_found"] = len(lat) >= 1
        mapping = default_mapping(len(tgt.faces), len(lat), tgt.faces)
        r = sw.swap(tgt, lat, mapping, SwapOptions(enhance=enh, device=args.device))
        report["swap"] = {k: v for k, v in r.items() if k != "img"}
        img = r["img"]
        c["full_resolution_kept"] = list(img.shape[:2]) == list(tgt.img.shape[:2])
        diff = np.abs(img.astype(np.int16) - tgt.img.astype(np.int16)).max(axis=2)
        changed = diff > 0
        # every swapped face region changed noticeably
        per_face = []
        for i, m in enumerate(mapping):
            if m < 0: continue
            x0, y0, x1, y1 = [int(v) for v in tgt.faces[i].bbox]
            reg = diff[max(0, y0):y1, max(0, x0):x1]
            per_face.append(float(reg.mean()) if reg.size else 0.0)
        report["face_mean_abs_diff"] = per_face
        c["faces_changed"] = bool(per_face) and min(per_face) > 2.0
        # face-only: nothing far away from the faces may change (no body / skin / background recolour)
        keep = np.ones_like(changed)
        for f in tgt.faces:
            x0, y0, x1, y1 = f.bbox; w, h = x1 - x0, y1 - y0
            keep[max(0, int(y0 - 1.0 * h)):int(y1 + 1.0 * h), max(0, int(x0 - 1.0 * w)):int(x1 + 1.0 * w)] = False
        c["face_only_no_body_recolour"] = int(changed[keep].sum()) == 0
        report["changed_fraction"] = float(changed.mean())
        # save PNG + JPEG at full size and decode back
        p_png = save_image(img, out / "selftest_swap.png", "png")
        p_jpg = save_image(img, out / "selftest_swap.jpg", "jpg", 97)
        back_png = cv2.imdecode(np.fromfile(str(p_png), np.uint8), cv2.IMREAD_COLOR)
        back_jpg = cv2.imdecode(np.fromfile(str(p_jpg), np.uint8), cv2.IMREAD_COLOR)
        c["png_lossless_full_size"] = back_png is not None and back_png.shape == img.shape and bool((back_png == img).all())
        c["jpeg_full_size"] = back_jpg is not None and back_jpg.shape == img.shape
        report["jpeg_psnr"] = float(cv2.PSNR(back_jpg, img)) if back_jpg is not None else 0.0
        c["jpeg_high_quality"] = report["jpeg_psnr"] > 38.0
        save_image(before_after(tgt.img, img), out / "selftest_before_after.jpg", "jpg", 92)
        # Flip must change the result for a 2-face mapping
        if sum(1 for m in mapping if m >= 0) >= 2:
            r2 = sw.swap(tgt, lat, flip_mapping(mapping), SwapOptions(enhance=None, device=args.device))
            c["flip_changes_result"] = bool(np.abs(r2["img"].astype(np.int16) - img.astype(np.int16)).max() > 0)
        report["total_s"] = round(time.perf_counter() - t0, 2)
        report["device"] = sw.engine.info.label()
        if args.dml_smoke:
            from . import dml_probe
            from .models import YOLOFACE
            os.environ["FSP_DML_REPROBE"] = "1"
            ok, why = dml_probe.probe_directml(str(store.path(YOLOFACE)))
            os.environ.pop("FSP_DML_REPROBE", None)
            report["directml"] = dict(ok=ok, reason=why[:300])
            dml_probe.clear_status()
        sw.close()
        report["ok"] = all(c.values())
    except Exception as e:  # noqa: BLE001
        report["error"] = f"{type(e).__name__}: {e}"; report["traceback"] = traceback.format_exc()
        log.error("selftest failed: %s", report["traceback"])
    (out / "selftest_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "versions"}, indent=2, default=str), flush=True)
    return 0 if report["ok"] else 1


def test_dml_fallback(args):
    """Simulate DirectML failure; Engine must land on CPU, process must stay alive, inference must work."""
    from . import detect, dml_probe
    from .engine import Engine
    report = {"ok": False, "checks": {}}
    try:
        os.environ["FSP_FORCE_DML_FAIL"] = "1"
        dml_probe.clear_status()
        store = _store(args)
        eng = Engine(store, "auto")
        info = eng.prepare(None)
        report.update(device=info.label(), active=info.active, fell_back=info.fell_back, fallback_reason=info.fallback_reason)
        report["checks"]["stayed_alive"] = True
        report["checks"]["active_is_cpu"] = info.active == "CPU"
        report["checks"]["marked_fell_back"] = bool(info.fell_back or info.fallback_reason)
        img = np.zeros((128, 128, 3), np.uint8); img[:] = (40, 60, 80)
        detect.detect_faces(img, engine=eng)
        report["checks"]["detect_on_cpu_ok"] = True
        st = dml_probe.read_status(); report["dml_status"] = st
        report["checks"]["status_file_records_failure"] = st.get("ok") is False
        report["ok"] = all(report["checks"].values())
        eng.close()
    except Exception as e:  # noqa: BLE001
        report["error"] = f"{type(e).__name__}: {e}"; report["traceback"] = traceback.format_exc()
    finally:
        os.environ.pop("FSP_FORCE_DML_FAIL", None)
        try:
            from . import dml_probe as _d
            _d.clear_status()
        except Exception:  # noqa: BLE001
            pass
    out = Path(args.out or "."); out.mkdir(parents=True, exist_ok=True)
    (out / "dml_fallback_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str), flush=True)
    return 0 if report["ok"] else 1


def main(argv=None):
    try:
        code = _main(argv)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 2)
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        try:
            from . import crashlog
            crashlog.install(); crashlog.record_current(where="main")
        except Exception:  # noqa: BLE001
            pass
        code = 1
    logging.shutdown()
    try:
        sys.stdout and sys.stdout.flush(); sys.stderr and sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code or 0)


def _main(argv=None):
    p = argparse.ArgumentParser(prog="FaceSwapPhoto", description=f"Face Swap Photo {__version__}")
    p.add_argument("--version", action="store_true")
    p.add_argument("--swap", metavar="TARGET", help="headless: photo to change")
    p.add_argument("--faces", nargs="+", metavar="PHOTO", help="face photo(s), faces taken left->right")
    p.add_argument("--target", help="GUI/selftest: photo to change")
    p.add_argument("--out"); p.add_argument("--before-after")
    p.add_argument("--map", help="who becomes who, e.g. '1,0' or '0,-' (target faces left->right)")
    p.add_argument("--flip", action="store_true")
    p.add_argument("--format", default=None, choices=["jpg", "png"])
    p.add_argument("--jpeg-quality", type=int, default=97)
    p.add_argument("--enhance", default="gpen512", choices=["off", "gpen256", "gpen512"])
    p.add_argument("--no-color-match", action="store_false", dest="color_match")
    p.add_argument("--no-detail", action="store_false", dest="detail")
    p.add_argument("--seamless", action="store_true")
    p.add_argument("--models"); p.add_argument("--device", default="auto", choices=["auto", "dml", "cpu"])
    p.add_argument("--selftest", action="store_true"); p.add_argument("--dml-smoke", action="store_true")
    p.add_argument("--dml-probe", metavar="MODEL"); p.add_argument("--test-dml-fallback", action="store_true")
    p.add_argument("--auto-swap", action="store_true", help="GUI: run the swap through the real window")
    p.add_argument("--auto-save", metavar="PATH", help="GUI: save the result through the real window")
    p.add_argument("--shots-dir", metavar="DIR", help="GUI: grab window screenshots + layout report")
    p.add_argument("--screen-grab", metavar="PNG", help="GUI: grab the whole screen at the end")
    p.add_argument("--hold", type=float, default=0, help="GUI autorun: keep the window open N seconds at the end")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(color_match=True, detail=True)
    args, _ = p.parse_known_args(argv)
    if args.dml_probe:
        from . import dml_probe
        return dml_probe.run_probe_in_this_process(args.dml_probe)
    if args.version:
        print(json.dumps(versions(), indent=2)); return 0
    headless = args.selftest or args.swap or args.test_dml_fallback
    _setup_logging(verbose=bool(headless) and not args.quiet)
    if args.test_dml_fallback:
        return test_dml_fallback(args)
    if args.selftest:
        if not args.target or not args.faces:
            p.error("--selftest needs --target and --faces")
        return selftest(args)
    if args.swap:
        if not args.faces or not args.out:
            p.error("--swap needs --faces and --out")
        return swap_headless(args)
    from .gui.app import run_gui
    return run_gui(args)
