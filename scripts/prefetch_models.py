"""CI helper: download / verify models into FSP_MODELS (or --models dir). usage: prefetch_models.py DIR [light] [hq]"""
import sys, time
sys.path.insert(0, '.')
from pathlib import Path
from fsp.models import ModelStore, REQUIRED, ENHANCER_LIGHT, ENHANCER_HQ

d = Path(sys.argv[1])
want = list(REQUIRED)
if 'light' in sys.argv: want.append(ENHANCER_LIGHT)
if 'hq' in sys.argv: want.append(ENHANCER_HQ)
s = ModelStore(d, shared=[])
last = [0.0]
def prog(f, done, total, bps, verifying):
    if time.time() - last[0] > 5 or done == total:
        last[0] = time.time()
        print(f"{f}: {done / 1e6:.0f}/{total / 1e6:.0f} MB {'verifying' if verifying else f'{bps / 1e6:.1f} MB/s'}", flush=True)
t0 = time.time()
s.ensure(want, progress=prog)
print('models ready in', s.dir, f'{time.time() - t0:.0f}s', [(m.file, s.source_of(m)) for m in want])
