"""Download the 6000x4000 CC0 couple photo (Wikimedia Commons) for the full-resolution CI test."""
import hashlib, sys, urllib.request
from pathlib import Path
URL = ("https://upload.wikimedia.org/wikipedia/commons/6/60/"
       "Portrait_of_beautiful_young_couple_enjoying_nature_on_a_mountain_background._%2846867669051%29.jpg")
SHA = "f4dbaa3d42d5714bbec1bc70ad12e5a9de0772ff46f053edb937cb86bf44b750"
out = Path(sys.argv[1] if len(sys.argv) > 1 else "clips/target_couple_cc0_6000x4000.jpg")
out.parent.mkdir(parents=True, exist_ok=True)
if not out.is_file() or hashlib.sha256(out.read_bytes()).hexdigest() != SHA:
    req = urllib.request.Request(URL, headers={"User-Agent": "FaceSwapPhoto-CI/1.0 (github.com/vanukrishnans-source)"})
    out.write_bytes(urllib.request.urlopen(req, timeout=120).read())
d = hashlib.sha256(out.read_bytes()).hexdigest()
assert d == SHA, f"sha mismatch {d}"
print("ok", out, out.stat().st_size)
