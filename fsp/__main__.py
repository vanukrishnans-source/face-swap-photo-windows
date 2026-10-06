import os, sys
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")
try:
    from fsp.crashlog import install as _install_crashlog
    _install_crashlog()
except Exception:
    pass
from fsp.cli import main
raise SystemExit(main())
