import sys as _sys
from pathlib import Path as _Path
_d = str(_Path(__file__).parent)
if _d not in _sys.path:
    _sys.path.insert(0, _d)
