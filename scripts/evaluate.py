from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from evaluation import metrics as _metrics

if __name__ == "__main__":
    _metrics.main()
