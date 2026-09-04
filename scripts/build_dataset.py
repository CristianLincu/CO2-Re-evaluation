"""Download and cache one year of training data."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import CACHE_DIR
from pipeline.data import build_training_frame

if __name__ == "__main__":
    frame = build_training_frame(years=1.0)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / "training_frame.parquet"
    frame.to_parquet(out, index=False)
    print(f"\nWrote {out}")
    print(f"rows={len(frame)} cols={len(frame.columns)}")
    print(f"range: {frame['timestamp'].min()} -> {frame['timestamp'].max()}")
    print("nulls:\n", frame.isna().sum()[lambda s: s > 0].to_string())
