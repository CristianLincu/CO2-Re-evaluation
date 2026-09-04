"""Derive the operating envelope and ramp limits from observed behaviour."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

import pandas as pd

from pipeline.config import CACHE_DIR, DECISION_COLUMNS, MODELS_DIR, TECHNICAL_LIMITS
from pipeline.optimize import derive_limits


def main():
    frame = pd.read_parquet(CACHE_DIR / "training_frame.parquet")
    lower, upper, ramp = derive_limits(frame, technical=TECHNICAL_LIMITS)

    payload = {
        "lower": {c: float(v) for c, v in zip(DECISION_COLUMNS, lower)},
        "upper": {c: float(v) for c, v in zip(DECISION_COLUMNS, upper)},
        "ramp": {c: float(v) for c, v in zip(DECISION_COLUMNS, ramp)},
        "note": (
            "Bounds are the observed 0.1/99.9 percentiles capped by technical "
            "transfer capacity. Ramp limits are the 99th percentile of realised "
            "15-minute changes over the training year."
        ),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "operating_limits.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    table = pd.DataFrame(
        {"lower": lower, "upper": upper, "ramp_per_15min": ramp}, index=DECISION_COLUMNS
    )
    print(table.to_string(float_format=lambda v: f"{v:9.1f}"))
    print(f"\nsaved {MODELS_DIR / 'operating_limits.json'}")


if __name__ == "__main__":
    main()
