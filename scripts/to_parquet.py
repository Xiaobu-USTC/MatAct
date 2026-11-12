#!/usr/bin/env python3
"""Convert data/rl/*.jsonl to Parquet for verl (requires pandas and pyarrow)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", default=["data/rl/train.jsonl", "data/rl/valid.jsonl"])
    args = ap.parse_args()
    import pandas as pd

    for name in args.files:
        path = Path(name)
        with path.open(encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        out = path.with_suffix(".parquet")
        pd.DataFrame(rows).to_parquet(out, index=False)
        print(f"{path} -> {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
