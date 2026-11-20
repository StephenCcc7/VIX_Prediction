"""Summarize overlapping rv5 observations before adding another data source."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_series(path, symbol):
    frame = pd.read_csv(path)
    if "Symbol" in frame:
        frame = frame.loc[frame.Symbol.eq(symbol)].copy()
    frame["date"] = pd.to_datetime(frame.date.str[:10], format="%Y-%m-%d")
    if frame.date.duplicated().any():
        raise ValueError(f"Duplicate session dates: {path}")
    if not (np.isfinite(frame.rv5).all() and frame.rv5.gt(0).all()):
        raise ValueError(f"rv5 must contain positive finite variances: {path}")
    return frame.set_index("date")[["rv5"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extension", help="CSV with date, rv5; optional Symbol")
    parser.add_argument("--archive", default="data/raw/MAN_data.csv")
    parser.add_argument("--symbol", default=".SPX")
    parser.add_argument("--output", default="data/overlap.csv")
    args = parser.parse_args()
    old = read_series(args.archive, args.symbol)
    new = read_series(args.extension, args.symbol)
    overlap = old.join(new, how="inner", lsuffix="_archive", rsuffix="_extension")
    if overlap.empty:
        raise ValueError("No overlapping dates for source comparison.")
    overlap["ratio"] = overlap.rv5_extension / overlap.rv5_archive
    overlap["log_difference"] = np.log(overlap.ratio)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(target)
    print(f"Overlapping dates: {len(overlap)}")
    print(overlap[["ratio", "log_difference"]].describe().to_string())
    print(f"Log-variance correlation: {np.log(overlap.rv5_archive).corr(np.log(overlap.rv5_extension)):.6f}")
    print(f"Per-date comparison: {target.resolve()}")
    print("Confirm five-minute estimator, open-to-close session boundaries, and variance units from source metadata before merging.")


if __name__ == "__main__":
    main()
