"""Read the archive and keep missing sessions in the calendar grid."""

from pathlib import Path
from urllib.request import urlretrieve

import exchange_calendars as xc
import numpy as np
import pandas as pd

ARCHIVE_URL = (
    "https://raw.githubusercontent.com/mdsunivie/HARNet/"
    "3b2832f163ec52cab2435163e82f2e816ae348c9/data/MAN_data.csv"
)


def download_archive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(ARCHIVE_URL, path)
    return path


def read_data(config, output):
    raw = pd.read_csv(config["data_path"])
    # Session labels are written dates; converting timezone would shift some dates.
    raw["date"] = pd.to_datetime(raw["date"].str[:10], format="%Y-%m-%d")
    selected = raw.loc[
        raw.Symbol.eq(config["symbol"])
        & raw.date.between(config["start"], config["end"]),
        ["date", "rv5", "nobs"],
    ].copy()
    if selected.empty:
        raise ValueError("No observations in the requested symbol/date range.")
    if selected.date.duplicated().any():
        raise ValueError("Duplicate session labels in input data.")
    values = selected[["rv5", "nobs"]].to_numpy(float)
    if not (np.isfinite(values).all() and (values > 0).all()):
        raise ValueError("rv5 and nobs must be finite and strictly positive.")
    if not np.equal(selected.nobs, np.floor(selected.nobs)).all():
        raise ValueError("nobs must be integer observation counts.")
    calendar = xc.get_calendar("XNYS", start=config["start"], end=config["end"])
    schedule = calendar.schedule.loc[config["start"]:config["end"]]
    invalid = selected.loc[~selected.date.isin(schedule.index)]
    if not invalid.empty:
        raise ValueError(f"Non-XNYS session labels: {invalid.date.dt.strftime('%Y-%m-%d').tolist()}")
    grid = selected.set_index("date").reindex(schedule.index)
    grid.index.name = "date"
    grid["minutes"] = (schedule["close"] - schedule["open"]).dt.total_seconds() / 60
    grid["full_session"] = grid.minutes.eq(390)
    grid["missing"] = grid.rv5.isna()
    # A descriptive sparse-session flag; threshold uses initial fitting data only.
    initial = grid.loc[:"2013-12-31"]
    rate = (initial.nobs / initial.minutes).median()
    grid["sparse"] = (grid.nobs / grid.minutes) < 0.25 * rate
    grid.to_csv(output / "session_audit.csv")
    annual = grid.assign(year=grid.index.year).groupby("year").agg(
        scheduled=("rv5", "size"), observed=("rv5", "count"),
        missing=("missing", "sum"), early_closes=("full_session", lambda x: (~x).sum()),
        sparse=("sparse", "sum"), nobs_min=("nobs", "min"),
        nobs_median=("nobs", "median"), nobs_max=("nobs", "max"),
    )
    return grid, annual


def make_windows(grid):
    windows = np.lib.stride_tricks.sliding_window_view(grid.rv5.to_numpy(), 21)
    complete = np.isfinite(windows).all(axis=1)
    indices = np.flatnonzero(complete) + 20
    values = windows[complete]
    meta = pd.DataFrame({
        "date": grid.index[indices], "origin": grid.index[indices - 1],
        "actual": values[:, -1], "origin_variance": values[:, -2],
        "full_session": grid.full_session.iloc[indices].to_numpy(),
        "sparse": grid["sparse"].iloc[indices].to_numpy(),
    })
    return values[:, :-1].copy(), values[:, -1].copy(), meta


def har_features(x):
    return np.column_stack([np.ones(len(x)), x[:, -1], x[:, -5:].mean(1), x.mean(1)])


def tree_features(x):
    z = np.log(x)
    return np.column_stack([z, har_features(z)[:, 1:]])
