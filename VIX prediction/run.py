"""Run the proposal's experiment: python run.py."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import pandas as pd
import torch

from volatility.data import download_archive, make_windows, read_data
from volatility.evaluation import block_inference, plots, scores, write_report
from volatility.experiment import annual_forecasts, select_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--download", action="store_true", help="Download the proposal's public archive if absent")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    torch.set_num_threads(config["threads"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if args.download and not Path(config["data_path"]).exists():
        download_archive(config["data_path"])
    start = perf_counter()
    grid, audit = read_data(config, output)
    x, y, meta = make_windows(grid)
    counts = meta.groupby(meta.date.dt.year).size()
    audit["complete_windows"] = counts.reindex(audit.index, fill_value=0)
    audit["excluded_windows"] = audit.scheduled - audit.complete_windows
    audit.to_csv(output / "data_audit.csv")
    print(f"Observed sessions: {grid.rv5.count()}; missing: {grid.missing.sum()}; complete windows: {len(y)}", flush=True)
    selected = select_settings(x, y, meta, config, output)
    predictions = annual_forecasts(x, y, meta, config, selected, output)
    threshold = float(grid.loc[:"2016-12-31", "rv5"].quantile(0.75))
    score_table = scores(predictions, threshold, config, output)
    inference = block_inference(predictions, grid, config, output)
    plots(predictions, score_table, output)
    runtime = dict(
        total_seconds=perf_counter() - start, platform=platform.platform(),
        fitting_seconds=float(
            pd.read_csv(output / "validation.csv").fit_seconds.sum()
            + pd.read_csv(output / "fitting_times.csv").fit_seconds.sum()
        ),
        processor=platform.processor(), logical_cpus=os.cpu_count(),
        python=platform.python_version(), torch_threads=torch.get_num_threads(),
        high_volatility_threshold=threshold, config=config,
        packages={name: importlib.metadata.version(name) for name in [
            "numpy", "pandas", "scipy", "matplotlib", "scikit-learn", "xgboost", "torch", "exchange-calendars",
        ]},
    )
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    write_report(config, audit, score_table, inference, runtime, output)
    print(inference.loc[inference.role.eq("primary")].to_string(index=False), flush=True)
    print(f"Completed in {runtime['total_seconds']:.1f}s. Results: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
