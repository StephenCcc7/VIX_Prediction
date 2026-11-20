"""Score simple causal baselines on the saved forecast dates; no model fitting."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from volatility.data import make_windows
from volatility.models import qlike


def main():
    config = json.loads(Path("config.json").read_text(encoding="utf-8"))
    output = Path("results/baselines")
    output.mkdir(exist_ok=True)
    grid = pd.read_csv("results/session_audit.csv", index_col="date", parse_dates=True)
    x, y, meta = make_windows(grid)
    meta = meta.set_index("date")
    baseline = pd.DataFrame(index=meta.index)
    baseline["actual"] = y
    baseline["Persistence_raw"] = x[:, -1]
    baseline["Mean5_raw"] = x[:, -5:].mean(axis=1)
    baseline["Mean20_raw"] = x.mean(axis=1)
    for year in [2016] + config["test_years"]:
        baseline.loc[str(year), "HistoricalMean_raw"] = y[meta.index.year < year].mean()
    names = ["Persistence", "Mean5", "Mean20", "HistoricalMean"]
    coefficients = []
    for year in config["test_years"]:
        previous = baseline.loc[str(year-1)]
        for model in names:
            c = float((previous.actual / previous[f"{model}_raw"]).mean())
            baseline.loc[str(year), f"{model}_calibrated"] = c * baseline.loc[str(year), f"{model}_raw"]
            coefficients.append(dict(year=year, model=model, c=c))
    predictions = pd.read_csv("results/predictions.csv", parse_dates=["date"], float_precision="round_trip").set_index("date")
    for model in names:
        for variant in ["raw", "calibrated"]:
            predictions[f"{model}_{variant}"] = baseline[f"{model}_{variant}"].reindex(predictions.index)
    names += ["Log-HAR", "XGBoost", "HARNet20"]
    rows = []
    for model in names:
        for variant in ["raw", "calibrated"]:
            prediction = predictions[f"{model}_{variant}"]
            ratio = prediction / predictions.actual
            rows.append(dict(
                model=model, variant=variant, n=len(predictions),
                qlike=float(qlike(predictions.actual, prediction).mean()),
                log_rmse=float(np.sqrt(np.mean(np.log(ratio) ** 2))),
                within_20_percent=float(ratio.between(0.8, 1.2).mean()),
                within_50_percent=float(ratio.between(0.5, 1.5).mean()),
                median_absolute_percentage_error=float(np.median(np.abs(ratio - 1))),
            ))
    scores = pd.DataFrame(rows)
    scores.to_csv(output / "scores.csv", index=False)
    pd.DataFrame(coefficients).to_csv(output / "calibration.csv", index=False)
    predictions.to_csv(output / "predictions.csv")
    keys = [f"{model}_{variant}" for variant in ["raw", "calibrated"] for model in names]
    losses = pd.DataFrame({key: qlike(predictions.actual, predictions[key]) for key in keys}).reindex(grid.index)
    samples, block = config["bootstrap_samples"], 20
    rng = np.random.default_rng(config["bootstrap_seed"] + block)
    sums = np.zeros((samples, len(keys)))
    counts = np.zeros_like(sums)
    for year in config["test_years"]:
        values = losses.loc[str(year)].to_numpy()
        n = len(values)
        for start in range(0, samples, 100):
            stop = min(start+100, samples)
            blocks = rng.integers(0, n-block+1, size=(stop-start, int(np.ceil(n/block))))
            indices = (blocks[..., None] + np.arange(block)).reshape(stop-start, -1)[:, :n]
            sampled = values[indices]
            sums[start:stop] += np.nansum(sampled, axis=1)
            counts[start:stop] += np.isfinite(sampled).sum(axis=1)
    means = sums / counts
    point = losses.mean().to_numpy()
    rows = []
    for variant in ["raw", "calibrated"]:
        for model in ["Log-HAR", "HARNet20"]:
            for reference in ["Persistence", "Mean5", "Mean20", "HistoricalMean"]:
                a, b = keys.index(f"{model}_{variant}"), keys.index(f"{reference}_{variant}")
                interval = np.quantile(means[:, a] / means[:, b], [0.025, 0.975])
                rows.append(dict(
                    model=model, reference=reference, variant=variant, n=len(predictions),
                    loss_ratio=point[a]/point[b], improvement=1-point[a]/point[b],
                    ratio_lower=interval[0], ratio_upper=interval[1], samples=samples, block_length=block,
                ))
    inference = pd.DataFrame(rows)
    inference.to_csv(output / "inference.csv", index=False)
    print(scores.to_string(index=False))
    print(inference.to_string(index=False))


if __name__ == "__main__":
    main()
