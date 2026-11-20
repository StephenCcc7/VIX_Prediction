"""Explain the saved experiment with calibration decomposition and small controls.

Runs independently of run.py; the original experiment outputs stay in results/.
All additional outputs are written to results/diagnostics/.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/diagnostics/matplotlib").resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch

from volatility.data import make_windows, read_data
from volatility.models import (
    HARNet20, fit_neural, initializer_predict, level_initializer,
    neural_predict, qlike,
)


def saved_result_analysis(output):
    pred = pd.read_csv("results/predictions.csv", parse_dates=["date", "origin"])
    calibration = pd.read_csv("results/calibration.csv")
    rows = []
    for year, frame in pred.groupby(pred.date.dt.year):
        for model in ["Log-HAR", "HARNet20", "HAR-initializer", "XGBoost"]:
            raw = frame[f"{model}_raw"]
            c = float(calibration.loc[calibration.year.eq(year) & calibration.model.eq(model), "c"].iloc[0])
            oracle = float((frame.actual / raw).mean())
            raw_loss = float(qlike(frame.actual, raw).mean())
            calibrated_loss = float(qlike(frame.actual, c * raw).mean())
            rows.append(dict(
                year=year, model=model, n=len(frame), applied_c=c,
                same_year_optimal_c_descriptive=oracle,
                raw_qlike=raw_loss, calibrated_qlike=calibrated_loss,
                calibration_loss_change=calibrated_loss - raw_loss,
                exact_change_from_formula=oracle * (1 / c - 1) + np.log(c),
                oracle_qlike_descriptive=float(qlike(frame.actual, oracle * raw).mean()),
                mean_actual=float(frame.actual.mean()), mean_prediction=float(raw.mean()),
            ))
    effects = pd.DataFrame(rows)
    effects.to_csv(output / "calibration_decomposition.csv", index=False)
    daily = pred[["date", "actual", "origin_variance", "full_session"]].copy()
    for variant in ["raw", "calibrated"]:
        daily[f"difference_{variant}"] = (
            qlike(pred.actual, pred[f"HARNet20_{variant}"])
            - qlike(pred.actual, pred[f"Log-HAR_{variant}"])
        )
    daily["calibration_change_in_difference"] = daily.difference_calibrated - daily.difference_raw
    daily.sort_values("calibration_change_in_difference", ascending=False).to_csv(output / "daily_loss_attribution.csv", index=False)
    monthly = daily.groupby(daily.date.dt.to_period("M"))[
        ["difference_raw", "difference_calibrated", "calibration_change_in_difference"]
    ].agg(["sum", "mean", "size"])
    monthly.to_csv(output / "monthly_attribution.csv")
    curves = pd.read_csv("results/learning_curves.csv")
    curve_rows = []
    for (model, year, seed), group in curves.groupby(["model", "year", "seed"]):
        best = group.loc[group.validation_qlike.idxmin()]
        curve_rows.append(dict(
            model=model, year=year, seed=seed, epochs_run=int(group.epoch.max()),
            best_epoch=int(best.epoch), initial_qlike=float(group.iloc[0].validation_qlike),
            best_qlike=float(best.validation_qlike), last_qlike=float(group.iloc[-1].validation_qlike),
        ))
    pd.DataFrame(curve_rows).to_csv(output / "original_training_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    effects.loc[effects.model.isin(["Log-HAR", "HARNet20"])].pivot(
        index="year", columns="model", values="calibration_loss_change"
    ).plot.bar(ax=axes[0], rot=0)
    axes[0].axhline(0, lw=0.7, color="0.4")
    axes[0].set(ylabel="Calibrated minus raw QLIKE", title="Effect of previous-year calibration")
    for variant in ["raw", "calibrated"]:
        axes[1].plot(daily.date, daily[f"difference_{variant}"].cumsum(), label=variant)
    axes[1].axhline(0, lw=0.7, color="0.4")
    axes[1].set(ylabel="Cumulative QLIKE difference", title="HARNet20 minus Log-HAR")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].legend()
    fig.savefig(output / "calibration_diagnosis.png", dpi=160)
    plt.close(fig)
    print(effects.to_string(index=False), flush=True)
    return pred


def head_only_fit(x, y, config, seed, epochs):
    """Same QLIKE/Adam/batches, but freeze both convolution filters at averages."""
    torch.manual_seed(seed)
    scale, beta = level_initializer(x, y)
    model = HARNet20(beta, config["floor"])
    model.week.weight.requires_grad_(False)
    model.month.weight.requires_grad_(False)
    tx = torch.as_tensor(x / scale, dtype=torch.float64)
    ty = torch.as_tensor(y / scale, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=config["nn_learning_rate"], weight_decay=0)
    rng = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for batch in torch.randperm(len(tx), generator=rng).split(config["batch_size"]):
            log_ratio = ty[batch].log() - model(tx[batch]).log()
            loss = (torch.expm1(log_ratio) - log_ratio).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model, scale


def implementation_and_controls(x, y, meta, config, selected, saved, output):
    years = meta.date.dt.year.to_numpy()
    seed, epochs = config["seeds"][0], selected["nn_epochs_qlike"]
    diagnostics, controls, weights, checks = [], [], [], []
    neural_label = f"HARNet20_seed{seed}"
    for year in [2016] + config["test_years"]:
        train, forecast = years < year, years == year
        scale, beta = level_initializer(x[train], y[train])
        initial = HARNet20(beta, config["floor"])
        exact, _ = initializer_predict(x, scale, beta, config["floor"])
        initial_pred, _ = neural_predict(initial, scale, x)
        checks.append(dict(
            year=year, parameters=sum(p.numel() for p in initial.parameters()),
            max_initializer_difference=float(np.max(np.abs(exact - initial_pred))),
            largest_training_target=meta.loc[train, "date"].max().strftime("%Y-%m-%d"),
            earliest_forecast_target=meta.loc[forecast, "date"].min().strftime("%Y-%m-%d"),
        ))
        model, scale, _, _ = fit_neural(x[train], y[train], seed, "qlike", epochs, config)
        pred, _ = neural_predict(model, scale, x[forecast])
        if year == 2016:
            reference = pd.read_csv("results/calibration_2016_forecasts.csv")
        else:
            reference = saved.loc[saved.date.dt.year.eq(year)]
        reproduction_error = float(np.max(np.abs(pred - reference[f"HARNet20_seed{seed}_raw"].to_numpy())))
        with torch.no_grad():
            tensor = torch.as_tensor(x / scale, dtype=torch.float64)
            week_pre = model.week(tensor[:, None, :])
            month_pre = model.month(torch.relu(week_pre))
            week = model.week.weight.numpy().ravel()
            month = model.month.weight.numpy().ravel()
            head = model.head.weight.numpy().ravel()
            # When both ReLUs are active the whole network is affine in its 20 inputs.
            effective = head[2] * np.kron(month, week)
            effective[-5:] += head[1] * week
            effective[-1] += head[0]
            linear_prediction = scale * model.head.bias.item() + x @ effective
            actual_prediction, _ = neural_predict(model, scale, x)
        diagnostics.append(dict(
            year=year, seed=seed, reproduction_max_error=reproduction_error,
            week_nonpositive_activations=int((week_pre <= 0).sum()),
            month_nonpositive_activations=int((month_pre <= 0).sum()),
            all_window_affine_max_error=float(np.max(np.abs(linear_prediction - actual_prediction))),
            intercept_source_units=scale * model.head.bias.item(),
            fitting_mean_variance=scale, minimum_week_weight=float(week.min()), minimum_month_weight=float(month.min()),
            raw_forecast_qlike=float(qlike(y[forecast], pred).mean()),
            initial_forecast_qlike=float(qlike(y[forecast], exact[forecast]).mean()),
        ))
        for lag, weight in enumerate(effective[::-1], start=1):
            weights.append(dict(year=year, lag=lag, effective_weight=float(weight)))
        head_model, head_scale = head_only_fit(x[train], y[train], config, seed, epochs)
        head_prediction, _ = neural_predict(head_model, head_scale, x[forecast])
        frame = meta.loc[forecast, ["date", "actual"]].copy()
        frame["head_only_raw"] = head_prediction
        frame[f"{neural_label}_raw"] = pred
        controls.append(frame)
        print(f"Control {year}: reproduced max error {reproduction_error:.3g}; HARNet={qlike(y[forecast],pred).mean():.6f}; head-only={qlike(y[forecast],head_prediction).mean():.6f}", flush=True)
    pd.DataFrame(checks).to_csv(output / "implementation_checks.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "network_diagnostics.csv", index=False)
    pd.DataFrame(weights).to_csv(output / "effective_lag_weights.csv", index=False)
    rows = []
    for position, frame in enumerate(controls):
        year = int(frame.date.dt.year.iloc[0])
        for model in ["head_only", neural_label]:
            pred = frame[f"{model}_raw"]
            rows.append(dict(year=year, model=model, variant="raw", n=len(frame), qlike=float(qlike(frame.actual, pred).mean())))
            if position:
                previous = controls[position - 1]
                c = float((previous.actual / previous[f"{model}_raw"]).mean())
                frame[f"{model}_calibrated"] = c * pred
                rows.append(dict(year=year, model=model, variant="calibrated", n=len(frame), qlike=float(qlike(frame.actual, c * pred).mean())))
    pd.concat(controls, ignore_index=True).to_csv(output / "head_only_predictions.csv", index=False)
    score_frame = pd.DataFrame(rows)
    score_frame.to_csv(output / "head_only_scores.csv", index=False)
    summary = []
    test_scores = score_frame.loc[score_frame.year.isin(config["test_years"])]
    for (model, variant), group in test_scores.groupby(["model", "variant"]):
        summary.append(dict(model=model, variant=variant, seed=seed, n=int(group.n.sum()),
                            qlike=float(np.average(group.qlike, weights=group.n))))
    pd.DataFrame(summary).to_csv(output / "head_only_summary.csv", index=False)


def extend_validation(x, y, meta, config, output):
    seed = config["seeds"][0]
    years = meta.date.dt.year.to_numpy()
    rows, curves = [], []
    for year in config["validation_years"]:
        train, val = years < year, years == year
        model, scale, best, history = fit_neural(
            x[train], y[train], seed, "qlike", 600, config,
            validation=(x[val], y[val]),
        )
        history_frame = pd.DataFrame(history)
        original_budget = history_frame.loc[history_frame.epoch.le(200), "validation_qlike"].min()
        extended = float(qlike(y[val], neural_predict(model, scale, x[val])[0]).mean())
        rows.append(dict(
            year=year, seed=seed, epoch_cap=600, patience=config["nn_patience"],
            epochs_run=history_frame.epoch.max(), best_epoch=best,
            best_within_200=original_budget, best_extended=extended,
            relative_improvement=1 - extended / original_budget,
        ))
        curves.extend(dict(year=year, seed=seed, **row) for row in history)
        print(f"Extended validation {year}: best epoch {best}; QLIKE {original_budget:.6f} -> {extended:.6f}", flush=True)
    pd.DataFrame(rows).to_csv(output / "extended_validation.csv", index=False)
    pd.DataFrame(curves).to_csv(output / "extended_validation_curves.csv", index=False)


def main():
    config = json.loads(Path("config.json").read_text(encoding="utf-8"))
    selected = json.loads(Path("results/selected_settings.json").read_text(encoding="utf-8"))
    output = Path("results/diagnostics")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["threads"])
    saved = saved_result_analysis(output)
    grid, _ = read_data(config, output)
    x, y, meta = make_windows(grid)
    implementation_and_controls(x, y, meta, config, selected, saved, output)
    extend_validation(x, y, meta, config, output)
    print(f"Diagnostics saved to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
