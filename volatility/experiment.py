"""Validation selects settings once; annual refits use only past targets."""

from itertools import product
import json
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from .models import (
    fit_neural, fit_tree, initializer_predict, level_initializer,
    log_har, neural_predict, qlike, tree_predict,
)

MODEL_NAMES = ["Log-HAR", "XGBoost", "HAR-initializer", "HARNet20", "HARNet20-log"]


def select_settings(x, y, meta, config, output):
    years = meta.date.dt.year.to_numpy()
    records, curves = [], []
    candidates = []
    for depth, rate, child in product([2, 3, 4], [0.03, 0.10], [1, 10]):
        setting = dict(max_depth=depth, eta=rate, min_child_weight=child)
        losses, rounds = [], []
        for year in config["validation_years"]:
            train, val = years < year, years == year
            start = perf_counter()
            model, scale = fit_tree(
                x[train], y[train], setting, config["xgb_max_rounds"], config,
                validation=(x[val], y[val]),
            )
            best = model.best_iteration + 1
            loss = qlike(y[val], tree_predict(model, scale, x[val], best))
            losses.extend(loss.tolist())
            rounds.append(best)
            records.append(dict(
                model="XGBoost", year=year, seed=451, **setting,
                best_steps=best, qlike=float(loss.mean()), n=len(loss),
                fit_seconds=perf_counter() - start,
            ))
        candidates.append(dict(**setting, rounds=int(np.median(rounds)), validation_qlike=float(np.mean(losses))))
        print(f"XGBoost {setting}: validation QLIKE={np.mean(losses):.5f}", flush=True)
    winner = min(candidates, key=lambda r: r["validation_qlike"])
    selected = {"xgboost": winner}
    for loss_name in ["qlike", "log"]:
        steps = []
        for year in config["validation_years"]:
            train, val = years < year, years == year
            for seed in config["seeds"]:
                start = perf_counter()
                model, scale, best, history = fit_neural(
                    x[train], y[train], seed, loss_name, config["nn_max_epochs"], config,
                    validation=(x[val], y[val]),
                )
                pred, floors = neural_predict(model, scale, x[val])
                steps.append(best)
                records.append(dict(
                    model=f"HARNet20-{loss_name}", year=year, seed=seed,
                    best_steps=best, qlike=float(qlike(y[val], pred).mean()),
                    n=len(pred), floor_count=floors, fit_seconds=perf_counter() - start,
                ))
                curves.extend(dict(model=f"HARNet20-{loss_name}", year=year, seed=seed, **row) for row in history)
                print(f"HARNet20 {loss_name}, validation {year}, seed {seed}: epoch {best}", flush=True)
        selected[f"nn_epochs_{loss_name}"] = int(np.median(steps))
    pd.DataFrame(records).to_csv(output / "validation.csv", index=False)
    pd.DataFrame(candidates).to_csv(output / "xgboost_candidates.csv", index=False)
    pd.DataFrame(curves).to_csv(output / "learning_curves.csv", index=False)
    (output / "selected_settings.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    return selected


def annual_forecasts(x, y, meta, config, selected, output):
    years = meta.date.dt.year.to_numpy()
    raw_years, timings, floors, coefficients = [], [], [], []
    # Refit 2016 using frozen settings to generate pre-test calibration forecasts.
    for year in [config["test_years"][0] - 1] + config["test_years"]:
        train, test = years < year, years == year
        if not test.any():
            raise ValueError(f"No complete windows in forecast year {year}.")
        frame = meta.loc[test].copy()
        current = {}
        start = perf_counter()
        current["Log-HAR"] = log_har(x[train], y[train], x[test])
        timings.append(dict(year=year, model="Log-HAR", seed="", fit_seconds=perf_counter() - start))
        start = perf_counter()
        scale, beta = level_initializer(x[train], y[train])
        current["HAR-initializer"], n_floor = initializer_predict(x[test], scale, beta, config["floor"])
        timings.append(dict(year=year, model="HAR-initializer", seed="", fit_seconds=perf_counter() - start))
        floors.append(dict(year=year, model="HAR-initializer", seed="", split="forecast", n=len(frame), floor_count=n_floor))
        setting = selected["xgboost"]
        start = perf_counter()
        model, tree_scale = fit_tree(
            x[train], y[train], {k: setting[k] for k in ["max_depth", "eta", "min_child_weight"]},
            setting["rounds"], config,
        )
        fit_seconds = perf_counter() - start
        current["XGBoost"] = tree_predict(model, tree_scale, x[test])
        timings.append(dict(year=year, model="XGBoost", seed=451, fit_seconds=fit_seconds))
        for loss_name, label in [("qlike", "HARNet20"), ("log", "HARNet20-log")]:
            seed_predictions = []
            for seed in config["seeds"]:
                start = perf_counter()
                model, nn_scale, _, _ = fit_neural(
                    x[train], y[train], seed, loss_name, selected[f"nn_epochs_{loss_name}"], config,
                )
                fit_seconds = perf_counter() - start
                pred, n_floor = neural_predict(model, nn_scale, x[test])
                _, train_floors = neural_predict(model, nn_scale, x[train])
                seed_predictions.append(pred)
                frame[f"{label}_seed{seed}_raw"] = pred
                timings.append(dict(year=year, model=label, seed=seed, fit_seconds=fit_seconds))
                floors.extend([
                    dict(year=year, model=label, seed=seed, split="forecast", n=len(frame), floor_count=n_floor),
                    dict(year=year, model=label, seed=seed, split="train_final", n=int(train.sum()), floor_count=train_floors),
                ])
            current[label] = np.mean(seed_predictions, axis=0)
        for name, pred in current.items():
            frame[f"{name}_raw"] = pred
            if year in config["test_years"]:
                # Always calibrate on preceding RAW forecasts, never calibrated ones.
                prev = raw_years[-1]
                c = float((prev.actual / prev[f"{name}_raw"]).mean())
                frame[f"{name}_calibrated"] = c * pred
                coefficients.append(dict(year=year, model=name, c=c, calibration_year=year - 1, n=len(prev)))
                if name in ["HARNet20", "HARNet20-log"]:
                    for seed in config["seeds"]:
                        col = f"{name}_seed{seed}"
                        seed_c = float((prev.actual / prev[f"{col}_raw"]).mean())
                        frame[f"{col}_calibrated"] = seed_c * frame[f"{col}_raw"]
                        coefficients.append(dict(year=year, model=col, c=seed_c, calibration_year=year - 1, n=len(prev)))
        raw_years.append(frame)
        print(f"Annual refit {year}: {train.sum()} fitting windows, {test.sum()} forecasts", flush=True)
    pd.DataFrame(timings).to_csv(output / "fitting_times.csv", index=False)
    pd.DataFrame(floors).to_csv(output / "floor_activations.csv", index=False)
    pd.DataFrame(coefficients).to_csv(output / "calibration.csv", index=False)
    raw_years[0].to_csv(output / "calibration_2016_forecasts.csv", index=False)
    predictions = pd.concat(raw_years[1:], ignore_index=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    return predictions
