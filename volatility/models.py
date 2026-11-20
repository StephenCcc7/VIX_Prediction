"""Log-HAR, log-link QLIKE boosting, and the 13-parameter HARNet20."""

import numpy as np
import torch
from torch import nn
import xgboost as xgb

from .data import har_features, tree_features


def qlike(y, prediction):
    log_ratio = np.log(y) - np.log(prediction)
    return np.expm1(log_ratio) - log_ratio


def log_har(x, y, forecast_x):
    beta = np.linalg.lstsq(har_features(np.log(x)), np.log(y), rcond=None)[0]
    return np.exp(har_features(np.log(forecast_x)) @ beta)


def level_initializer(x, y):
    scale = float(y.mean())
    beta = np.linalg.lstsq(har_features(x / scale), y / scale, rcond=None)[0]
    return scale, beta


def initializer_predict(x, scale, beta, floor):
    raw = har_features(x / scale) @ beta
    return scale * np.maximum(raw, floor), int((raw <= floor).sum())


def qlike_objective(margin, data):
    # L(y, exp(eta)) = y exp(-eta) + eta - log(y) - 1.
    ratio = data.get_label() * np.exp(-margin)
    return 1.0 - ratio, ratio


def qlike_metric(margin, data):
    log_ratio = np.log(data.get_label()) - margin
    return "qlike", float(np.mean(np.expm1(log_ratio) - log_ratio))


def fit_tree(x, y, settings, rounds, config, validation=None):
    scale = y.mean()
    train = xgb.DMatrix(tree_features(x), label=y / scale)
    params = dict(
        settings, subsample=0.8, reg_lambda=1.0, base_score=0.0,
        disable_default_eval_metric=1, tree_method="hist", seed=451,
        nthread=config["threads"],
    )
    kwargs = {}
    if validation is not None:
        vx, vy = validation
        kwargs = dict(
            evals=[(xgb.DMatrix(tree_features(vx), label=vy / scale), "validation")],
            early_stopping_rounds=config["xgb_patience"],
        )
    model = xgb.train(
        params, train, num_boost_round=rounds, obj=qlike_objective,
        custom_metric=qlike_metric, verbose_eval=False, **kwargs,
    )
    return model, float(scale)


def tree_predict(model, scale, x, rounds=None):
    stop = rounds if rounds is not None else model.num_boosted_rounds()
    eta = model.predict(xgb.DMatrix(tree_features(x)), output_margin=True, iteration_range=(0, stop))
    return scale * np.exp(eta.astype(float))


class HARNet20(nn.Module):
    def __init__(self, beta, floor):
        super().__init__()
        self.week = nn.Conv1d(1, 1, kernel_size=5, bias=False)
        self.month = nn.Conv1d(1, 1, kernel_size=4, dilation=5, bias=False)
        self.head = nn.Linear(3, 1)
        self.floor = floor
        self.double()
        with torch.no_grad():
            self.week.weight.fill_(1 / 5)
            self.month.weight.fill_(1 / 4)
            self.head.bias.copy_(torch.as_tensor(beta[:1]))
            self.head.weight.copy_(torch.as_tensor(beta[1:]).reshape(1, 3))

    def unfloored(self, x):
        weekly = torch.relu(self.week(x[:, None, :]))
        monthly = torch.relu(self.month(weekly))
        features = torch.stack([x[:, -1], weekly[:, 0, -1], monthly[:, 0, -1]], dim=1)
        return self.head(features).squeeze(1)

    def forward(self, x):
        return self.unfloored(x).clamp_min(self.floor)


def neural_predict(model, scale, x):
    with torch.no_grad():
        raw = model.unfloored(torch.as_tensor(x / scale, dtype=torch.float64))
        return scale * raw.clamp_min(model.floor).numpy(), int((raw <= model.floor).sum())


def fit_neural(x, y, seed, loss_name, epochs, config, validation=None):
    torch.manual_seed(seed)
    scale, beta = level_initializer(x, y)
    model = HARNet20(beta, config["floor"])
    tx = torch.as_tensor(x / scale, dtype=torch.float64)
    ty = torch.as_tensor(y / scale, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["nn_learning_rate"], weight_decay=0)
    rng = torch.Generator().manual_seed(seed)
    best_score, best_epoch, stale = float("inf"), 0, 0
    history = []
    # Epoch zero allows validation to retain the exact OLS initializer.
    if validation is not None:
        vx, vy = validation
        pred, floors = neural_predict(model, scale, vx)
        best_score = float(qlike(vy, pred).mean())
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        history.append(dict(epoch=0, validation_qlike=best_score, validation_floors=floors))
    for epoch in range(1, epochs + 1):
        for batch in torch.randperm(len(tx), generator=rng).split(config["batch_size"]):
            pred = model(tx[batch])
            if loss_name == "qlike":
                log_ratio = ty[batch].log() - pred.log()
                loss = (torch.expm1(log_ratio) - log_ratio).mean()
            else:
                loss = (ty[batch].log() - pred.log()).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if validation is not None:
            pred, floors = neural_predict(model, scale, vx)
            score = float(qlike(vy, pred).mean())
            history.append(dict(epoch=epoch, validation_qlike=score, validation_floors=floors))
            if score < best_score:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= config["nn_patience"]:
                break
    if validation is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
    return model, scale, best_epoch, history
