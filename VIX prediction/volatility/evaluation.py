"""Scores, paired within-year moving-block inference, and research figures."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .experiment import MODEL_NAMES
from .models import qlike


def scores(predictions, threshold, config, output):
    rows, seeds = [], []
    groups = [("all", predictions)]
    groups += [(str(year), part) for year, part in predictions.groupby(predictions.date.dt.year)]
    groups += [
        ("high_volatility_origins", predictions.loc[predictions.origin_variance > threshold]),
        ("full_sessions", predictions.loc[predictions.full_session]),
    ]
    for subset, frame in groups:
        for model in MODEL_NAMES:
            for variant in ["raw", "calibrated"]:
                pred = frame[f"{model}_{variant}"].to_numpy()
                rows.append(dict(
                    subset=subset, model=model, variant=variant, n=len(frame),
                    qlike=float(qlike(frame.actual, pred).mean()),
                    log_rmse=float(np.sqrt(np.mean((np.log(frame.actual) - np.log(pred)) ** 2))),
                ))
        for model in ["HARNet20", "HARNet20-log"]:
            for seed in config["seeds"]:
                for variant in ["raw", "calibrated"]:
                    pred = frame[f"{model}_seed{seed}_{variant}"]
                    seeds.append(dict(
                        subset=subset, model=model, seed=seed, variant=variant, n=len(frame),
                        qlike=float(qlike(frame.actual, pred).mean()),
                        log_rmse=float(np.sqrt(np.mean((np.log(frame.actual) - np.log(pred)) ** 2))),
                    ))
    result = pd.DataFrame(rows)
    result.to_csv(output / "scores.csv", index=False)
    seed_frame = pd.DataFrame(seeds)
    seed_frame.to_csv(output / "seed_scores.csv", index=False)
    seed_frame.groupby(["subset", "model", "variant"])[["qlike", "log_rmse"]].agg(
        ["mean", "std", "min", "max"]
    ).to_csv(output / "seed_variation.csv")
    return result


def block_inference(predictions, grid, config, output):
    comparisons = [
        ("HARNet20", "Log-HAR", "primary"),
        ("XGBoost", "Log-HAR", "secondary"),
        ("HAR-initializer", "Log-HAR", "secondary"),
        ("HARNet20-log", "Log-HAR", "secondary"),
        ("HARNet20", "HAR-initializer", "secondary"),
        ("HARNet20", "HARNet20-log", "secondary"),
    ]
    keys = [f"{model}_{variant}" for variant in ["raw", "calibrated"] for model in MODEL_NAMES]
    losses = pd.DataFrame({key: qlike(predictions.actual, predictions[key]) for key in keys})
    losses.index = pd.DatetimeIndex(predictions.date)
    loss_grid = losses.reindex(grid.index)
    point = losses.mean().to_numpy()
    rows = []
    for block in config["block_lengths"]:
        rng = np.random.default_rng(config["bootstrap_seed"] + block)
        totals = np.zeros((config["bootstrap_samples"], len(keys)))
        counts = np.zeros_like(totals)
        for year in config["test_years"]:
            values = loss_grid.loc[str(year)].to_numpy()
            n = len(values)
            # Sample complete calendar blocks, carrying NaNs together with losses.
            nblocks = int(np.ceil(n / block))
            for start in range(0, config["bootstrap_samples"], 100):
                stop = min(start + 100, config["bootstrap_samples"])
                origins = rng.integers(0, n - block + 1, size=(stop - start, nblocks))
                idx = (origins[..., None] + np.arange(block)).reshape(stop - start, -1)[:, :n]
                sampled = values[idx]
                totals[start:stop] += np.nansum(sampled, axis=1)
                counts[start:stop] += np.isfinite(sampled).sum(axis=1)
        means = totals / counts
        # All contrasts, ratios, and differences use these same paired draws.
        for variant in ["raw", "calibrated"]:
            for candidate, reference, role in comparisons:
                a, b = keys.index(f"{candidate}_{variant}"), keys.index(f"{reference}_{variant}")
                ratio_ci = np.quantile(means[:, a] / means[:, b], [0.025, 0.975])
                diff_ci = np.quantile(means[:, a] - means[:, b], [0.025, 0.975])
                rows.append(dict(
                    candidate=candidate, reference=reference, variant=variant,
                    role=role if variant == "calibrated" else "secondary",
                    block_length=block, samples=config["bootstrap_samples"], n=len(predictions),
                    ratio=point[a] / point[b], ratio_lower=ratio_ci[0], ratio_upper=ratio_ci[1],
                    difference=point[a] - point[b], difference_lower=diff_ci[0], difference_upper=diff_ci[1],
                ))
    result = pd.DataFrame(rows)
    result.to_csv(output / "inference.csv", index=False)
    return result


def plots(predictions, scores_table, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, layout="constrained")
    axes[0].plot(predictions.date, predictions.actual, color="0.65", lw=0.8, label="Observed rv5")
    for model in ["Log-HAR", "XGBoost", "HARNet20"]:
        axes[0].plot(predictions.date, predictions[f"{model}_calibrated"], lw=0.9, label=model)
    axes[0].set(yscale="log", ylabel="Realized variance (source units)", title="Next-session variance: calibrated forecasts")
    axes[0].legend(ncol=4, fontsize=9)
    baseline = qlike(predictions.actual, predictions["Log-HAR_calibrated"])
    for model in ["HARNet20", "XGBoost", "HARNet20-log"]:
        difference = qlike(predictions.actual, predictions[f"{model}_calibrated"]) - baseline
        axes[1].plot(predictions.date, difference.cumsum(), label=model)
    axes[1].axhline(0, color="0.4", lw=0.7)
    axes[1].set(ylabel="Cumulative QLIKE difference", xlabel="Target session", title="Model minus Log-HAR; lower values favor the model")
    axes[1].legend()
    fig.savefig(output / "forecasts.png", dpi=180)
    plt.close(fig)
    annual = scores_table.loc[scores_table.subset.str.fullmatch(r"\d{4}") & scores_table.variant.eq("calibrated")]
    table = annual.pivot(index="subset", columns="model", values="qlike").reindex(columns=MODEL_NAMES)
    ax = table.plot.bar(figsize=(10, 5), rot=0)
    ax.set(xlabel="Forecast year", ylabel="Mean QLIKE", title="Annual calibrated forecast accuracy")
    ax.legend(fontsize=8)
    ax.figure.tight_layout()
    ax.figure.savefig(output / "annual_qlike.png", dpi=180)
    plt.close(ax.figure)


def write_report(config, audit, score_table, inference, runtime, output):
    primary = inference.loc[inference.role.eq("primary") & inference.block_length.eq(20)].iloc[0]
    table = score_table.loc[score_table.subset.eq("all")].pivot(index="model", columns="variant", values=["qlike", "log_rmse"])
    lines = [
        "# 公开档案实验结果", "",
        f"标的：{config['symbol']}；目标：下一 XNYS 交易日的日内已实现方差 rv5。",
        f"评估年份：{', '.join(map(str, config['test_years']))}；完整预测窗口：{int(primary.n)}。", "",
        "## 总体预测损失", "",
        "| 模型 | 原始 QLIKE | 校准 QLIKE | 原始 log-RMSE | 校准 log-RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODEL_NAMES:
        r = table.loc[model]
        lines.append(f"| {model} | {r['qlike']['raw']:.6f} | {r['qlike']['calibrated']:.6f} | {r['log_rmse']['raw']:.6f} | {r['log_rmse']['calibrated']:.6f} |")
    conclusion = "支持 HARNet20 改善预测的预设假设" if primary.ratio_upper < 1 else "未达到支持 HARNet20 改善预测的预设区间标准"
    lines += [
        "", "## 主要比较", "",
        f"校准 HARNet20 / Log-HAR 的平均 QLIKE 比值为 **{primary.ratio:.4f}**，",
        f"20 交易日区块的 95% 配对 bootstrap 区间为 **[{primary.ratio_lower:.4f}, {primary.ratio_upper:.4f}]**。",
        f"平均损失差为 {primary.difference:.6f}，95% 区间 [{primary.difference_lower:.6f}, {primary.difference_upper:.6f}]。",
        f"按区间上界小于 1 的判定规则，本次结果{conclusion}。", "",
        "区间以五种子集成预测为条件；缺失预测日期随 XNYS 网格区块共同重采样。10/40 日敏感性结果见 inference.csv。", "",
        "## 数据与训练", "",
        f"档案有效观测 {int(audit.observed.sum())} 条；缺失交易日 {int(audit.missing.sum())} 个。",
        "session_audit.csv 保存逐日缺失、提前收市和稀疏标记；data_audit.csv 保存年度观察数及完整窗口数。",
        "稀疏标记使用初始训练期每分钟观察数中位数的 25% 阈值，仅用于描述；主样本保留正数、有限方差及有效 nobs 的观测。",
        "完整时段评分指目标日为日历规定的 390 分钟时段；高波动组按预测起点方差划分，阈值为 2017 年前观测的 75% 分位数。",
        "XGBoost 从 12 个参数组合中按验证期合并日均 QLIKE 选择；轮数取获选组合三折最佳轮数的中位数。",
        "两种 HARNet 损失分别用 QLIKE 早停；固定训练轮数取三折、五种子最佳 epoch 的中位数，包含 epoch 0 的 OLS 初始化。",
        "缩放常数为各次拟合的训练目标方差均值。2016 校准预测使用冻结设置重拟合；后续每年仅用前一年的原始预测计算校准系数。",
        "五种子输出先在方差尺度作算术平均，再校准；单种子结果各用其前一年原始预测校准。", "",
        f"实验总用时 {runtime['total_seconds']:.1f} 秒；全部拟合累计 {runtime['fitting_seconds']:.1f} 秒（包括验证搜索与全部五种子）。",
        "分次拟合用时分别见 validation.csv、fitting_times.csv。",
        "运行环境与实际依赖版本见 runtime.json；预测下限触发数见 floor_activations.csv，验证期触发数见 learning_curves.csv。", "",
        "公开档案覆盖至 2019 年；本次结论对应 2017–2019 年。2020–2025 年扩展可使用 compare_sources.py 检查重叠期后接入同一实验。", "",
        "## 输出", "",
        "- predictions.csv：逐日实际值、原始/校准预测及五种子结果。",
        "- scores.csv：总体、年度、高波动和完整时段评分。",
        "- seed_scores.csv / seed_variation.csv：各种子评分及变动。",
        "- inference.csv：主次比较、损失比值/差值及配对区间。",
        "- forecasts.png / annual_qlike.png：预测、累计损失差及年度比较。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
