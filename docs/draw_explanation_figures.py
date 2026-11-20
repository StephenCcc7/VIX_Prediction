"""Draw the teaching figures from formulas and saved results; no model fitting."""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "Microsoft YaHei", "font.size": 12,
    "axes.unicode_minus": False, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
})
BLUE, GREEN, ORANGE, INK = "#2671ad", "#258273", "#d57c28", "#24364b"


def box(ax, x, y, width, height, label, color=BLUE, size=12):
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.018,rounding_size=0.04",
        facecolor=color, edgecolor="none", alpha=0.10,
    ))
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=size, color=INK)


def arrow(ax, start, end, color="#728095"):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, mutation_scale=14))


def pipeline():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set(xlim=(0, 12), ylim=(0, 6))
    ax.axis("off")
    ax.text(0.2, 5.65, "整个项目：把过去二十天，变成对明天的一个数值预测", fontsize=19, weight="bold", color=INK)
    labels = [
        (0.25, "每日已实现方差\n档案字段 rv5"),
        (4.15, "按交易日排列\n组成 20 天输入窗口"),
        (8.05, "预测模型\n输出明天的方差"),
    ]
    for x, label in labels:
        box(ax, x, 3.65, 3.15, 1.15, label)
    arrow(ax, (3.5, 4.2), (4.0, 4.2))
    arrow(ax, (7.4, 4.2), (7.9, 4.2))
    for x, label, color in [
        (8.05, "年度校准\n乘上过去数据确定的系数", ORANGE),
        (4.15, "等明天结束\n拿预测与真实 rv5 比较", GREEN),
        (0.25, "汇总评分与区间\n判断改善是否稳定", GREEN),
    ]:
        box(ax, x, 1.05, 3.15, 1.15, label, color)
    arrow(ax, (9.62, 3.5), (9.62, 2.35))
    arrow(ax, (7.95, 1.62), (7.45, 1.62))
    arrow(ax, (4.05, 1.62), (3.55, 1.62))
    ax.text(0.25, 0.25, "当前输入是已计算好的每日 rv5；读取 CSV 后，每一行是一个交易日。", color="#526174", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "01_pipeline.png", dpi=170)
    plt.close(fig)


def harnet():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set(xlim=(0, 12), ylim=(0, 8))
    ax.axis("off")
    ax.text(0.15, 7.55, "HARNet20：把固定的周、月平均，变成可学习的加权汇总", fontsize=18, weight="bold", color=INK)
    ax.text(0.15, 7.08, "输入由旧到新排列；图中画出月分支会使用的四个周输出。", color="#526174")
    xs = [0.2, 3.15, 6.1, 9.05]
    day_labels = ["第 1—5 天", "第 6—10 天", "第 11—15 天", "第 16—20 天"]
    qlabels = ["q[0]", "q[5]", "q[10]", "q[15]（最近一周）"]
    for x, days, q in zip(xs, day_labels, qlabels):
        box(ax, x, 5.85, 2.6, 0.65, days)
        arrow(ax, (x + 1.3, 5.75), (x + 1.3, 4.98))
        box(ax, x, 4.15, 2.6, 0.75, f"{q}\n共享同一组 5 个权重", GREEN, 11)
        arrow(ax, (x + 1.3, 4.03), (5.95, 3.25))
    ax.text(0.25, 5.2, "第一层：长度 5 的卷积 + ReLU", color=GREEN, fontsize=11,
            bbox=dict(facecolor="white", edgecolor="none", pad=2))
    box(ax, 3.65, 2.35, 4.6, 0.85, "月输出 m：用 4 个权重组合\nq[0]、q[5]、q[10]、q[15]", ORANGE, 11)
    ax.text(0.3, 2.55, "第二层：每隔 5 个周输出取一个\n扩张率 dilation = 5", color=ORANGE, fontsize=10)
    box(ax, 3.0, 0.65, 5.9, 0.85, "线性头：截距 + 今天 × 系数 + 最近一周 × 系数 + 月输出 × 系数", BLUE, 11)
    arrow(ax, (5.95, 2.22), (5.95, 1.62))
    arrow(ax, (10.65, 4.0), (8.5, 1.62))
    ax.text(9.5, 2.6, "最近一周\n单独送入线性头", fontsize=10, color=INK,
            bbox=dict(facecolor="white", edgecolor="none", pad=2))
    arrow(ax, (0.8, 0.98), (2.9, 0.98))
    ax.text(0.25, 1.48, "今天的方差\n也直接送入", fontsize=10, color=INK)
    ax.text(0.2, 0.08, "参数数目：第一层 5 个 + 第二层 4 个 + 线性头 4 个 = 13 个。初始权重分别设为 1/5 和 1/4。", fontsize=11, color="#526174")
    fig.tight_layout()
    fig.savefig(OUT / "02_harnet.png", dpi=170)
    plt.close(fig)


def timeline():
    columns = ["2000—2013", "2014", "2015", "2016", "2017", "2018", "2019"]
    rows = ["验证折 1", "验证折 2", "验证折 3", "2017 年预测", "2018 年预测", "2019 年预测"]
    state = [
        [1, 2, 0, 0, 0, 0, 0], [1, 1, 2, 0, 0, 0, 0], [1, 1, 1, 2, 0, 0, 0],
        [1, 1, 1, 1, 3, 0, 0], [1, 1, 1, 1, 1, 3, 0], [1, 1, 1, 1, 1, 1, 3],
    ]
    colors, words = {0: "#f0f2f5", 1: "#dae9f5", 2: "#fbe8cf", 3: "#d5eae4"}, {0: "", 1: "拟合", 2: "验证", 3: "预测 / 评分"}
    fig, ax = plt.subplots(figsize=(12, 5.4))
    for i, row in enumerate(state):
        for j, value in enumerate(row):
            ax.add_patch(Rectangle((j, 5-i), 0.95, 0.8, facecolor=colors[value], edgecolor="white"))
            ax.text(j + .475, 5-i + .4, words[value], ha="center", va="center", fontsize=10, color=INK)
    ax.set(xlim=(-1.5, 7), ylim=(-.55, 6.4))
    ax.axis("off")
    for j, label in enumerate(columns):
        ax.text(j + .475, 6.08, label, ha="center", fontsize=11, weight="bold")
    for i, label in enumerate(rows):
        ax.text(-.15, 5-i + .4, label, ha="right", va="center", fontsize=11)
    ax.text(-1.42, -.4, "前 3 行用于选择设置，后 3 行用于评价。模型每年重估一次；年内每天更新输入窗口。首列合并了 14 年。", fontsize=11, color="#526174")
    ax.set_title("时间顺序就是实验设计的一部分：用过去拟合，再向后预测", fontsize=17, pad=20, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "03_timeline.png", dpi=170)
    plt.close(fig)


def losses():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    ratio = np.linspace(.23, 3.0, 400)
    loss = 1/ratio + np.log(ratio) - 1
    axes[0].plot(ratio, loss, color=BLUE, lw=2.5)
    axes[0].axvline(1, color="#728095", ls="--", lw=1)
    for r, label, xytext in [(.5, "预测为真实值的一半\nQLIKE ≈ 0.307", (.38, .68)), (2, "预测为真实值的两倍\nQLIKE ≈ 0.193", (1.6, .9))]:
        value = 1/r + np.log(r) - 1
        axes[0].scatter(r, value, color=ORANGE, zorder=3)
        axes[0].annotate(label, (r, value), xytext=xytext, fontsize=10,
                         arrowprops=dict(arrowstyle="->", color=ORANGE))
    axes[0].set(xlabel="预测值 / 真实值", ylabel="QLIKE 损失", title="评分比较的是相对误差", ylim=(-.03, 2.0))
    axes[1].bar([0, 1], [3, 5], color=[BLUE, GREEN], width=.55)
    axes[1].set_xticks([0, 1], ["平方 log 误差的最佳预测", "QLIKE 的最佳预测"])
    axes[1].set(ylabel="预测值（教学示例单位）", ylim=(0, 6.5), title="同样的信息，损失函数不同，最佳预测也不同")
    for i, value in enumerate([3, 5]):
        axes[1].text(i, value + .12, str(value), ha="center", fontsize=15, weight="bold")
    axes[1].text(.5, 6.05, "教学例子：明天的方差以相同概率等于 1 或 9", ha="center", fontsize=10, color="#526174")
    fig.savefig(OUT / "04_losses.png", dpi=170)
    plt.close(fig)


def calibration():
    source = HERE.parent / "results"
    table = pd.read_csv(source / "diagnostics/calibration_decomposition.csv")
    selected = table.loc[table.model.isin(["Log-HAR", "HARNet20"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    for offset, model, color in [(-.17, "Log-HAR", BLUE), (.17, "HARNet20", GREEN)]:
        frame = selected.loc[selected.model.eq(model)]
        axes[0].bar(np.arange(3) + offset, frame.calibration_loss_change, width=.32, label=model, color=color)
    axes[0].axhline(0, color="#728095", lw=1)
    axes[0].set_xticks(np.arange(3), ["2017", "2018", "2019"])
    axes[0].set(ylabel="校准后损失 − 原始损失", title="2018 年：同一校准规则，对两模型的影响相反")
    axes[0].legend()
    inference = pd.read_csv(source / "inference.csv")
    for i, variant in enumerate(["raw", "calibrated"]):
        row = inference.loc[
            inference.candidate.eq("HARNet20") & inference.reference.eq("Log-HAR")
            & inference.variant.eq(variant) & inference.block_length.eq(20)
        ].iloc[0]
        axes[1].errorbar(row.ratio, 1-i, xerr=[[row.ratio-row.ratio_lower], [row.ratio_upper-row.ratio]],
                         fmt="o", color=[GREEN, ORANGE][i], capsize=6, markersize=8)
        axes[1].text(row.ratio, 1-i + .17, f"{row.ratio:.4f}", ha="center", fontsize=11)
    axes[1].axvline(1, color="#728095", ls="--")
    axes[1].set_yticks([1, 0], ["原始预测", "年度校准后"])
    axes[1].set(xlim=(.79, 1.12), ylim=(-.5, 1.5), xlabel="HARNet20 平均损失 / Log-HAR 平均损失", title="点表示损失比，横线表示 95% 区间")
    axes[1].text(.82, -.35, "低于 1：HARNet20 更好", fontsize=10, color=GREEN)
    fig.savefig(OUT / "05_results.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    pipeline()
    harnet()
    timeline()
    losses()
    calibration()
    print(f"Teaching figures: {OUT}")
