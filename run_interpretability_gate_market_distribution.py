# -*- coding: utf-8 -*-

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import mannwhitneyu


MARKET_CN = {
    "SSE": "上证指数",
    "SZI": "深证成指",
}


def setup_fonts():
    available = {
        f.name for f in font_manager.fontManager.ttflist
    }

    candidates = [
        "SimSun",
        "宋体",
        "STSong",
        "Songti SC",
        "Microsoft YaHei",
        "SimHei",
    ]

    chinese = next(
        (x for x in candidates if x in available),
        None,
    )

    if chinese:
        plt.rcParams["font.sans-serif"] = [
            chinese,
            "Times New Roman",
        ]

    plt.rcParams["axes.unicode_minus"] = False


def describe_gate(df):
    rows = []

    for market, g in df.groupby("market"):
        s = g["gate_news_increment"].astype(float)

        rows.append({
            "market": market,
            "market_cn": MARKET_CN.get(market, market),
            "n": len(s),
            "mean": s.mean(),
            "std": s.std(ddof=1),
            "median": s.median(),
            "q25": s.quantile(0.25),
            "q75": s.quantile(0.75),
            "min": s.min(),
            "max": s.max(),
        })

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--root",
        default=r"E:\Code\stockv2",
    )

    p.add_argument(
        "--input_dir",
        default="outputs_interpretability_finalnews",
    )

    p.add_argument(
        "--out_dir",
        default="outputs_interpretability_finalnews",
    )

    args = p.parse_args()

    root = Path(args.root)

    input_path = (
        root
        / args.input_dir
        / "gate_interpretability_sample_data_2x2.csv"
    )

    out_dir = (
        root
        / args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
    )

    df["market"] = (
        df["market"]
        .astype(str)
        .replace({
            "SH": "SSE",
            "SZ": "SZI",
        })
    )

    df["gate_news_increment"] = pd.to_numeric(
        df["gate_news_increment"],
        errors="raise",
    )

    # ==========================================
    # 1. 描述统计
    # ==========================================

    stats = describe_gate(df)

    stats.to_csv(
        out_dir / "fig6_gate_distribution_stats.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ==========================================
    # 2. 跨市场非参数检验
    # ==========================================

    sse = df.loc[
        df["market"] == "SSE",
        "gate_news_increment",
    ].to_numpy()

    szi = df.loc[
        df["market"] == "SZI",
        "gate_news_increment",
    ].to_numpy()

    u_stat, p_value = mannwhitneyu(
        sse,
        szi,
        alternative="two-sided",
    )

    test_df = pd.DataFrame([{
        "comparison": "SSE_vs_SZI",
        "SSE_n": len(sse),
        "SZI_n": len(szi),
        "SSE_mean": np.mean(sse),
        "SZI_mean": np.mean(szi),
        "mean_difference_SZI_minus_SSE":
            np.mean(szi) - np.mean(sse),
        "SSE_median": np.median(sse),
        "SZI_median": np.median(szi),
        "mann_whitney_U": u_stat,
        "p_value": p_value,
    }])

    test_df.to_csv(
        out_dir / "fig6_gate_market_difference_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ==========================================
    # 3. Figure
    # ==========================================

    setup_fonts()

    fig, ax = plt.subplots(
        figsize=(8.4, 5.8)
    )

    data = [
        sse,
        szi,
    ]

    labels = [
        f"上证指数\n(n={len(sse)})",
        f"深证成指\n(n={len(szi)})",
    ]

    bp = ax.boxplot(
        data,
        labels=labels,
        showmeans=True,
        showfliers=True,
        widths=0.50,
        meanprops={
            "marker": "^",
            "markersize": 7,
        },
    )

    # 叠加轻微抖动散点，让整体分布更直观
    rng = np.random.default_rng(42)

    for i, values in enumerate(data, start=1):
        jitter = rng.normal(
            loc=i,
            scale=0.035,
            size=len(values),
        )

        ax.scatter(
            jitter,
            values,
            s=15,
            alpha=0.22,
        )

    sse_mean = float(np.mean(sse))
    szi_mean = float(np.mean(szi))

    ax.text(
        0.03,
        0.97,
        (
            f"均值：SSE={sse_mean:.4f}，"
            f"SZI={szi_mean:.4f}\n"
            f"均值差(SZI−SSE)="
            f"{szi_mean - sse_mean:+.4f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )

    ax.set_ylabel(
        "新闻增量门控系数",
        fontsize=12,
    )

    ax.set_title(
        "不同市场的新闻增量门控系数分布",
        fontsize=16,
        pad=12,
    )

    ax.grid(
        axis="y",
        alpha=0.18,
        linewidth=0.6,
    )

    ax.tick_params(
        axis="both",
        labelsize=11,
    )

    fig.text(
        0.5,
        0.015,
        (
            "注：Gate 表示新闻增量通道的开放程度，"
            "不表示新闻信息在最终预测中的贡献比例。"
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )

    fig.tight_layout(
        rect=[0.03, 0.05, 1.0, 1.0]
    )

    png = (
        out_dir
        / "fig6_gate_distribution_by_market.png"
    )

    pdf = (
        out_dir
        / "fig6_gate_distribution_by_market.pdf"
    )

    fig.savefig(
        png,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("=" * 72)
    print("FIG.6 | GATE DISTRIBUTION BY MARKET")
    print("=" * 72)

    print("\n========== Descriptive statistics ==========")
    print(
        stats.to_string(index=False)
    )

    print("\n========== Market difference ==========")
    print(
        test_df.to_string(index=False)
    )

    print("\n[SAVE]")
    print(png)
    print(pdf)
    print(
        out_dir
        / "fig6_gate_distribution_stats.csv"
    )
    print(
        out_dir
        / "fig6_gate_market_difference_test.csv"
    )


if __name__ == "__main__":
    main()