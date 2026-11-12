# -*- coding: utf-8 -*-
r"""
run_interpretability_gate_2x2.py

图5最终建议版：2×2 分市场绘制 Gate 分布
------------------------------------------------------------
目的：
    避免把 SSE 与 SZI 混在一起后，将“市场身份差异”
    误解释成“市场状态差异”。

图5结构：
                    上证指数（SSE）       深证成指（SZI）
    收益方向        上行 vs 下行          上行 vs 下行
    波动状态        低波动 vs 高波动      低波动 vs 高波动

使用：
    当前正式最终模型（No Cross-Attention + Scalar Gate）
    H=1 默认。

状态定义：
    1) 收益方向：
       预测基准日 t 的 change_pct >= 0 -> 上行，否则下行。

    2) 波动状态：
       对每个市场分别使用训练期 amplitude_pct 中位数作为阈值。
       amplitude_pct <= 训练期本市场中位数 -> 低波动
       amplitude_pct > 训练期本市场中位数 -> 高波动

所有状态均只使用预测基准日 t 可见信息，不使用未来价格。

重要解释：
    当前 Gate 是：
        g_t = sigmoid(W[h_P; z; q_m; F_t] + b)

    融合：
        h_F = LayerNorm(h_P + g_t * W_N z)

因此 Gate 应解释为：
    “新闻增量通道开放程度”
或
    “新闻增量信息利用强度”

不要解释成：
    “新闻贡献率”
或
    “新闻占最终预测的百分比”。

运行：
    cd /d E:\Code\stockv2

    python run_interpretability_gate_2x2.py ^
        --horizon 1 ^
        --step 1 ^
        --device cuda

输出：
    outputs_interpretability/
        fig5_gate_by_market_state_2x2.png
        fig5_gate_by_market_state_2x2.pdf
        fig5_gate_state_stats_by_market.csv
        fig5_gate_state_differences_by_market.csv
        gate_market_state_spearman_by_market.csv
        gate_interpretability_sample_data_2x2.csv

说明：
    图5只做“市场内部状态比较”。
    SSE 与 SZI 的整体 Gate 水平差异建议继续由图6单独展示，
    不要在图5中用统一 y 轴强行比较。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from postexp_utils import (
    load_final_model_context,
    infer_loader,
    load_raw_market_states,
    merge_predictions_states,
    setup_matplotlib_fonts,
)


MARKET_CN = {
    "SSE": "上证指数",
    "SZI": "深证成指",
}


def gate_stats(
    x: pd.DataFrame,
    state_col: str,
) -> pd.DataFrame:
    """
    分 market × state 统计 Gate。
    """
    out = (
        x.groupby(
            [
                "market",
                state_col,
            ]
        )[
            "gate_news_increment"
        ]
        .agg(
            n="count",
            mean="mean",
            std="std",
            median="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
            min="min",
            max="max",
        )
        .reset_index()
    )

    out["state_type"] = state_col

    return out


def compute_within_market_differences(
    trend_stats: pd.DataFrame,
    vol_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    给正文解释准备一个很直观的差异表：

        trend:
            下行 - 上行

        volatility:
            高波动 - 低波动

    同时给绝对差异与相对于第一组均值的百分比差异。
    """
    rows = []

    for market in [
        "SSE",
        "SZI",
    ]:
        # ----------------------------
        # Trend: Down - Up
        # ----------------------------
        t = trend_stats[
            trend_stats["market"] == market
        ].set_index(
            "trend_state"
        )

        if (
            "上行" in t.index
            and "下行" in t.index
        ):
            up_mean = float(
                t.loc[
                    "上行",
                    "mean",
                ]
            )
            down_mean = float(
                t.loc[
                    "下行",
                    "mean",
                ]
            )

            up_median = float(
                t.loc[
                    "上行",
                    "median",
                ]
            )
            down_median = float(
                t.loc[
                    "下行",
                    "median",
                ]
            )

            rows.append({
                "market": market,
                "comparison": "下行-上行",
                "group_a": "上行",
                "group_b": "下行",
                "mean_a": up_mean,
                "mean_b": down_mean,
                "mean_diff_b_minus_a": (
                    down_mean
                    - up_mean
                ),
                "mean_diff_pct_vs_a": (
                    (
                        down_mean
                        - up_mean
                    )
                    / (
                        abs(up_mean)
                        + 1e-12
                    )
                    * 100.0
                ),
                "median_a": up_median,
                "median_b": down_median,
                "median_diff_b_minus_a": (
                    down_median
                    - up_median
                ),
            })

        # ----------------------------
        # Vol: High - Low
        # ----------------------------
        v = vol_stats[
            vol_stats["market"] == market
        ].set_index(
            "volatility_state"
        )

        if (
            "低波动" in v.index
            and "高波动" in v.index
        ):
            low_mean = float(
                v.loc[
                    "低波动",
                    "mean",
                ]
            )
            high_mean = float(
                v.loc[
                    "高波动",
                    "mean",
                ]
            )

            low_median = float(
                v.loc[
                    "低波动",
                    "median",
                ]
            )
            high_median = float(
                v.loc[
                    "高波动",
                    "median",
                ]
            )

            rows.append({
                "market": market,
                "comparison": "高波动-低波动",
                "group_a": "低波动",
                "group_b": "高波动",
                "mean_a": low_mean,
                "mean_b": high_mean,
                "mean_diff_b_minus_a": (
                    high_mean
                    - low_mean
                ),
                "mean_diff_pct_vs_a": (
                    (
                        high_mean
                        - low_mean
                    )
                    / (
                        abs(low_mean)
                        + 1e-12
                    )
                    * 100.0
                ),
                "median_a": low_median,
                "median_b": high_median,
                "median_diff_b_minus_a": (
                    high_median
                    - low_median
                ),
            })

    return pd.DataFrame(
        rows
    )


def get_market_ylim(
    market_df: pd.DataFrame,
):
    """
    每个市场单独确定 y 轴范围。

    同一个市场的“收益方向”和“波动状态”两个面板使用完全相同的
    y 轴范围，便于市场内部比较；SSE 与 SZI 不强制使用同一个范围，
    避免因两个市场 Gate 基础水平差异较大而把箱体压扁。
    """
    y = (
        market_df[
            "gate_news_increment"
        ]
        .astype(float)
        .to_numpy()
    )

    ymin = float(
        np.nanmin(y)
    )
    ymax = float(
        np.nanmax(y)
    )

    span = max(
        ymax - ymin,
        1e-4,
    )

    pad = max(
        span * 0.10,
        0.0015,
    )

    lo = max(
        0.0,
        ymin - pad,
    )

    hi = min(
        1.0,
        ymax + pad,
    )

    return lo, hi


def draw_one_boxplot(
    ax,
    market_df: pd.DataFrame,
    group_col: str,
    categories,
    market: str,
    y_lim,
    show_ylabel: bool,
):
    data = []
    labels = []

    for category in categories:
        s = (
            market_df.loc[
                market_df[
                    group_col
                ] == category,
                "gate_news_increment",
            ]
            .astype(float)
            .to_numpy()
        )

        data.append(
            s
        )

        labels.append(
            f"{category}\n(n={len(s)})"
        )

    ax.boxplot(
        data,
        labels=labels,
        showmeans=True,
        showfliers=True,
        widths=0.55,
        meanprops={
            "marker": "^",
            "markersize": 6,
        },
    )

    ax.set_ylim(
        *y_lim
    )

    if show_ylabel:
        ax.set_ylabel(
            "新闻增量门控系数",
            fontsize=11,
        )

    ax.tick_params(
        axis="both",
        labelsize=10,
    )

    ax.grid(
        axis="y",
        alpha=0.18,
        linewidth=0.6,
    )

    # 左上角补充“均值差”，让读者不用只靠箱线图判断
    means = []
    for category in categories:
        s = market_df.loc[
            market_df[group_col]
            == category,
            "gate_news_increment",
        ]
        means.append(
            float(
                s.mean()
            )
        )

    diff = (
        means[1]
        - means[0]
    )

    ax.text(
        0.03,
        0.96,
        (
            f"均值差({categories[1]}−{categories[0]})"
            f" = {diff:+.4f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )


def save_2x2_figure(
    x: pd.DataFrame,
    out_dir: Path,
):
    setup_matplotlib_fonts()

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(
            10.5,
            7.6,
        ),
    )

    # 每一列对应一个市场；
    # 该市场上下两个面板使用同一 y 轴范围。
    ylims = {
        market: get_market_ylim(
            x[
                x["market"] == market
            ]
        )
        for market in [
            "SSE",
            "SZI",
        ]
    }

    # ----------------------------
    # Row 1: trend
    # ----------------------------
    for col, market in enumerate(
        [
            "SSE",
            "SZI",
        ]
    ):
        g = x[
            x["market"]
            == market
        ].copy()

        draw_one_boxplot(
            ax=axes[
                0,
                col,
            ],
            market_df=g,
            group_col=(
                "trend_state"
            ),
            categories=[
                "上行",
                "下行",
            ],
            market=market,
            y_lim=ylims[
                market
            ],
            show_ylabel=(
                col == 0
            ),
        )

        axes[
            0,
            col,
        ].set_title(
            MARKET_CN[
                market
            ],
            fontsize=13,
            pad=9,
        )

    # ----------------------------
    # Row 2: volatility
    # ----------------------------
    for col, market in enumerate(
        [
            "SSE",
            "SZI",
        ]
    ):
        g = x[
            x["market"]
            == market
        ].copy()

        draw_one_boxplot(
            ax=axes[
                1,
                col,
            ],
            market_df=g,
            group_col=(
                "volatility_state"
            ),
            categories=[
                "低波动",
                "高波动",
            ],
            market=market,
            y_lim=ylims[
                market
            ],
            show_ylabel=(
                col == 0
            ),
        )

    # 行标题
    axes[
        0,
        0,
    ].text(
        -0.20,
        0.5,
        "收益方向",
        transform=axes[
            0,
            0,
        ].transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=12,
    )

    axes[
        1,
        0,
    ].text(
        -0.20,
        0.5,
        "波动状态",
        transform=axes[
            1,
            0,
        ].transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=12,
    )

    fig.suptitle(
        "不同市场状态下的新闻增量门控系数",
        fontsize=16,
        y=0.995,
    )

    # 底部注释：避免把不同列 y 范围误当成可直接纵向比较
    fig.text(
        0.5,
        0.015,
        (
            "注：每个市场上下两幅子图采用相同纵轴范围；"
            "SSE 与 SZI 分别缩放，以突出市场内部状态差异。"
            "跨市场整体水平差异建议结合图6分析。"
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )

    fig.tight_layout(
        rect=[
            0.05,
            0.055,
            1.0,
            0.96,
        ]
    )

    png = (
        out_dir
        / "fig5_gate_by_market_state_2x2.png"
    )

    pdf = (
        out_dir
        / "fig5_gate_by_market_state_2x2.pdf"
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

    plt.close(
        fig
    )

    return png, pdf


def compute_spearman(
    x: pd.DataFrame,
) -> pd.DataFrame:
    """
    只输出分市场相关，不再把 ALL 作为主要解释对象，
    避免跨市场基础 Gate 差异造成混杂。
    """
    rows = []

    for market, g in x.groupby(
        "market"
    ):
        for var in [
            "amplitude_pct",
            "abs_return_pct",
            "turnover_rate_pct",
        ]:
            rho, pval = spearmanr(
                g[
                    "gate_news_increment"
                ],
                g[var],
                nan_policy="omit",
            )

            rows.append({
                "market": market,
                "market_cn": MARKET_CN.get(
                    market,
                    market,
                ),
                "market_state_variable": var,
                "spearman_rho": float(
                    rho
                ),
                "p_value": float(
                    pval
                ),
                "n": int(
                    len(g)
                ),
            })

    return pd.DataFrame(
        rows
    )


def main():
    p = argparse.ArgumentParser(
        description=(
            "图5：2×2 分市场 Gate 可解释性分析"
        )
    )

    p.add_argument(
        "--root",
        default=r"E:\Code\stockv2",
    )

    p.add_argument(
        "--horizon",
        type=int,
        default=1,
    )

    p.add_argument(
        "--step",
        type=int,
        default=1,
    )

    p.add_argument(
        "--model_dir",
        default=(
            "outputs_ablation_no_cross_attention"
        ),
    )

    p.add_argument(
        "--out_dir",
        default=(
            "outputs_interpretability"
        ),
    )

    p.add_argument(
        "--device",
        choices=[
            "cuda",
            "cpu",
        ],
        default="cuda",
    )

    args = p.parse_args()

    root = Path(
        args.root
    )

    out_dir = (
        root
        / args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. Load formal final model
    # ========================================================
    (
        model,
        prepared,
        datasets,
        loaders,
        device,
        config,
    ) = load_final_model_context(
        root=root,
        horizon=args.horizon,
        model_dir=args.model_dir,
        device=args.device,
    )

    pred = infer_loader(
        model,
        loaders[
            "test"
        ],
        device,
    )

    pred = pred[
        pred[
            "horizon_step"
        ] == args.step
    ].copy()

    # ========================================================
    # 2. Attach as-of-t market state
    # ========================================================
    states = load_raw_market_states(
        root
    )

    x = merge_predictions_states(
        pred,
        states,
    )

    # ========================================================
    # 3. Market-specific training volatility threshold
    # ========================================================
    train_states = states[
        states["date"]
        <= prepared.strict_train_asof_end
    ].copy()

    amp_threshold = (
        train_states
        .groupby(
            "market"
        )[
            "amplitude_pct"
        ]
        .median()
        .to_dict()
    )

    x[
        "trend_state"
    ] = np.where(
        x[
            "change_pct"
        ] >= 0,
        "上行",
        "下行",
    )

    x[
        "volatility_state"
    ] = [
        (
            "低波动"
            if amp
            <= amp_threshold[
                market
            ]
            else "高波动"
        )
        for amp, market
        in zip(
            x[
                "amplitude_pct"
            ],
            x[
                "market"
            ],
        )
    ]

    # ========================================================
    # 4. Stats
    # ========================================================
    trend_stats = gate_stats(
        x,
        "trend_state",
    )

    vol_stats = gate_stats(
        x,
        "volatility_state",
    )

    stats = pd.concat(
        [
            trend_stats,
            vol_stats,
        ],
        ignore_index=True,
        sort=False,
    )

    stats_path = (
        out_dir
        / "fig5_gate_state_stats_by_market.csv"
    )

    stats.to_csv(
        stats_path,
        index=False,
        encoding="utf-8-sig",
    )

    differences = (
        compute_within_market_differences(
            trend_stats,
            vol_stats,
        )
    )

    diff_path = (
        out_dir
        / "fig5_gate_state_differences_by_market.csv"
    )

    differences.to_csv(
        diff_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 5. Spearman within each market
    # ========================================================
    corr = compute_spearman(
        x
    )

    corr_path = (
        out_dir
        / "gate_market_state_spearman_by_market.csv"
    )

    corr.to_csv(
        corr_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 6. 2×2 Fig.5
    # ========================================================
    png, pdf = save_2x2_figure(
        x,
        out_dir,
    )

    # ========================================================
    # 7. Save sample data
    # ========================================================
    sample_path = (
        out_dir
        / "gate_interpretability_sample_data_2x2.csv"
    )

    x.to_csv(
        sample_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 8. Console output
    # ========================================================
    print(
        "\n"
        + "=" * 80
    )

    print(
        "FIG.5 | 2x2 MARKET-SPECIFIC GATE ANALYSIS"
    )

    print(
        "=" * 80
    )

    print(
        f"Checkpoint: {config['checkpoint']}"
    )

    print(
        f"H={args.horizon}, step={args.step}"
    )

    print(
        "\nVolatility thresholds "
        "(training-period market-specific median):"
    )

    for market in [
        "SSE",
        "SZI",
    ]:
        print(
            f"  {market}: "
            f"{amp_threshold[market]:.6f}"
        )

    print(
        "\n========== Gate by trend, within market =========="
    )

    print(
        trend_stats.to_string(
            index=False
        )
    )

    print(
        "\n========== Gate by volatility, within market =========="
    )

    print(
        vol_stats.to_string(
            index=False
        )
    )

    print(
        "\n========== Within-market mean/median differences =========="
    )

    print(
        differences.to_string(
            index=False
        )
    )

    print(
        "\n========== Gate-state Spearman, within market =========="
    )

    print(
        corr.to_string(
            index=False
        )
    )

    print(
        "\n[SAVE]"
    )

    for path in [
        png,
        pdf,
        stats_path,
        diff_path,
        corr_path,
        sample_path,
    ]:
        print(
            path
        )

    print(
        "\nInterpretation note:"
    )

    print(
        "图5只用于判断同一市场内部的状态差异；"
        "不要用不同列的纵轴高度直接比较 SSE 与 SZI。"
    )

    print(
        "跨市场 Gate 基础水平差异继续由图6单独解释。"
    )


if __name__ == "__main__":
    main()
