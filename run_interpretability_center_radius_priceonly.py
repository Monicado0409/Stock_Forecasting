# -*- coding: utf-8 -*-
"""
run_interpretability_center_radius_priceonly.py

图4：文本信息对区间中心与半径的作用路径
      —— Full Model vs Price-only Model

适用于当前最终模型：
    Full:
        Price GRU
        + News GRU
        + Scalar Information Absorption Gate
        + Residual Fusion
        + Center/Radius Heads
        （无 Cross-Attention）

对照模型：
    Price-only:
        Price GRU
        + Center/Radius Heads

本脚本不重新训练模型，而是直接读取两个正式实验已经生成的
test_predictions.csv，并在完全相同的测试样本上比较预测结果。

定义：
    DeltaCenter_rel =
        (Center_full - Center_price_only) / base_close

    DeltaRadius_rel =
        (Radius_full - Radius_price_only) / base_close

解释：
    DeltaCenter_rel > 0：
        引入新闻后，完整模型的预测中心高于 Price-only。

    DeltaCenter_rel < 0：
        引入新闻后，完整模型的预测中心低于 Price-only。

    DeltaRadius_rel > 0：
        引入新闻后，完整模型的预测区间半径更宽。

    DeltaRadius_rel < 0：
        引入新闻后，完整模型的预测区间半径更窄。

注意：
    这里反映的是“模型级增量修正”，不是因果效应。
    Full 和 Price-only 是分别训练的两个模型，因此它们之间的预测差异
    同时包含“是否引入新闻信息”以及重新训练后的参数适配效应。
    论文中建议使用：
        “文本信息的模型级增量修正”
    而不要使用：
        “文本信息的因果效应”。

默认文件：
    E:\\Code\\stockv2\\
        outputs_ablation_no_cross_attention\\H1\\test_predictions.csv
        outputs_ablation_price_only\\H1\\test_predictions.csv

运行：
    cd /d E:\\Code\\stockv2

    python run_interpretability_center_radius_priceonly.py ^
        --horizon 1 ^
        --step 1

输出：
    outputs_interpretability\\
        fig4_full_vs_priceonly_center_radius.png
        fig4_full_vs_priceonly_center_radius.pdf
        fig4_full_vs_priceonly_data.csv
        fig4_full_vs_priceonly_stats.csv
        fig4_full_vs_priceonly_quadrant_stats.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

from stockv2_data_tune import (
    load_price_data,
)


# ============================================================
# 1. Basic utilities
# ============================================================

def normalize_market(
    s: pd.Series,
) -> pd.Series:
    mapping = {
        "SH": "SSE",
        "SSE": "SSE",
        "SZ": "SZI",
        "SZI": "SZI",
    }

    raw = (
        s.astype(str)
        .str.strip()
        .str.upper()
    )

    return (
        raw.map(mapping)
        .fillna(raw)
    )


def setup_matplotlib_fonts():
    """
    尽量使用宋体；如果环境没有宋体，则回退到常见中文字体。
    """
    available = {
        f.name
        for f in font_manager.fontManager.ttflist
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
        (
            name
            for name in candidates
            if name in available
        ),
        None,
    )

    if chinese is not None:
        plt.rcParams[
            "font.sans-serif"
        ] = [
            chinese,
            "Times New Roman",
        ]

    plt.rcParams[
        "axes.unicode_minus"
    ] = False


def check_required_columns(
    df: pd.DataFrame,
    name: str,
):
    required = {
        "market",
        "as_of_date",
        "target_date",
        "horizon_step",
        "true_center",
        "pred_center",
        "true_radius",
        "pred_radius",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{name} 缺少字段：{missing}"
        )


# ============================================================
# 2. Load and align predictions
# ============================================================

def load_prediction_file(
    path: Path,
    name: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 {name}：\n{path}"
        )

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    check_required_columns(
        df,
        name,
    )

    df["market"] = normalize_market(
        df["market"]
    )

    df["as_of_date"] = pd.to_datetime(
        df["as_of_date"],
        errors="raise",
    )

    df["target_date"] = pd.to_datetime(
        df["target_date"],
        errors="raise",
    )

    df["horizon_step"] = pd.to_numeric(
        df["horizon_step"],
        errors="raise",
    ).astype(int)

    return df


def add_base_close(
    root: Path,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    使用预测基准日 t 的 close 作为相对修正的归一化基准，
    与主模型恢复绝对中心/半径时使用的 base_close 保持一致。
    """
    price = load_price_data(
        root
    ).copy()

    price["market"] = normalize_market(
        price["market"]
    )

    price["date"] = pd.to_datetime(
        price["date"]
    )

    base = (
        price[
            [
                "market",
                "date",
                "close",
            ]
        ]
        .rename(
            columns={
                "date": "as_of_date",
                "close": "base_close",
            }
        )
        .copy()
    )

    out = df.merge(
        base,
        on=[
            "market",
            "as_of_date",
        ],
        how="left",
        validate="many_to_one",
    )

    if out[
        "base_close"
    ].isna().any():
        bad = out.loc[
            out["base_close"].isna(),
            [
                "market",
                "as_of_date",
            ],
        ].drop_duplicates()

        raise ValueError(
            "部分预测样本无法匹配 base_close：\n"
            + bad.to_string(index=False)
        )

    return out


def align_full_and_priceonly(
    full: pd.DataFrame,
    price_only: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "market",
        "as_of_date",
        "target_date",
        "horizon_step",
    ]

    full_cols = keys + [
        "true_center",
        "true_radius",
        "pred_center",
        "pred_radius",
    ]

    po_cols = keys + [
        "true_center",
        "true_radius",
        "pred_center",
        "pred_radius",
    ]

    x = full[
        full_cols
    ].rename(
        columns={
            "true_center": "true_center_full",
            "true_radius": "true_radius_full",
            "pred_center": "pred_center_full",
            "pred_radius": "pred_radius_full",
        }
    ).merge(
        price_only[
            po_cols
        ].rename(
            columns={
                "true_center": "true_center_price_only",
                "true_radius": "true_radius_price_only",
                "pred_center": "pred_center_price_only",
                "pred_radius": "pred_radius_price_only",
            }
        ),
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    if len(x) != len(full) or len(x) != len(
        price_only
    ):
        print(
            "[WARN] Full 与 Price-only 的预测记录数不完全一致。"
        )
        print(
            f"       Full={len(full)}, "
            f"Price-only={len(price_only)}, "
            f"Matched={len(x)}"
        )

    # 检查真实目标是否一致
    center_diff = np.max(
        np.abs(
            x["true_center_full"].to_numpy()
            - x[
                "true_center_price_only"
            ].to_numpy()
        )
    )

    radius_diff = np.max(
        np.abs(
            x["true_radius_full"].to_numpy()
            - x[
                "true_radius_price_only"
            ].to_numpy()
        )
    )

    if (
        center_diff > 1e-5
        or radius_diff > 1e-5
    ):
        raise ValueError(
            "Full 与 Price-only 在同一日期的真实目标不一致，"
            "不能直接做图4比较。\n"
            f"max center diff={center_diff:.8f}, "
            f"max radius diff={radius_diff:.8f}"
        )

    x["true_center"] = (
        x["true_center_full"]
    )
    x["true_radius"] = (
        x["true_radius_full"]
    )

    return x


# ============================================================
# 3. Compute contribution statistics
# ============================================================

def add_contribution_columns(
    x: pd.DataFrame,
) -> pd.DataFrame:
    x = x.copy()

    x["delta_center_abs"] = (
        x["pred_center_full"]
        - x[
            "pred_center_price_only"
        ]
    )

    x["delta_radius_abs"] = (
        x["pred_radius_full"]
        - x[
            "pred_radius_price_only"
        ]
    )

    base = (
        x["base_close"]
        .astype(float)
        .clip(lower=1e-8)
    )

    x["delta_center_rel"] = (
        x["delta_center_abs"]
        / base
    )

    x["delta_radius_rel"] = (
        x["delta_radius_abs"]
        / base
    )

    # 同时计算“加入新闻后是否改善中心/半径误差”，
    # 作为后续解释的辅助诊断数据，不直接用于散点坐标。
    x["center_abs_error_full"] = np.abs(
        x["pred_center_full"]
        - x["true_center"]
    )

    x[
        "center_abs_error_price_only"
    ] = np.abs(
        x["pred_center_price_only"]
        - x["true_center"]
    )

    x[
        "radius_abs_error_full"
    ] = np.abs(
        x["pred_radius_full"]
        - x["true_radius"]
    )

    x[
        "radius_abs_error_price_only"
    ] = np.abs(
        x["pred_radius_price_only"]
        - x["true_radius"]
    )

    x[
        "center_error_improvement"
    ] = (
        x[
            "center_abs_error_price_only"
        ]
        - x[
            "center_abs_error_full"
        ]
    )

    x[
        "radius_error_improvement"
    ] = (
        x[
            "radius_abs_error_price_only"
        ]
        - x[
            "radius_abs_error_full"
        ]
    )

    # 四象限标签
    conditions = [
        (
            (x["delta_center_rel"] >= 0)
            & (x["delta_radius_rel"] >= 0)
        ),
        (
            (x["delta_center_rel"] < 0)
            & (x["delta_radius_rel"] >= 0)
        ),
        (
            (x["delta_center_rel"] < 0)
            & (x["delta_radius_rel"] < 0)
        ),
        (
            (x["delta_center_rel"] >= 0)
            & (x["delta_radius_rel"] < 0)
        ),
    ]

    labels = [
        "中心上移-半径扩大",
        "中心下移-半径扩大",
        "中心下移-半径收窄",
        "中心上移-半径收窄",
    ]

    x["quadrant"] = np.select(
        conditions,
        labels,
        default="边界",
    )

    return x


def make_stats(
    x: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    scopes = [
        ("ALL", x),
    ] + [
        (market, g)
        for market, g
        in x.groupby(
            "market"
        )
    ]

    for scope, g in scopes:
        rows.append({
            "scope": scope,
            "n": int(len(g)),

            "delta_center_rel_mean": float(
                g["delta_center_rel"].mean()
            ),
            "delta_center_rel_std": float(
                g["delta_center_rel"].std(
                    ddof=1
                )
            ),
            "delta_center_rel_median": float(
                g["delta_center_rel"].median()
            ),
            "delta_center_rel_q25": float(
                g["delta_center_rel"].quantile(
                    0.25
                )
            ),
            "delta_center_rel_q75": float(
                g["delta_center_rel"].quantile(
                    0.75
                )
            ),

            "delta_radius_rel_mean": float(
                g["delta_radius_rel"].mean()
            ),
            "delta_radius_rel_std": float(
                g["delta_radius_rel"].std(
                    ddof=1
                )
            ),
            "delta_radius_rel_median": float(
                g["delta_radius_rel"].median()
            ),
            "delta_radius_rel_q25": float(
                g["delta_radius_rel"].quantile(
                    0.25
                )
            ),
            "delta_radius_rel_q75": float(
                g["delta_radius_rel"].quantile(
                    0.75
                )
            ),

            # 辅助：新闻模型相对 Price-only
            # 对中心/半径误差的平均改善
            "center_error_improvement_mean": float(
                g[
                    "center_error_improvement"
                ].mean()
            ),
            "radius_error_improvement_mean": float(
                g[
                    "radius_error_improvement"
                ].mean()
            ),
        })

    return pd.DataFrame(
        rows
    )


def make_quadrant_stats(
    x: pd.DataFrame,
) -> pd.DataFrame:
    q = (
        x.groupby(
            [
                "market",
                "quadrant",
            ]
        )
        .size()
        .rename("n")
        .reset_index()
    )

    totals = (
        q.groupby("market")[
            "n"
        ]
        .transform("sum")
    )

    q["ratio_pct"] = (
        q["n"]
        / totals
        * 100.0
    )

    return q


# ============================================================
# 4. Plot
# ============================================================

def plot_figure(
    x: pd.DataFrame,
    out_dir: Path,
):
    setup_matplotlib_fonts()

    fig, ax = plt.subplots(
        figsize=(
            10.4,
            7.1,
        )
    )

    market_order = [
        "SSE",
        "SZI",
    ]

    labels = {
        "SSE": "上证指数",
        "SZI": "深证成指",
    }

    for market in market_order:
        g = x[
            x["market"] == market
        ]

        if len(g) == 0:
            continue

        ax.scatter(
            g["delta_center_rel"],
            g["delta_radius_rel"],
            s=34,
            alpha=0.56,
            label=labels.get(
                market,
                market,
            ),
        )

    ax.axhline(
        y=0,
        linewidth=0.9,
        alpha=0.70,
    )

    ax.axvline(
        x=0,
        linewidth=0.9,
        alpha=0.70,
    )

    ax.set_xlabel(
        "文本信息对区间中心的相对修正",
        fontsize=13,
    )

    ax.set_ylabel(
        "文本信息对区间半径的相对修正",
        fontsize=13,
    )

    ax.set_title(
        "文本信息对区间中心与半径的作用路径",
        fontsize=17,
        pad=12,
    )

    ax.grid(
        alpha=0.18,
        linewidth=0.6,
    )

    ax.legend(
        frameon=False,
        fontsize=12,
        loc="best",
    )

    ax.tick_params(
        axis="both",
        labelsize=11,
    )

    fig.tight_layout()

    png = (
        out_dir
        / "fig4_full_vs_priceonly_center_radius.png"
    )

    pdf = (
        out_dir
        / "fig4_full_vs_priceonly_center_radius.pdf"
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

    return png, pdf


# ============================================================
# 5. Main
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description=(
            "图4：Full Model vs Price-only "
            "中心/半径增量修正分析"
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
        "--full_dir",
        default=(
            "outputs_ablation_no_cross_attention"
        ),
    )

    p.add_argument(
        "--price_only_dir",
        default=(
            "outputs_ablation_price_only"
        ),
    )

    p.add_argument(
        "--out_dir",
        default=(
            "outputs_interpretability"
        ),
    )

    args = p.parse_args()

    root = Path(
        args.root
    )

    full_path = (
        root
        / args.full_dir
        / f"H{args.horizon}"
        / "test_predictions.csv"
    )

    price_only_path = (
        root
        / args.price_only_dir
        / f"H{args.horizon}"
        / "test_predictions.csv"
    )

    out_dir = (
        root
        / args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 80
    )
    print(
        "FIG.4 | FULL MODEL vs PRICE-ONLY"
    )
    print(
        "=" * 80
    )
    print(
        f"Full       : {full_path}"
    )
    print(
        f"Price-only : {price_only_path}"
    )
    print(
        f"Horizon    : H={args.horizon}"
    )
    print(
        f"Step       : {args.step}"
    )

    full = load_prediction_file(
        full_path,
        "Full model predictions",
    )

    price_only = load_prediction_file(
        price_only_path,
        "Price-only predictions",
    )

    # 只保留同一步预测
    full = full[
        full["horizon_step"]
        == args.step
    ].copy()

    price_only = price_only[
        price_only[
            "horizon_step"
        ] == args.step
    ].copy()

    x = align_full_and_priceonly(
        full,
        price_only,
    )

    x = add_base_close(
        root,
        x,
    )

    x = add_contribution_columns(
        x
    )

    # --------------------------------------------------------
    # 保存明细
    # --------------------------------------------------------
    data_path = (
        out_dir
        / "fig4_full_vs_priceonly_data.csv"
    )

    x.to_csv(
        data_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 保存描述统计
    # --------------------------------------------------------
    stats = make_stats(
        x
    )

    stats_path = (
        out_dir
        / "fig4_full_vs_priceonly_stats.csv"
    )

    stats.to_csv(
        stats_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 保存四象限统计
    # --------------------------------------------------------
    quadrant_stats = (
        make_quadrant_stats(
            x
        )
    )

    quadrant_path = (
        out_dir
        / "fig4_full_vs_priceonly_quadrant_stats.csv"
    )

    quadrant_stats.to_csv(
        quadrant_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 画图
    # --------------------------------------------------------
    png, pdf = plot_figure(
        x,
        out_dir,
    )

    # --------------------------------------------------------
    # 打印
    # --------------------------------------------------------
    print(
        "\n========== Fig.4 statistics =========="
    )
    print(
        stats.to_string(
            index=False
        )
    )

    print(
        "\n========== Quadrant distribution =========="
    )
    print(
        quadrant_stats.to_string(
            index=False
        )
    )

    print(
        "\n[SAVE]"
    )
    print(
        data_path
    )
    print(
        stats_path
    )
    print(
        quadrant_path
    )
    print(
        png
    )
    print(
        pdf
    )

    print(
        "\n论文解释口径："
    )
    print(
        "DeltaCenter = Full center - Price-only center"
    )
    print(
        "DeltaRadius = Full radius - Price-only radius"
    )
    print(
        "正/负值分别表示引入新闻后相应预测量上调/下调。"
    )
    print(
        "该结果应称为“模型级增量修正”，不要表述为因果效应。"
    )


if __name__ == "__main__":
    main()
