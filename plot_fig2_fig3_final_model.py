# -*- coding: utf-8 -*-
"""
plot_fig2_fig3_final_model.py

用途：
    基于最终模型 No-CrossAttention + Scalar Gate 的 test_predictions.csv，
    重新绘制论文图2（上证指数）和图3（深证成指）。

默认设置：
    - 使用 H=1 的测试结果；
    - horizon_step=1；
    - 若未指定日期，自动选择一个“代表性窗口”：
      在所有连续窗口中，计算平均区间绝对误差，
      选择误差最接近全体窗口中位数的窗口，
      避免人为挑选最好看的区间；
    - 每张图单独输出，不使用 subplot；
    - 输出 PNG（600 dpi）和 PDF；
    - 中文优先宋体，英文/数字优先 Times New Roman。

最终模型结果目录默认：
    E:\\Code\\stockv2\\outputs_ablation_no_cross_attention\\H1\\test_predictions.csv

运行示例：
    python plot_fig2_fig3_final_model.py

指定窗口长度：
    python plot_fig2_fig3_final_model.py --window 60

指定日期：
    python plot_fig2_fig3_final_model.py ^
        --start_date 2025-03-03 ^
        --end_date 2025-05-30

画 H=5 的第5步预测：
    python plot_fig2_fig3_final_model.py ^
        --horizon 5 ^
        --step 5

输出：
    E:\\Code\\stockv2\\paper_figures\\
        fig2_sse_interval_prediction_H1_step1.png
        fig2_sse_interval_prediction_H1_step1.pdf
        fig3_szi_interval_prediction_H1_step1.png
        fig3_szi_interval_prediction_H1_step1.pdf
        selected_windows.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager


# ============================================================
# 1. 字体
# ============================================================

def setup_fonts() -> None:
    """
    中文优先宋体；若本机没有宋体，则按顺序回退。
    不依赖外部字体文件。
    """
    available = {f.name for f in font_manager.fontManager.ttflist}

    chinese_candidates = [
        "SimSun",
        "宋体",
        "STSong",
        "Songti SC",
        "Microsoft YaHei",
        "SimHei",
    ]

    chinese_font = next(
        (x for x in chinese_candidates if x in available),
        None,
    )

    if chinese_font:
        plt.rcParams["font.sans-serif"] = [
            chinese_font,
            "Times New Roman",
        ]
    else:
        plt.rcParams["font.sans-serif"] = [
            "DejaVu Sans",
        ]

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10


# ============================================================
# 2. 数据读取
# ============================================================

def normalize_market(s: pd.Series) -> pd.Series:
    mapping = {
        "SH": "SSE",
        "SSE": "SSE",
        "SZ": "SZI",
        "SZI": "SZI",
    }
    return s.astype(str).str.upper().map(mapping).fillna(
        s.astype(str).str.upper()
    )


def load_predictions(
    root: Path,
    horizon: int,
) -> pd.DataFrame:
    path = (
        root
        / "outputs_ablation_no_cross_attention"
        / f"H{horizon}"
        / "test_predictions.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"未找到最终模型预测文件：\n{path}\n"
            "请确认已经运行 No-CrossAttention + Scalar Gate 最终模型。"
        )

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    required = [
        "market",
        "target_date",
        "horizon_step",
        "true_low",
        "true_high",
        "pred_low",
        "pred_high",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"test_predictions.csv 缺少字段：{missing}\n"
            f"当前字段：{list(df.columns)}"
        )

    df["market"] = normalize_market(
        df["market"]
    )

    df["target_date"] = pd.to_datetime(
        df["target_date"],
        errors="coerce",
    )

    numeric_cols = [
        "horizon_step",
        "true_low",
        "true_high",
        "pred_low",
        "pred_high",
    ]

    for c in numeric_cols:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "target_date",
            "horizon_step",
            "true_low",
            "true_high",
            "pred_low",
            "pred_high",
        ]
    ).copy()

    df["horizon_step"] = (
        df["horizon_step"]
        .astype(int)
    )

    return df.sort_values(
        ["market", "target_date", "horizon_step"]
    ).reset_index(drop=True)


# ============================================================
# 3. 代表性窗口选择
# ============================================================

def interval_abs_error(df: pd.DataFrame) -> np.ndarray:
    """
    与 IMAE 的单期定义一致：
        (|Lhat-L| + |Uhat-U|)/2
    """
    return (
        np.abs(
            df["pred_low"].to_numpy()
            - df["true_low"].to_numpy()
        )
        + np.abs(
            df["pred_high"].to_numpy()
            - df["true_high"].to_numpy()
        )
    ) / 2.0


def select_representative_window(
    df: pd.DataFrame,
    window: int,
) -> Tuple[pd.DataFrame, dict]:
    """
    科研上尽量避免 cherry-picking：
    计算所有连续窗口的平均区间绝对误差，
    选择最接近这些窗口误差“中位数”的窗口。

    这样选出的不是最好窗口，也不是最差窗口，
    而是测试集中的典型表现窗口。
    """
    g = df.sort_values(
        "target_date"
    ).reset_index(drop=True).copy()

    if len(g) <= window:
        info = {
            "selection_method": "all_available",
            "window_size": len(g),
            "window_error": float(
                interval_abs_error(g).mean()
            ),
        }
        return g, info

    err = interval_abs_error(g)

    rolling_mean = (
        pd.Series(err)
        .rolling(
            window=window,
            min_periods=window,
        )
        .mean()
        .to_numpy()
    )

    valid_end = np.arange(
        window - 1,
        len(g),
    )

    valid_errors = rolling_mean[
        valid_end
    ]

    median_error = float(
        np.median(valid_errors)
    )

    best_pos = int(
        np.argmin(
            np.abs(
                valid_errors
                - median_error
            )
        )
    )

    end_idx = int(
        valid_end[best_pos]
    )

    start_idx = (
        end_idx - window + 1
    )

    selected = g.iloc[
        start_idx:end_idx + 1
    ].copy()

    info = {
        "selection_method": (
            "rolling_window_error_closest_to_median"
        ),
        "window_size": window,
        "window_error": float(
            valid_errors[best_pos]
        ),
        "median_window_error": (
            median_error
        ),
    }

    return selected, info


def choose_window(
    df: pd.DataFrame,
    market: str,
    step: int,
    window: int,
    start_date: Optional[str],
    end_date: Optional[str],
) -> Tuple[pd.DataFrame, dict]:
    g = df[
        (df["market"] == market)
        & (df["horizon_step"] == step)
    ].copy()

    if g.empty:
        raise ValueError(
            f"没有找到 market={market}, horizon_step={step} 的数据。"
        )

    g = g.sort_values(
        "target_date"
    ).reset_index(drop=True)

    if start_date or end_date:
        if start_date:
            g = g[
                g["target_date"]
                >= pd.Timestamp(start_date)
            ]

        if end_date:
            g = g[
                g["target_date"]
                <= pd.Timestamp(end_date)
            ]

        if g.empty:
            raise ValueError(
                f"{market} 指定日期区间内无数据。"
            )

        info = {
            "selection_method": (
                "user_defined_date_range"
            ),
            "window_size": len(g),
            "window_error": float(
                interval_abs_error(g).mean()
            ),
        }

        return g.copy(), info

    return select_representative_window(
        g,
        window=window,
    )


# ============================================================
# 4. 绘图
# ============================================================

def draw_interval_figure(
    df: pd.DataFrame,
    market_label: str,
    figure_no: int,
    horizon: int,
    step: int,
    out_dir: Path,
) -> None:
    """
    每张图独立绘制：
    - 真实上下界：两条实线
    - 预测上下界：两条虚线
    - 真实区间：浅色填充
    - 预测区间：更浅填充

    不指定固定颜色，使用 Matplotlib 默认颜色循环。
    """
    x = df["target_date"]

    true_low = df["true_low"].to_numpy()
    true_high = df["true_high"].to_numpy()
    pred_low = df["pred_low"].to_numpy()
    pred_high = df["pred_high"].to_numpy()

    fig, ax = plt.subplots(
        figsize=(10.5, 5.4)
    )

    # 先绘制两组区间带
    ax.fill_between(
        x,
        true_low,
        true_high,
        alpha=0.16,
        label="真实价格区间",
    )

    ax.fill_between(
        x,
        pred_low,
        pred_high,
        alpha=0.10,
        label="预测价格区间",
    )

    # 再绘制边界线
    ax.plot(
        x,
        true_high,
        linewidth=1.5,
        label="真实上界",
    )

    ax.plot(
        x,
        true_low,
        linewidth=1.5,
        label="真实下界",
    )

    ax.plot(
        x,
        pred_high,
        linewidth=1.35,
        linestyle="--",
        label="预测上界",
    )

    ax.plot(
        x,
        pred_low,
        linewidth=1.35,
        linestyle="--",
        label="预测下界",
    )

    # 论文图一般不在图内重复写“图2”
    ax.set_title(
        f"{market_label}真实价格区间与预测价格区间对比"
    )

    ax.set_xlabel(
        "交易日期"
    )

    ax.set_ylabel(
        "指数点位"
    )

    # 时间轴
    locator = mdates.AutoDateLocator(
        minticks=5,
        maxticks=9,
    )

    formatter = mdates.ConciseDateFormatter(
        locator
    )

    ax.xaxis.set_major_locator(
        locator
    )

    ax.xaxis.set_major_formatter(
        formatter
    )

    ax.grid(
        axis="y",
        alpha=0.20,
        linewidth=0.6,
    )

    ax.legend(
        loc="best",
        frameon=False,
        ncol=2,
    )

    fig.tight_layout()

    prefix = (
        f"fig{figure_no}_"
        f"{'sse' if market_label == '上证指数' else 'szi'}_"
        f"interval_prediction_"
        f"H{horizon}_step{step}"
    )

    png_path = (
        out_dir
        / f"{prefix}.png"
    )

    pdf_path = (
        out_dir
        / f"{prefix}.pdf"
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[SAVE] {png_path}"
    )
    print(
        f"[SAVE] {pdf_path}"
    )


# ============================================================
# 5. 主程序
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Generate paper Fig.2 and Fig.3 "
            "from final NoCross + Scalar Gate model."
        )
    )

    p.add_argument(
        "--root",
        type=str,
        default=r"E:\Code\stockv2",
    )

    p.add_argument(
        "--horizon",
        type=int,
        default=1,
        choices=[1, 3, 5],
        help="读取哪个 H 模型目录。",
    )

    p.add_argument(
        "--step",
        type=int,
        default=1,
        help=(
            "画 H 模型中的第几个预测步。"
            "例如 H=5 且希望画第5日预测，使用 --step 5。"
        ),
    )

    p.add_argument(
        "--window",
        type=int,
        default=60,
        help="自动选择代表性窗口时包含的交易日数量。",
    )

    p.add_argument(
        "--start_date",
        type=str,
        default="",
        help="可选，例如 2025-03-03。",
    )

    p.add_argument(
        "--end_date",
        type=str,
        default="",
        help="可选，例如 2025-05-30。",
    )

    p.add_argument(
        "--out_dir",
        type=str,
        default="paper_figures",
    )

    return p.parse_args()


def main():
    args = parse_args()

    if args.step < 1 or args.step > args.horizon:
        raise ValueError(
            f"--step 必须在 1 到 {args.horizon} 之间。"
        )

    if args.window < 20:
        raise ValueError(
            "--window 建议至少 20 个交易日。"
        )

    setup_fonts()

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

    df = load_predictions(
        root=root,
        horizon=args.horizon,
    )

    start_date = (
        args.start_date.strip()
        or None
    )

    end_date = (
        args.end_date.strip()
        or None
    )

    selected_info = []

    # --------------------------------------------------------
    # 图2：上证指数
    # --------------------------------------------------------
    sse, sse_info = choose_window(
        df=df,
        market="SSE",
        step=args.step,
        window=args.window,
        start_date=start_date,
        end_date=end_date,
    )

    draw_interval_figure(
        df=sse,
        market_label="上证指数",
        figure_no=2,
        horizon=args.horizon,
        step=args.step,
        out_dir=out_dir,
    )

    selected_info.append({
        "figure": "图2",
        "market": "SSE",
        "horizon": args.horizon,
        "horizon_step": args.step,
        "start_date": str(
            sse["target_date"]
            .min()
            .date()
        ),
        "end_date": str(
            sse["target_date"]
            .max()
            .date()
        ),
        **sse_info,
    })

    # --------------------------------------------------------
    # 图3：深证成指
    # --------------------------------------------------------
    szi, szi_info = choose_window(
        df=df,
        market="SZI",
        step=args.step,
        window=args.window,
        start_date=start_date,
        end_date=end_date,
    )

    draw_interval_figure(
        df=szi,
        market_label="深证成指",
        figure_no=3,
        horizon=args.horizon,
        step=args.step,
        out_dir=out_dir,
    )

    selected_info.append({
        "figure": "图3",
        "market": "SZI",
        "horizon": args.horizon,
        "horizon_step": args.step,
        "start_date": str(
            szi["target_date"]
            .min()
            .date()
        ),
        "end_date": str(
            szi["target_date"]
            .max()
            .date()
        ),
        **szi_info,
    })

    info_df = pd.DataFrame(
        selected_info
    )

    info_path = (
        out_dir
        / "selected_windows.csv"
    )

    info_df.to_csv(
        info_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n========== Selected windows =========="
    )

    print(
        info_df.to_string(
            index=False
        )
    )

    print(
        f"\n[SAVE] {info_path}"
    )


if __name__ == "__main__":
    main()
