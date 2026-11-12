# -*- coding: utf-8 -*-
"""
run_robustness_market_state.py

对应原稿 4.5（2）不同市场状态分析 / 表4。
不重新训练模型。

保留原稿两类状态：
    1) 波动性：低波动 / 高波动
       按训练期 amplitude_pct 中位数划分。
    2) 趋势：上行 / 下行
       按预测基准日 t 的 change_pct 正负划分。

重要：
    分组只使用预测时已经可见的 as-of date t 市场状态，
    不使用未来 t+1 的涨跌，避免未来信息泄漏。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from postexp_utils import (
    load_final_model_context,
    infer_loader,
    load_raw_market_states,
    merge_predictions_states,
    grouped_metrics,
)


def main():
    p = argparse.ArgumentParser()

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
        "--model_dir",
        default="outputs_ablation_no_cross_attention",
    )
    p.add_argument(
        "--out_dir",
        default="outputs_robustness_market_state",
    )
    p.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
    )

    args = p.parse_args()
    root = Path(args.root)
    out_dir = root / args.out_dir
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        loaders["test"],
        device,
    )

    # H>1 时整体表可能包含多个 step；
    # 原稿稳健性默认 H=1。
    if args.horizon > 1:
        pred = pred[
            pred["horizon_step"] == 1
        ].copy()

    states = load_raw_market_states(
        root
    )

    train_states = states[
        states["date"]
        <= prepared.strict_train_asof_end
    ].copy()

    # 每个市场分别使用训练期中位数，
    # 避免不同指数水平/交易结构造成阈值混淆。
    thresholds = (
        train_states
        .groupby("market")[
            "amplitude_pct"
        ]
        .median()
        .to_dict()
    )

    x = merge_predictions_states(
        pred,
        states,
    )

    x["volatility_state"] = [
        (
            "低波动"
            if amp <= thresholds[m]
            else "高波动"
        )
        for amp, m in zip(
            x["amplitude_pct"],
            x["market"],
        )
    ]

    x["trend_state"] = x[
        "change_pct"
    ].apply(
        lambda v: (
            "上行"
            if v >= 0
            else "下行"
        )
    )

    vol_metrics = grouped_metrics(
        x,
        "volatility_state",
    )
    vol_metrics[
        "market_state"
    ] = (
        "波动性-"
        + vol_metrics[
            "volatility_state"
        ].astype(str)
    )

    trend_metrics = grouped_metrics(
        x,
        "trend_state",
    )
    trend_metrics[
        "market_state"
    ] = (
        "趋势-"
        + trend_metrics[
            "trend_state"
        ].astype(str)
    )

    table4 = pd.concat(
        [
            vol_metrics,
            trend_metrics,
        ],
        ignore_index=True,
    )

    cols = [
        "market_state",
        "scope",
        "n",
        "IMAPE_pct",
        "IRMSE",
        "IMAE",
        "center_MAE",
        "radius_MAE",
        "violation_rate",
    ]

    table4 = table4[
        [
            c for c in cols
            if c in table4.columns
        ]
    ]

    table4.to_csv(
        out_dir
        / "table4_market_state_robustness.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [
            {
                "market": k,
                "train_amplitude_median": v,
            }
            for k, v in thresholds.items()
        ]
    ).to_csv(
        out_dir
        / "market_state_thresholds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    x.to_csv(
        out_dir
        / "test_predictions_with_market_states.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n========== Table 4 =========="
    )
    print(
        table4.to_string(index=False)
    )


if __name__ == "__main__":
    main()
