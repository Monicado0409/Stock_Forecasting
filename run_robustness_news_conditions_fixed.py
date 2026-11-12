# -*- coding: utf-8 -*-
"""
run_robustness_news_conditions.py

对应原稿 4.5（3）不同新闻状态分析 / 表5。

保留原结构：
    - 新闻密集
    - 新闻稀疏
    - 加入新闻属性噪声

更新点：
    1) 新闻密度定义为模型实际输入窗口 t-L ... t-1
       内的平均每日新闻数。
    2) 测试集唯一预测日期按新闻密度排序后进行近似50/50中位二分，避免并列值导致某一组为空。
    3) 噪声不再只扰动 sentiment，
       而是同时扰动：
           sentiment / impact / relevance
       扰动标准差 = 0.1 × 训练标准差。
    4) 测试扰动重复 20 次，报告均值与标准差。
    5) semantic PCA16 不扰动。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stockv2_data_tune import (
    load_news_raw,
)

from run_stockv2_tune import (
    metric_record,
)

from postexp_utils import (
    load_final_model_context,
    infer_loader,
)


def build_news_count_series(
    root: Path,
):
    raw = load_news_raw(
        root
    )
    items = raw.items.copy()
    items["date"] = pd.to_datetime(
        items["date"]
    )

    counts = (
        items
        .groupby("date")
        .size()
        .to_dict()
    )
    return counts


def sample_density(
    sample,
    trading_dates,
    date_to_pos,
    counts,
    lookback,
):
    asof = pd.Timestamp(
        sample.as_of_date
    )
    i = date_to_pos[asof]

    # 与模型新闻输入严格一致：
    # t-L ... t-1
    news_dates = trading_dates[
        i - lookback:i
    ]

    vals = [
        counts.get(
            pd.Timestamp(d),
            0,
        )
        for d in news_dates
    ]

    return float(
        np.mean(vals)
    )


def attach_density(
    pred,
    dataset,
    prepared,
    counts,
    lookback,
):
    dates = sorted(
        prepared.price_panel.loc[
            prepared.price_panel[
                "market"
            ] == "SSE",
            "date",
        ].unique()
    )
    dates = [
        pd.Timestamp(x)
        for x in dates
    ]
    date_to_pos = {
        d: i
        for i, d in enumerate(dates)
    }

    # dataset 中 SSE/SZI 同一 as-of 日期各一条，
    # 新闻池是共享的，因此按 as_of_date 生成密度即可。
    density_map = {}

    for s in dataset.samples:
        d = pd.Timestamp(
            s.as_of_date
        )
        if d not in density_map:
            density_map[d] = sample_density(
                s,
                dates,
                date_to_pos,
                counts,
                lookback,
            )

    out = pred.copy()
    out["as_of_date"] = pd.to_datetime(
        out["as_of_date"]
    )
    out["news_density"] = out[
        "as_of_date"
    ].map(density_map)

    if out["news_density"].isna().any():
        raise ValueError(
            "存在未匹配的新闻密度。"
        )

    return out


def condition_metrics(
    df,
    label,
):
    rows = []
    for market, g in df.groupby(
        "market"
    ):
        r = metric_record(
            market,
            g,
        )
        r["news_condition"] = label
        rows.append(r)
    return rows


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
        default="outputs_robustness_news_conditions",
    )
    p.add_argument(
        "--noise_strength",
        type=float,
        default=0.10,
    )
    p.add_argument(
        "--noise_repeats",
        type=int,
        default=20,
    )
    p.add_argument(
        "--noise_seed",
        type=int,
        default=2026,
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

    lookback = config["lookback"]

    # --------------------------------------------------------
    # A. 新闻密集 / 稀疏
    # --------------------------------------------------------
    baseline_pred = infer_loader(
        model,
        loaders["test"],
        device,
    )

    if args.horizon > 1:
        baseline_pred = baseline_pred[
            baseline_pred[
                "horizon_step"
            ] == 1
        ].copy()

    counts = build_news_count_series(
        root
    )

    # --------------------------------------------------------
    # 新闻密集 / 新闻稀疏分组
    #
    # 与原稿保持一致：基于测试集新闻密度进行中位分组。
    # 由于每日新闻数量可能大量并列，若直接使用 > median，
    # 可能导致某一组为空。因此先对“唯一 as_of_date”
    # 按 news_density 排序，再按样本数做近似 50/50 二分：
    #
    #   前半部分 = 新闻稀疏
    #   后半部分 = 新闻密集
    #
    # 分组仅使用新闻数量，不使用未来价格、预测误差或标签，
    # 不引入目标信息泄漏。
    # --------------------------------------------------------

    test_density = attach_density(
        baseline_pred,
        datasets["test"],
        prepared,
        counts,
        lookback,
    )

    test_density["as_of_date"] = pd.to_datetime(
        test_density["as_of_date"]
    )

    # 同一 as_of_date 在 SSE/SZI 中共享同一新闻窗口，
    # 先按唯一日期进行分组，再映射回两个市场。
    unique_dates = (
        test_density[
            ["as_of_date", "news_density"]
        ]
        .drop_duplicates(
            subset=["as_of_date"]
        )
        .sort_values(
            ["news_density", "as_of_date"]
        )
        .reset_index(drop=True)
    )

    n_dates = len(unique_dates)
    if n_dates < 2:
        raise ValueError(
            "测试集唯一预测日期不足2个，无法划分新闻密集/稀疏组。"
        )

    split_pos = n_dates // 2

    unique_dates["news_condition"] = "新闻密集"
    unique_dates.loc[
        unique_dates.index < split_pos,
        "news_condition",
    ] = "新闻稀疏"

    condition_map = dict(
        zip(
            unique_dates["as_of_date"],
            unique_dates["news_condition"],
        )
    )

    test_density["news_condition"] = (
        test_density["as_of_date"]
        .map(condition_map)
    )

    if test_density["news_condition"].isna().any():
        raise ValueError(
            "存在未匹配的新闻密集/稀疏标签。"
        )

    # 保存分组边界，方便论文复现。
    density_threshold = float(
        unique_dates["news_density"].median()
    )

    sparse_max = float(
        unique_dates.loc[
            unique_dates["news_condition"] == "新闻稀疏",
            "news_density",
        ].max()
    )

    dense_min = float(
        unique_dates.loc[
            unique_dates["news_condition"] == "新闻密集",
            "news_density",
        ].min()
    )

    table_rows = []

    for condition in [
        "新闻密集",
        "新闻稀疏",
    ]:
        sub = test_density[
            test_density[
                "news_condition"
            ] == condition
        ]
        table_rows.extend(
            condition_metrics(
                sub,
                condition,
            )
        )

    # --------------------------------------------------------
    # B. 新闻属性噪声
    # --------------------------------------------------------
    structured_cols = [
        c
        for c in [
            "sentiment",
            "impact",
            "relevance",
        ]
        if c in prepared.news_cols
    ]
    idx = [
        prepared.news_cols.index(c)
        for c in structured_cols
    ]

    scaler = prepared.news_scaler

    mean = torch.tensor(
        scaler.mean_[idx],
        dtype=torch.float32,
        device=device,
    ).view(1, 1, -1)

    scale = torch.tensor(
        scaler.scale_[idx],
        dtype=torch.float32,
        device=device,
    ).view(1, 1, -1)

    news_train = prepared.news_daily[
        prepared.news_daily[
            "date"
        ] <= prepared.train_news_end
    ]

    train_min = torch.tensor(
        news_train[
            structured_cols
        ].min().to_numpy(
            dtype=np.float32
        ),
        device=device,
    ).view(1, 1, -1)

    train_max = torch.tensor(
        news_train[
            structured_cols
        ].max().to_numpy(
            dtype=np.float32
        ),
        device=device,
    ).view(1, 1, -1)

    noise_records = []

    for rep in range(
        args.noise_repeats
    ):
        seed = (
            args.noise_seed
            + rep
        )
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                seed
            )

        def transform(news_seq):
            out = news_seq.clone()

            x_scaled = out[
                :, :, idx
            ]

            x_raw = (
                x_scaled * scale
                + mean
            )

            noise = torch.randn(
                x_raw.shape,
                dtype=x_raw.dtype,
                device=x_raw.device,
            ) * (
                args.noise_strength
                * scale
            )

            x_noisy = (
                x_raw + noise
            )

            x_noisy = torch.maximum(
                torch.minimum(
                    x_noisy,
                    train_max,
                ),
                train_min,
            )

            out[:, :, idx] = (
                x_noisy - mean
            ) / scale

            return out

        pred_noise = infer_loader(
            model,
            loaders["test"],
            device,
            news_transform=transform,
        )

        if args.horizon > 1:
            pred_noise = pred_noise[
                pred_noise[
                    "horizon_step"
                ] == 1
            ].copy()

        for market, g in pred_noise.groupby(
            "market"
        ):
            r = metric_record(
                market,
                g,
            )
            r["repeat"] = rep + 1
            r["noise_seed"] = seed
            noise_records.append(r)

        print(
            f"[Noise] repeat "
            f"{rep + 1}/{args.noise_repeats}"
        )

    noise_df = pd.DataFrame(
        noise_records
    )

    noise_df.to_csv(
        out_dir
        / "news_attribute_noise_repetitions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metric_cols = [
        "IMAPE_pct",
        "IRMSE",
        "IMAE",
        "center_MAE",
        "radius_MAE",
        "violation_rate",
    ]

    noise_summary = (
        noise_df
        .groupby("scope")[
            metric_cols
        ]
        .agg(["mean", "std"])
    )
    noise_summary.columns = [
        f"{a}_{b}"
        for a, b
        in noise_summary.columns
    ]
    noise_summary = (
        noise_summary
        .reset_index()
    )

    noise_summary.to_csv(
        out_dir
        / "news_attribute_noise_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 表5使用噪声重复实验的均值
    for row in noise_summary.to_dict(
        orient="records"
    ):
        table_rows.append({
            "news_condition": (
                "加入新闻属性噪声"
            ),
            "scope": row["scope"],
            "n": int(
                len(
                    baseline_pred[
                        baseline_pred[
                            "market"
                        ] == row["scope"]
                    ]
                )
            ),
            "IMAPE_pct": row[
                "IMAPE_pct_mean"
            ],
            "IRMSE": row[
                "IRMSE_mean"
            ],
            "IMAE": row[
                "IMAE_mean"
            ],
            "center_MAE": row[
                "center_MAE_mean"
            ],
            "radius_MAE": row[
                "radius_MAE_mean"
            ],
            "violation_rate": row[
                "violation_rate_mean"
            ],
        })

    table5 = pd.DataFrame(
        table_rows
    )

    table5.to_csv(
        out_dir
        / "table5_news_condition_robustness.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_density.to_csv(
        out_dir
        / "test_predictions_with_news_density.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 保存每个测试日期的新闻密度分组
    unique_dates.to_csv(
        out_dir
        / "news_density_date_groups.csv",
        index=False,
        encoding="utf-8-sig",
    )

    group_counts = (
        unique_dates["news_condition"]
        .value_counts()
        .to_dict()
    )

    pd.DataFrame([
        {
            "density_grouping": (
                "test_unique_asof_dates_rank_median_split"
            ),
            "test_news_density_median": (
                density_threshold
            ),
            "sparse_max_density": (
                sparse_max
            ),
            "dense_min_density": (
                dense_min
            ),
            "sparse_unique_dates": int(
                group_counts.get(
                    "新闻稀疏",
                    0,
                )
            ),
            "dense_unique_dates": int(
                group_counts.get(
                    "新闻密集",
                    0,
                )
            ),
            "lookback": lookback,
            "noise_strength": (
                args.noise_strength
            ),
            "noise_repeats": (
                args.noise_repeats
            ),
        }
    ]).to_csv(
        out_dir
        / "news_condition_settings.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n========== Table 5 =========="
    )
    print(
        table5.to_string(index=False)
    )


if __name__ == "__main__":
    main()
