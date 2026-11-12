# -*- coding: utf-8 -*-
r"""
run_robustness_news_lag.py

新闻时间滞后稳健性实验（H=1）：
    Lag-1 / Lag-2 / Lag-3

对应当前最终模型：
    Price GRU
    + News GRU
    + Scalar Information Absorption Gate
    + Residual Fusion
    + Center/Radius heads
    无 Cross-Attention

正式主模型时间协议：
    Price : t-L+1 ... t
    News  : t-L   ... t-1
    Target: t+1

本脚本仅改变新闻窗口相对预测基准日 t 的截止位置：

    Lag-1:
        News = t-L   ... t-1

    Lag-2:
        News = t-L-1 ... t-2

    Lag-3:
        News = t-L-2 ... t-3

其余设置全部固定。

------------------------------------------------------------
为什么使用“共同样本日历”？
------------------------------------------------------------
若直接按每个 lag 的最低历史需求构造样本，
Lag-2 会比 Lag-1 少最前面的 1 个训练日，
Lag-3 会少最前面的 2 个训练日。

为了让 Lag-1 / Lag-2 / Lag-3 的比较只反映“新闻时效性”，
本脚本默认使用 max(lags) 决定共同起始位置，
使三个模型拥有完全相同的 train / val / test 样本日期。

因此：
    Lag-1 在本稳健性实验中的结果可能与论文主表中的正式 Lag-1
    有极小差异，因为这里为了公平比较，统一去掉了最早的少量训练样本。

这属于更严格的对照设计。

------------------------------------------------------------
PCA / Scaler
------------------------------------------------------------
每个 Lag 模型都会重新调用当前正式 prepare_for_horizon()：
    - PCA 仅使用训练期新闻拟合
    - price/news/state scaler 仅使用训练期拟合
    - split 日期与正式模型一致
    - H=1 无多步边界重叠问题

PCA / scaler 的训练期定义对三个 lag 保持一致，
以尽量做到“只改变新闻窗口位置”。

------------------------------------------------------------
运行示例
------------------------------------------------------------
cd /d E:\Code\stockv2

python run_robustness_news_lag.py ^
    --lags 1,2,3 ^
    --device cuda

主要输出：
    outputs_robustness_news_lag/
        table_news_lag_robustness.csv
        table_news_lag_vs_lag1.csv
        lag_protocol_audit.csv

每个 lag 的完整训练结果：
    outputs_robustness_news_lag/Lag1/H1/
    outputs_robustness_news_lag/Lag2/H1/
    outputs_robustness_news_lag/Lag3/H1/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import pandas as pd
import torch

import run_ablation_no_cross_attention as final_train

from stockv2_data_tune import (
    PRICE_COLS,
    STATE_COLS,
)

from run_stockv2_tune import (
    Sample,
    StockIntervalDataset,
)


# ============================================================
# 1. Lagged dataset
# ============================================================

class LaggedStockIntervalDataset(
    StockIntervalDataset
):
    """
    与正式 StockIntervalDataset 保持一致，
    唯一实质变化：
        news_seq 的时间位置向过去移动。

    为保证 Lag-1/2/3 公平比较，
    all variants use the same common_start_i.
    """

    def __init__(
        self,
        prepared,
        split: str,
        lookback: int,
        horizon: int,
        news_lag: int,
        common_max_lag: int,
    ) -> None:

        if split not in {
            "train",
            "val",
            "test",
        }:
            raise ValueError(split)

        if horizon != 1:
            raise ValueError(
                "本脚本专门用于 H=1 新闻滞后稳健性实验。"
            )

        if news_lag < 1:
            raise ValueError(
                "news_lag 必须 >= 1"
            )

        if common_max_lag < news_lag:
            raise ValueError(
                "common_max_lag 不能小于 news_lag"
            )

        self.samples: List[Sample] = []

        price = prepared.price_panel
        news = prepared.news_daily
        news_cols = prepared.news_cols

        train_end = pd.Timestamp(
            prepared.train_end
        )
        val_end = pd.Timestamp(
            prepared.val_end
        )

        # ----------------------------------------------------
        # 统一三个 lag 的最早可用 as-of index。
        #
        # 例如 L=20, max_lag=3：
        #     common_start_i = 22
        #
        # Lag1 news: i-20 ... i-1
        # Lag2 news: i-21 ... i-2
        # Lag3 news: i-22 ... i-3
        #
        # 三者都恰好 20 天。
        # ----------------------------------------------------
        common_start_i = (
            lookback
            + common_max_lag
            - 1
        )

        lag_offset = (
            news_lag - 1
        )

        for market in (
            "SSE",
            "SZI",
        ):
            p = (
                price[
                    price["market"]
                    == market
                ]
                .sort_values("date")
                .reset_index(drop=True)
            )

            n = (
                news[
                    news["market"]
                    == market
                ]
                .sort_values("date")
                .reset_index(drop=True)
            )

            if not np.array_equal(
                p["date"].to_numpy(
                    dtype="datetime64[ns]"
                ),
                n["date"].to_numpy(
                    dtype="datetime64[ns]"
                ),
            ):
                raise ValueError(
                    f"{market} 的价格和新闻日期未严格对齐。"
                )

            p_scaled = (
                prepared
                .price_scaler
                .transform(
                    p[
                        PRICE_COLS
                    ].astype(float)
                )
                .astype(np.float32)
            )

            n_scaled = (
                prepared
                .news_scaler
                .transform(
                    n[
                        news_cols
                    ].astype(float)
                )
                .astype(np.float32)
            )

            f_scaled = (
                prepared
                .state_scaler
                .transform(
                    p[
                        STATE_COLS
                    ].astype(float)
                )
                .astype(np.float32)
            )

            dates = pd.to_datetime(
                p["date"]
            ).tolist()

            date_to_pos = {
                pd.Timestamp(d): i
                for i, d
                in enumerate(dates)
            }

            train_end_pos = (
                date_to_pos[
                    train_end
                ]
            )

            val_end_pos = (
                date_to_pos[
                    val_end
                ]
            )

            close = (
                p["close"]
                .astype(float)
                .to_numpy()
            )

            y_low = (
                p["interval_low"]
                .astype(float)
                .to_numpy()
            )

            y_high = (
                p["interval_high"]
                .astype(float)
                .to_numpy()
            )

            y_center = (
                p["interval_center"]
                .astype(float)
                .to_numpy()
            )

            y_radius = (
                p["interval_radius"]
                .astype(float)
                .to_numpy()
            )

            # ------------------------------------------------
            # 所有 lag 共享相同 i 范围。
            # ------------------------------------------------
            for i in range(
                common_start_i,
                len(p) - horizon,
            ):
                as_of = pd.Timestamp(
                    dates[i]
                )

                target_last_pos = (
                    i + horizon
                )

                # 与正式代码完全相同的 split / purge 规则
                if split == "train":
                    if not (
                        as_of <= train_end
                        and target_last_pos
                        <= train_end_pos + 1
                    ):
                        continue

                elif split == "val":
                    if not (
                        train_end
                        < as_of
                        <= val_end
                        and target_last_pos
                        <= val_end_pos + 1
                    ):
                        continue

                else:
                    if not (
                        as_of > val_end
                    ):
                        continue

                t0 = i + 1
                t1 = (
                    i + 1 + horizon
                )

                # --------------------------------------------
                # 核心变化：新闻窗口
                #
                # Lag-k:
                #   start = i - L - (k-1)
                #   stop  = i - (k-1)   [exclusive]
                #
                # Lag1 = i-L     : i
                # Lag2 = i-L-1   : i-1
                # Lag3 = i-L-2   : i-2
                # --------------------------------------------
                news_start = (
                    i
                    - lookback
                    - lag_offset
                )
                news_stop = (
                    i
                    - lag_offset
                )

                news_seq = n_scaled[
                    news_start:
                    news_stop
                ]

                if (
                    len(news_seq)
                    != lookback
                ):
                    raise RuntimeError(
                        f"{market} "
                        f"{as_of.date()} "
                        f"Lag-{news_lag} "
                        f"新闻窗口长度异常："
                        f"{len(news_seq)} "
                        f"!= {lookback}"
                    )

                self.samples.append(
                    Sample(
                        market=market,
                        as_of_date=as_of,
                        target_dates=[
                            pd.Timestamp(d)
                            for d
                            in dates[t0:t1]
                        ],
                        base_close=float(
                            close[i]
                        ),

                        # Price 不变：
                        # t-L+1 ... t
                        price_seq=p_scaled[
                            i - lookback + 1:
                            i + 1
                        ],

                        # News 改为指定 lag
                        news_seq=news_seq,

                        # 当前 t 日市场状态不变
                        market_state=(
                            f_scaled[i]
                        ),

                        y_low=y_low[
                            t0:t1
                        ].astype(
                            np.float32
                        ),

                        y_high=y_high[
                            t0:t1
                        ].astype(
                            np.float32
                        ),

                        y_center=y_center[
                            t0:t1
                        ].astype(
                            np.float32
                        ),

                        y_radius=y_radius[
                            t0:t1
                        ].astype(
                            np.float32
                        ),
                    )
                )


# ============================================================
# 2. CLI args passed to current final training code
# ============================================================

def make_train_args(
    cli,
    lag: int,
):
    return SimpleNamespace(
        root=cli.root,

        # train_one_horizon 会再追加 /H1
        out_dir=(
            f"{cli.out_dir}/Lag{lag}"
        ),

        horizons="1",

        # 当前正式配置
        lookback=20,
        pca_dim=16,
        train_ratio=0.70,
        val_ratio=0.15,
        hidden_dim=64,
        market_emb_dim=8,
        num_layers=1,
        dropout=0.20,

        # 旧参数保留兼容，
        # 当前 No-Cross 模型不会使用 attention
        attn_heads=4,

        lambda_boundary=1.0,
        lambda_center_radius=0.5,
        lambda_news_smooth=0.005,
        huber_delta=0.01,

        epochs=cli.epochs,
        batch_size=64,
        lr=3e-4,
        weight_decay=1e-4,
        patience=cli.patience,
        grad_clip=1.0,

        seed=42,
        device=cli.device,
        print_every=10,
    )


# ============================================================
# 3. Helpers
# ============================================================

def parse_lags(
    s: str,
) -> List[int]:
    lags = sorted(
        set(
            int(x.strip())
            for x in s.split(",")
            if x.strip()
        )
    )

    if not lags:
        raise ValueError(
            "lags 不能为空。"
        )

    if min(lags) < 1:
        raise ValueError(
            "lag 必须 >= 1。"
        )

    return lags


def patch_summary_json(
    path: Path,
    lag: int,
    common_max_lag: int,
):
    """
    给原训练脚本生成的 summary 加入本实验审计信息。
    """
    if not path.exists():
        return

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        obj = json.load(f)

    obj[
        "robustness_experiment"
    ] = {
        "name": (
            "news_time_lag"
        ),
        "news_lag": lag,
        "common_max_lag": (
            common_max_lag
        ),
        "horizon": 1,
        "price_protocol": (
            "t-L+1 ... t"
        ),
        "news_protocol": (
            f"t-L-{lag-1} ... t-{lag}"
        ),
        "target_protocol": (
            "t+1"
        ),
        "common_sample_calendar": True,
        "cross_attention": False,
        "all_other_hyperparameters_fixed": True,
    }

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# 4. Main
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description=(
            "H=1 新闻时间滞后稳健性："
            "Lag-1 / Lag-2 / Lag-3"
        )
    )

    p.add_argument(
        "--root",
        type=str,
        default=r"E:\Code\stockv2",
    )

    p.add_argument(
        "--out_dir",
        type=str,
        default=(
            "outputs_robustness_news_lag"
        ),
    )

    p.add_argument(
        "--lags",
        type=str,
        default="1,2,3",
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=200,
    )

    p.add_argument(
        "--patience",
        type=int,
        default=25,
    )

    p.add_argument(
        "--device",
        choices=[
            "cuda",
            "cpu",
        ],
        default="cuda",
    )

    cli = p.parse_args()

    lags = parse_lags(
        cli.lags
    )

    common_max_lag = max(
        lags
    )

    root = Path(
        cli.root
    )

    out_root = (
        root
        / cli.out_dir
    )
    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        cli.device == "cuda"
        and torch.cuda.is_available()
    ):
        device = torch.device(
            "cuda"
        )
    else:
        if cli.device == "cuda":
            print(
                "[WARN] CUDA 不可用，自动切换 CPU。"
            )
        device = torch.device(
            "cpu"
        )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "NEWS LAG ROBUSTNESS | H=1"
    )
    print(
        f"Lags             : {lags}"
    )
    print(
        f"Common max lag   : {common_max_lag}"
    )
    print(
        "Price            : t-L+1 ... t"
    )
    print(
        "Target           : t+1"
    )
    print(
        "Common calendar  : ENABLED"
    )
    print(
        "Final model      : No-Cross + Scalar Gate"
    )
    print(
        "=" * 80
    )

    # 保存原类，最终一定恢复
    original_dataset = (
        final_train
        .StockIntervalDataset
    )

    all_rows = []
    audit_rows = []

    try:
        for lag in lags:

            print(
                "\n"
                + "#" * 80
            )
            print(
                f"RUNNING Lag-{lag}"
            )
            print(
                "News window: "
                f"t-L-{lag-1} ... t-{lag}"
            )
            print(
                "#" * 80
            )

            # -----------------------------------------------
            # 用 closure 固定当前 lag。
            # train_one_horizon 调用接口不变：
            # Dataset(prepared, split, lookback, horizon)
            # -----------------------------------------------
            class CurrentLagDataset(
                LaggedStockIntervalDataset
            ):
                def __init__(
                    self,
                    prepared,
                    split,
                    lookback,
                    horizon,
                ):
                    super().__init__(
                        prepared=prepared,
                        split=split,
                        lookback=lookback,
                        horizon=horizon,
                        news_lag=lag,
                        common_max_lag=(
                            common_max_lag
                        ),
                    )

            final_train.StockIntervalDataset = (
                CurrentLagDataset
            )

            train_args = make_train_args(
                cli,
                lag,
            )

            result = (
                final_train
                .train_one_horizon(
                    train_args,
                    horizon=1,
                    device=device,
                )
            )

            result = result.copy()
            result["news_lag"] = lag
            result[
                "news_condition"
            ] = f"Lag-{lag}"

            all_rows.append(
                result
            )

            lag_dir = (
                out_root
                / f"Lag{lag}"
                / "H1"
            )

            patch_summary_json(
                lag_dir
                / "run_summary.json",
                lag=lag,
                common_max_lag=(
                    common_max_lag
                ),
            )

            # 从 summary 读取实际样本数和日期
            summary_path = (
                lag_dir
                / "run_summary.json"
            )

            if summary_path.exists():
                with open(
                    summary_path,
                    "r",
                    encoding="utf-8",
                ) as f:
                    sm = json.load(f)

                audit_rows.append({
                    "news_lag": lag,
                    "news_protocol": (
                        f"t-L-{lag-1} ... t-{lag}"
                    ),
                    "train_samples": (
                        sm.get(
                            "train_samples"
                        )
                    ),
                    "val_samples": (
                        sm.get(
                            "val_samples"
                        )
                    ),
                    "test_samples": (
                        sm.get(
                            "test_samples"
                        )
                    ),
                    "train_end": (
                        sm.get(
                            "train_end_split_boundary"
                        )
                    ),
                    "strict_train_asof_end": (
                        sm.get(
                            "strict_train_asof_end_after_purge"
                        )
                    ),
                    "val_end": (
                        sm.get(
                            "val_end_split_boundary"
                        )
                    ),
                    "pca_dim": (
                        sm.get(
                            "pca_dim"
                        )
                    ),
                    "pca_evr": (
                        sm.get(
                            "pca_evr"
                        )
                    ),
                    "common_max_lag": (
                        common_max_lag
                    ),
                })

    finally:
        # 无论中途是否报错，都恢复原 Dataset
        final_train.StockIntervalDataset = (
            original_dataset
        )

    # ========================================================
    # 5. 汇总结果
    # ========================================================
    table = pd.concat(
        all_rows,
        ignore_index=True,
    )

    # 统一表格顺序
    preferred = [
        "news_condition",
        "news_lag",
        "scope",
        "n",
        "IMAPE_pct",
        "IRMSE",
        "IMAE",
        "center_MAE",
        "radius_MAE",
        "violation_rate",
        "H",
    ]

    keep = [
        c for c in preferred
        if c in table.columns
    ]

    table = (
        table[keep]
        .sort_values(
            [
                "news_lag",
                "scope",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    table_path = (
        out_root
        / "table_news_lag_robustness.csv"
    )

    table.to_csv(
        table_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 相对 Lag-1 的变化百分比
    # 正数 = 误差变大（性能变差）
    # 负数 = 误差降低（性能改善）
    # --------------------------------------------------------
    compare_rows = []

    lag1 = table[
        table["news_lag"] == 1
    ].copy()

    if len(lag1) > 0:
        for lag in lags:
            cur = table[
                table["news_lag"]
                == lag
            ]

            for scope in cur[
                "scope"
            ].unique():
                a = cur[
                    cur["scope"]
                    == scope
                ].iloc[0]

                bdf = lag1[
                    lag1["scope"]
                    == scope
                ]

                if len(bdf) == 0:
                    continue

                b = bdf.iloc[0]

                row = {
                    "news_condition": (
                        f"Lag-{lag}"
                    ),
                    "news_lag": lag,
                    "scope": scope,
                }

                for metric in [
                    "IMAPE_pct",
                    "IRMSE",
                    "IMAE",
                ]:
                    base = float(
                        b[metric]
                    )
                    val = float(
                        a[metric]
                    )

                    row[
                        metric
                    ] = val

                    row[
                        f"{metric}_change_vs_Lag1_pct"
                    ] = (
                        (
                            val - base
                        )
                        / max(
                            abs(base),
                            1e-12,
                        )
                        * 100.0
                    )

                compare_rows.append(
                    row
                )

    compare = pd.DataFrame(
        compare_rows
    )

    compare_path = (
        out_root
        / "table_news_lag_vs_lag1.csv"
    )

    compare.to_csv(
        compare_path,
        index=False,
        encoding="utf-8-sig",
    )

    audit = pd.DataFrame(
        audit_rows
    )

    audit_path = (
        out_root
        / "lag_protocol_audit.csv"
    )

    audit.to_csv(
        audit_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 6. Print
    # ========================================================
    print(
        "\n"
        + "=" * 80
    )
    print(
        "NEWS LAG ROBUSTNESS RESULTS"
    )
    print(
        "=" * 80
    )

    print(
        table.to_string(
            index=False
        )
    )

    if len(compare):
        print(
            "\n"
            + "-" * 80
        )
        print(
            "Change vs Lag-1 "
            "(positive = worse)"
        )
        print(
            "-" * 80
        )
        print(
            compare.to_string(
                index=False
            )
        )

    print(
        "\n[SAVE]"
    )
    print(
        table_path
    )
    print(
        compare_path
    )
    print(
        audit_path
    )

    print(
        "\n解释时重点看："
    )
    print(
        "1) 若 Lag-1 < Lag-2 < Lag-3（误差），"
        "说明新闻信息具有明显时效衰减。"
    )
    print(
        "2) 若 Lag-2/3 与 Lag-1 接近，"
        "说明模型对新闻截止日的小幅变化较稳健。"
    )
    print(
        "3) 若 Lag-2/3 反而明显优于 Lag-1，"
        "不要强行解释为“时效性”，应进一步检查"
        "近期新闻噪声、聚合方式或市场异质性。"
    )


if __name__ == "__main__":
    main()
