# -*- coding: utf-8 -*-
"""
run_robustness_window_length.py

对应原稿 4.5（1）不同历史窗口长度分析 / 表3。

核心原则：
    - H=1（默认）
    - L = 5,10,20,30
    - 最终模型结构固定
    - 为避免“窗口长度变化同时改变 train/val/test 日期边界”的混杂，
      所有 L 使用主模型 L=20 的同一 train_end / val_end。
    - PCA 和 scaler 仍严格只用训练可见期拟合。
    - L=20 可复用当前最终模型正式结果。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import run_ablation_no_cross_attention as final_train

from stockv2_data_tune import (
    PRICE_COLS,
    STATE_COLS,
    HorizonPrepared,
    load_price_data,
    load_news_raw,
    preserve_or_fill_trading_day_news,
    aggregate_news_daily,
    get_news_cols,
    prepare_for_horizon,
)


def prepare_fixed_split(
    root: Path,
    horizon: int,
    lookback: int,
    train_ratio: float,
    val_ratio: float,
    pca_dim: int,
    save_dir: Optional[Path] = None,
):
    """
    固定使用主模型 L=20 的 train/val 日期边界，
    仅改变输入历史长度。
    """
    baseline = prepare_for_horizon(
        root,
        horizon=horizon,
        lookback=20,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        pca_dim=pca_dim,
        save_dir=None,
    )

    train_end = pd.Timestamp(
        baseline.train_end
    )
    val_end = pd.Timestamp(
        baseline.val_end
    )

    price = load_price_data(root)
    trading_dates = sorted(
        price.loc[
            price["market"] == "SSE",
            "date",
        ].unique()
    )
    trading_dates = [
        pd.Timestamp(x)
        for x in trading_dates
    ]

    train_pos = trading_dates.index(
        train_end
    )
    strict_pos = (
        train_pos - (horizon - 1)
    )
    strict_train_asof_end = (
        trading_dates[strict_pos]
    )
    train_news_end = (
        trading_dates[strict_pos - 1]
    )

    news_raw = load_news_raw(root)
    news_raw = preserve_or_fill_trading_day_news(
        news_raw,
        trading_dates,
    )

    news_dates = pd.to_datetime(
        news_raw.items["date"]
    )
    train_mask = (
        news_dates <= train_news_end
    ).to_numpy()

    pca = PCA(
        n_components=pca_dim,
        random_state=42,
    )
    pca.fit(
        news_raw.embeddings[
            train_mask
        ]
    )

    reduced = pca.transform(
        news_raw.embeddings
    ).astype(np.float32)

    news_daily = aggregate_news_daily(
        news_raw.items,
        reduced,
        trading_dates,
    )

    news_cols = get_news_cols(
        pca_dim
    )

    price_train = price[
        price["date"]
        <= strict_train_asof_end
    ]
    news_train = news_daily[
        news_daily["date"]
        <= train_news_end
    ]

    price_scaler = StandardScaler().fit(
        price_train[
            PRICE_COLS
        ].astype(float)
    )
    news_scaler = StandardScaler().fit(
        news_train[
            news_cols
        ].astype(float)
    )
    state_scaler = StandardScaler().fit(
        price_train[
            STATE_COLS
        ].astype(float)
    )

    prepared = HorizonPrepared(
        horizon=horizon,
        pca_dim=pca_dim,
        train_end=train_end,
        val_end=val_end,
        strict_train_asof_end=(
            strict_train_asof_end
        ),
        train_news_end=train_news_end,
        price_panel=price,
        news_daily=news_daily,
        news_cols=news_cols,
        pca=pca,
        price_scaler=price_scaler,
        news_scaler=news_scaler,
        state_scaler=state_scaler,
        pca_explained_variance_ratio_sum=float(
            pca
            .explained_variance_ratio_
            .sum()
        ),
    )

    if save_dir is not None:
        save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        news_daily.to_csv(
            save_dir / "news_daily.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return prepared


def build_args(
    cli,
    lookback: int,
):
    return SimpleNamespace(
        root=cli.root,
        out_dir=(
            f"{cli.out_dir}/L{lookback}"
        ),
        horizons=str(cli.horizon),
        lookback=lookback,
        pca_dim=cli.pca_dim,
        train_ratio=cli.train_ratio,
        val_ratio=cli.val_ratio,
        hidden_dim=cli.hidden_dim,
        market_emb_dim=cli.market_emb_dim,
        num_layers=1,
        dropout=cli.dropout,
        attn_heads=4,
        lambda_boundary=1.0,
        lambda_center_radius=0.5,
        lambda_news_smooth=0.005,
        huber_delta=0.01,
        epochs=cli.epochs,
        batch_size=cli.batch_size,
        lr=cli.lr,
        weight_decay=1e-4,
        patience=cli.patience,
        grad_clip=1.0,
        seed=cli.seed,
        device=cli.device,
        print_every=10,
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--root",
        default=r"E:\Code\stockv2",
    )
    p.add_argument(
        "--out_dir",
        default="outputs_robustness_window_length",
    )
    p.add_argument(
        "--windows",
        default="5,10,20,30",
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=1,
    )
    p.add_argument(
        "--pca_dim",
        type=int,
        default=16,
    )
    p.add_argument(
        "--train_ratio",
        type=float,
        default=0.70,
    )
    p.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
    )
    p.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
    )
    p.add_argument(
        "--market_emb_dim",
        type=int,
        default=8,
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.20,
    )
    p.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=200,
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    p.add_argument(
        "--patience",
        type=int,
        default=25,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
    )
    p.add_argument(
        "--reuse_L20",
        dest="reuse_L20",
        action="store_true",
        help="L=20 时复用正式最终模型结果（默认开启）。",
    )
    p.add_argument(
        "--no_reuse_L20",
        dest="reuse_L20",
        action="store_false",
        help="L=20 也重新训练。",
    )
    p.set_defaults(reuse_L20=True)

    cli = p.parse_args()

    device = (
        torch.device("cuda")
        if (
            cli.device == "cuda"
            and torch.cuda.is_available()
        )
        else torch.device("cpu")
    )

    windows = [
        int(x.strip())
        for x in cli.windows.split(",")
        if x.strip()
    ]

    root = Path(cli.root)
    out_root = root / cli.out_dir
    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 临时替换训练脚本的数据准备函数，
    # 使不同 L 的日期边界保持固定。
    original_prepare = (
        final_train.prepare_for_horizon
    )

    rows = []

    try:
        final_train.prepare_for_horizon = (
            prepare_fixed_split
        )

        for L in windows:
            print(
                "\n"
                + "#" * 80
                + f"\nHistorical Window Robustness: L={L}\n"
                + "#" * 80
            )

            if (
                L == 20
                and cli.reuse_L20
            ):
                existing = (
                    root
                    / "outputs_ablation_no_cross_attention"
                    / f"H{cli.horizon}"
                    / "test_metrics_overall.csv"
                )

                if existing.exists():
                    m = pd.read_csv(
                        existing,
                        encoding="utf-8-sig",
                    )
                    m["lookback"] = L
                    m["source"] = (
                        "reused_final_model"
                    )
                    rows.append(m)
                    print(
                        f"[REUSE] {existing}"
                    )
                    continue

            args = build_args(
                cli,
                L,
            )

            result = (
                final_train.train_one_horizon(
                    args,
                    cli.horizon,
                    device,
                )
            )

            result["lookback"] = L
            result["source"] = (
                "retrained_fixed_split"
            )
            rows.append(result)

    finally:
        final_train.prepare_for_horizon = (
            original_prepare
        )

    final = pd.concat(
        rows,
        ignore_index=True,
    )

    keep = [
        "lookback",
        "scope",
        "n",
        "IMAPE_pct",
        "IRMSE",
        "IMAE",
        "center_MAE",
        "radius_MAE",
        "violation_rate",
        "source",
    ]
    keep = [
        c for c in keep
        if c in final.columns
    ]

    final = final[keep].sort_values(
        ["lookback", "scope"]
    )

    out = (
        out_root
        / "table3_window_length_robustness.csv"
    )
    final.to_csv(
        out,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n========== Table 3 =========="
    )
    print(
        final.to_string(index=False)
    )
    print(
        f"\n[SAVE] {out}"
    )


if __name__ == "__main__":
    main()
