# -*- coding: utf-8 -*-
r"""
run_stockv2_tune.py

第一轮单变量调参脚本。

默认目录：
    E:\Code\stockv2

与当前最终模型结构一致，并新增命令行参数：
    --exp_name
    --lr
    --lambda_smooth / --lambda_news_smooth
    --pca_dim
    --lookback / --seq_len
    --hidden_dim
    --dropout

默认值以当前代码为基准：
    lr=5e-4
    lambda_news_smooth=0.005
    pca_dim=16
    lookback=20
    hidden_dim=64
    dropout=0.20

输出互不覆盖：
    E:\Code\stockv2\outputs_tuning\<exp_name>\H1\...

例：
python run_stockv2_tune.py --horizons 1 --device cuda --exp_name A0
python run_stockv2_tune.py --horizons 1 --device cuda --exp_name A1_lr3e4 --lr 3e-4
python run_stockv2_tune.py --horizons 1 --device cuda --exp_name A2_smooth001 --lambda_smooth 0.001
python run_stockv2_tune.py --horizons 1 --device cuda --exp_name A3_pca32 --pca_dim 32
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
)

from stockv2_data_tune import (
    MARKET_TO_ID,
    PRICE_COLS,
    ROOT_DEFAULT,
    STATE_COLS,
    prepare_for_horizon,
)
from stockv2_model_tune import (
    FinalDualStreamIntervalModel,
    FinalIntervalLoss,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class Sample:
    market: str
    as_of_date: pd.Timestamp
    target_dates: List[pd.Timestamp]
    base_close: float
    price_seq: np.ndarray
    news_seq: np.ndarray
    market_state: np.ndarray
    y_low: np.ndarray
    y_high: np.ndarray
    y_center: np.ndarray
    y_radius: np.ndarray


class StockIntervalDataset(Dataset):
    def __init__(
        self,
        prepared,
        split: str,
        lookback: int,
        horizon: int,
    ) -> None:
        if split not in {
            "train",
            "val",
            "test",
        }:
            raise ValueError(split)

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

        for market in ("SSE", "SZI"):
            p = (
                price[
                    price["market"] == market
                ]
                .sort_values("date")
                .reset_index(drop=True)
            )
            n = (
                news[
                    news["market"] == market
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
                prepared.price_scaler
                .transform(
                    p[PRICE_COLS].astype(float)
                )
                .astype(np.float32)
            )
            n_scaled = (
                prepared.news_scaler
                .transform(
                    n[news_cols].astype(float)
                )
                .astype(np.float32)
            )
            f_scaled = (
                prepared.state_scaler
                .transform(
                    p[STATE_COLS].astype(float)
                )
                .astype(np.float32)
            )

            dates = pd.to_datetime(
                p["date"]
            ).tolist()
            date_to_pos = {
                pd.Timestamp(d): i
                for i, d in enumerate(dates)
            }
            train_end_pos = date_to_pos[
                train_end
            ]
            val_end_pos = date_to_pos[
                val_end
            ]

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

            # i=lookback：
            # Price: i-L+1...i
            # News : i-L...i-1
            for i in range(
                lookback,
                len(p) - horizon,
            ):
                as_of = pd.Timestamp(
                    dates[i]
                )

                # --------------------------------------------
                # Purged boundary:
                # train/val/test 的多步 target window 不重叠。
                # H=1 时与当前结果完全相同；
                # H>1 时仅 purge 边界处 H-1 个 as-of 样本。
                # --------------------------------------------
                target_last_pos = i + horizon

                if split == "train":
                    if not (
                        as_of <= train_end
                        and target_last_pos
                        <= train_end_pos + 1
                    ):
                        continue

                elif split == "val":
                    if not (
                        train_end < as_of <= val_end
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
                t1 = i + 1 + horizon

                self.samples.append(
                    Sample(
                        market=market,
                        as_of_date=as_of,
                        target_dates=[
                            pd.Timestamp(d)
                            for d in dates[t0:t1]
                        ],
                        base_close=float(
                            close[i]
                        ),
                        price_seq=p_scaled[
                            i - lookback + 1:
                            i + 1
                        ],
                        news_seq=n_scaled[
                            i - lookback:i
                        ],
                        market_state=f_scaled[i],
                        y_low=y_low[
                            t0:t1
                        ].astype(np.float32),
                        y_high=y_high[
                            t0:t1
                        ].astype(np.float32),
                        y_center=y_center[
                            t0:t1
                        ].astype(np.float32),
                        y_radius=y_radius[
                            t0:t1
                        ].astype(np.float32),
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "price_seq": torch.tensor(
                s.price_seq,
                dtype=torch.float32,
            ),
            "news_seq": torch.tensor(
                s.news_seq,
                dtype=torch.float32,
            ),
            "market_state": torch.tensor(
                s.market_state,
                dtype=torch.float32,
            ),
            "market_id": torch.tensor(
                MARKET_TO_ID[s.market],
                dtype=torch.long,
            ),
            "base_close": torch.tensor(
                s.base_close,
                dtype=torch.float32,
            ),
            "y_low": torch.tensor(
                s.y_low,
                dtype=torch.float32,
            ),
            "y_high": torch.tensor(
                s.y_high,
                dtype=torch.float32,
            ),
            "y_center": torch.tensor(
                s.y_center,
                dtype=torch.float32,
            ),
            "y_radius": torch.tensor(
                s.y_radius,
                dtype=torch.float32,
            ),
            "market_name": s.market,
            "as_of_date": str(
                s.as_of_date.date()
            ),
            "target_dates": [
                str(x.date())
                for x in s.target_dates
            ],
        }


def collate_fn(batch):
    tensor_keys = [
        "price_seq",
        "news_seq",
        "market_state",
        "market_id",
        "base_close",
        "y_low",
        "y_high",
        "y_center",
        "y_radius",
    ]

    out = {
        k: torch.stack(
            [b[k] for b in batch],
            dim=0,
        )
        for k in tensor_keys
    }
    out["market_name"] = [
        b["market_name"] for b in batch
    ]
    out["as_of_date"] = [
        b["as_of_date"] for b in batch
    ]
    out["target_dates"] = [
        b["target_dates"] for b in batch
    ]
    return out


def move_batch(batch, device):
    for k in [
        "price_seq",
        "news_seq",
        "market_state",
        "market_id",
        "base_close",
        "y_low",
        "y_high",
        "y_center",
        "y_radius",
    ]:
        batch[k] = batch[k].to(device)
    return batch


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    rows = []
    losses = []
    gates = []
    attn_all = []

    for batch in loader:
        batch = move_batch(
            batch, device
        )
        pred = model(
            batch["price_seq"],
            batch["news_seq"],
            batch["market_state"],
            batch["market_id"],
            batch["base_close"],
        )

        loss, parts = criterion(
            batch, pred
        )
        losses.append(
            parts["total"]
        )

        gate = (
            pred["gate"]
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        attn = (
            pred["attention_weights"]
            .detach()
            .cpu()
            .numpy()
        )

        gates.extend(
            gate.tolist()
        )
        attn_all.append(attn)

        arrays = {
            "pred_low": (
                pred["low"]
                .detach()
                .cpu()
                .numpy()
            ),
            "pred_high": (
                pred["high"]
                .detach()
                .cpu()
                .numpy()
            ),
            "pred_center": (
                pred["center"]
                .detach()
                .cpu()
                .numpy()
            ),
            "pred_radius": (
                pred["radius"]
                .detach()
                .cpu()
                .numpy()
            ),
            "true_low": (
                batch["y_low"]
                .detach()
                .cpu()
                .numpy()
            ),
            "true_high": (
                batch["y_high"]
                .detach()
                .cpu()
                .numpy()
            ),
            "true_center": (
                batch["y_center"]
                .detach()
                .cpu()
                .numpy()
            ),
            "true_radius": (
                batch["y_radius"]
                .detach()
                .cpu()
                .numpy()
            ),
        }

        B, H = arrays[
            "pred_low"
        ].shape

        for i in range(B):
            for h in range(H):
                rows.append(
                    {
                        "market": (
                            batch[
                                "market_name"
                            ][i]
                        ),
                        "as_of_date": (
                            batch[
                                "as_of_date"
                            ][i]
                        ),
                        "target_date": (
                            batch[
                                "target_dates"
                            ][i][h]
                        ),
                        "horizon_step": h + 1,
                        "gate_news_increment": float(
                            gate[i]
                        ),
                        "true_low": float(
                            arrays[
                                "true_low"
                            ][i, h]
                        ),
                        "pred_low": float(
                            arrays[
                                "pred_low"
                            ][i, h]
                        ),
                        "true_high": float(
                            arrays[
                                "true_high"
                            ][i, h]
                        ),
                        "pred_high": float(
                            arrays[
                                "pred_high"
                            ][i, h]
                        ),
                        "true_center": float(
                            arrays[
                                "true_center"
                            ][i, h]
                        ),
                        "pred_center": float(
                            arrays[
                                "pred_center"
                            ][i, h]
                        ),
                        "true_radius": float(
                            arrays[
                                "true_radius"
                            ][i, h]
                        ),
                        "pred_radius": float(
                            arrays[
                                "pred_radius"
                            ][i, h]
                        ),
                    }
                )

    pred_df = pd.DataFrame(rows)

    attn_mat = (
        np.concatenate(
            attn_all, axis=0
        )
        if attn_all
        else np.empty((0, 0))
    )

    diag = {
        "loss": (
            float(np.mean(losses))
            if losses
            else np.nan
        ),
        "gate_mean": (
            float(np.mean(gates))
            if gates
            else np.nan
        ),
        "gate_std": (
            float(np.std(gates))
            if gates
            else np.nan
        ),
        "gate_min": (
            float(np.min(gates))
            if gates
            else np.nan
        ),
        "gate_max": (
            float(np.max(gates))
            if gates
            else np.nan
        ),
    }

    if attn_mat.size:
        lag_profile = pd.DataFrame(
            {
                "news_position": np.arange(
                    attn_mat.shape[1]
                ),
                "relative_day": np.arange(
                    -attn_mat.shape[1],
                    0,
                ),
                "mean_attention": (
                    attn_mat.mean(axis=0)
                ),
                "std_attention": (
                    attn_mat.std(axis=0)
                ),
            }
        )
    else:
        lag_profile = pd.DataFrame()

    return (
        pred_df,
        diag,
        lag_profile,
    )


def metric_record(name, g):
    low_err = (
        g["pred_low"].to_numpy()
        - g["true_low"].to_numpy()
    )
    high_err = (
        g["pred_high"].to_numpy()
        - g["true_high"].to_numpy()
    )

    eps = 1e-8
    low_den = np.maximum(
        np.abs(
            g["true_low"].to_numpy()
        ),
        eps,
    )
    high_den = np.maximum(
        np.abs(
            g["true_high"].to_numpy()
        ),
        eps,
    )

    return {
        "scope": name,
        "n": int(len(g)),
        "IMAE": float(
            np.mean(
                (
                    np.abs(low_err)
                    + np.abs(high_err)
                )
                / 2.0
            )
        ),
        "IRMSE": float(
            np.sqrt(
                np.mean(
                    (
                        low_err ** 2
                        + high_err ** 2
                    )
                    / 2.0
                )
            )
        ),
        "IMAPE_pct": float(
            100.0
            * np.mean(
                (
                    np.abs(low_err)
                    / low_den
                    + np.abs(high_err)
                    / high_den
                )
                / 2.0
            )
        ),
        "center_MAE": float(
            np.mean(
                np.abs(
                    g[
                        "pred_center"
                    ].to_numpy()
                    - g[
                        "true_center"
                    ].to_numpy()
                )
            )
        ),
        "radius_MAE": float(
            np.mean(
                np.abs(
                    g[
                        "pred_radius"
                    ].to_numpy()
                    - g[
                        "true_radius"
                    ].to_numpy()
                )
            )
        ),
        "violation_rate": float(
            np.mean(
                g[
                    "pred_high"
                ].to_numpy()
                < g[
                    "pred_low"
                ].to_numpy()
            )
        ),
    }


def compute_metrics(pred_df):
    overall = []

    for market, g in pred_df.groupby(
        "market"
    ):
        overall.append(
            metric_record(
                market, g
            )
        )

    overall.append(
        metric_record(
            "ALL", pred_df
        )
    )

    by_step = []

    for (
        market,
        h,
    ), g in pred_df.groupby(
        [
            "market",
            "horizon_step",
        ]
    ):
        r = metric_record(
            market, g
        )
        r["horizon_step"] = int(h)
        by_step.append(r)

    for h, g in pred_df.groupby(
        "horizon_step"
    ):
        r = metric_record(
            "ALL", g
        )
        r["horizon_step"] = int(h)
        by_step.append(r)

    return (
        pd.DataFrame(overall),
        pd.DataFrame(by_step),
    )


def print_config(
    args,
    horizon,
    prepared=None,
):
    print("=" * 72)
    print(
        f"Experiment       : {args.exp_name}"
    )
    print(
        f"Horizon          : {horizon}"
    )
    print(
        f"Learning rate    : {args.lr}"
    )
    print(
        f"Weight decay     : {args.weight_decay}"
    )
    print(
        f"Lambda smooth    : {args.lambda_news_smooth}"
    )
    print(
        f"Lambda C/R       : {args.lambda_center_radius}"
    )
    print(
        f"PCA dim          : {args.pca_dim}"
    )
    print(
        f"Lookback         : {args.lookback}"
    )
    print(
        f"Hidden dim       : {args.hidden_dim}"
    )
    print(
        f"Dropout          : {args.dropout}"
    )
    print(
        f"Attention heads  : {args.attn_heads}"
    )
    print(
        f"Batch size       : {args.batch_size}"
    )
    print(
        f"Seed             : {args.seed}"
    )
    if prepared is not None:
        print(
            f"News input dim   : {len(prepared.news_cols)}"
        )
        print(
            f"PCA EVR          : "
            f"{prepared.pca_explained_variance_ratio_sum:.4f}"
        )
    print("=" * 72)


def train_horizon(
    args,
    horizon,
    device,
):
    # 单独跑 H=1 与批量跑 H=1,3,5 时，初始化保持一致
    set_seed(args.seed)

    exp_root = (
        Path(args.root)
        / args.out_dir
        / args.exp_name
    )
    out_dir = (
        exp_root / f"H{horizon}"
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepared = prepare_for_horizon(
        Path(args.root),
        horizon=horizon,
        lookback=args.lookback,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        pca_dim=args.pca_dim,
        save_dir=out_dir / "prepared",
    )

    train_ds = StockIntervalDataset(
        prepared,
        "train",
        args.lookback,
        horizon,
    )
    val_ds = StockIntervalDataset(
        prepared,
        "val",
        args.lookback,
        horizon,
    )
    test_ds = StockIntervalDataset(
        prepared,
        "test",
        args.lookback,
        horizon,
    )

    print_config(
        args,
        horizon,
        prepared,
    )

    print(
        f"[H={horizon}] "
        f"train_end={prepared.train_end.date()}, "
        f"strict_train_end="
        f"{prepared.strict_train_asof_end.date()}, "
        f"val_end={prepared.val_end.date()}, "
        f"samples train/val/test="
        f"{len(train_ds)}/"
        f"{len(val_ds)}/"
        f"{len(test_ds)}, "
        f"PCA{args.pca_dim} "
        f"EVR="
        f"{prepared.pca_explained_variance_ratio_sum:.4f}"
    )

    if min(
        len(train_ds),
        len(val_ds),
        len(test_ds),
    ) == 0:
        raise RuntimeError(
            "某个 split 样本数为0，请检查切分参数。"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = FinalDualStreamIntervalModel(
        price_dim=len(PRICE_COLS),
        news_dim=len(
            prepared.news_cols
        ),
        state_dim=len(STATE_COLS),
        horizon=horizon,
        hidden_dim=args.hidden_dim,
        market_emb_dim=(
            args.market_emb_dim
        ),
        num_layers=args.num_layers,
        dropout=args.dropout,
        attn_heads=args.attn_heads,
    ).to(device)

    criterion = FinalIntervalLoss(
        lambda_boundary=(
            args.lambda_boundary
        ),
        lambda_center_radius=(
            args.lambda_center_radius
        ),
        lambda_news_smooth=(
            args.lambda_news_smooth
        ),
        huber_delta=args.huber_delta,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=8,
        )
    )

    best_val = float("inf")
    best_epoch = 0
    wait = 0
    log_rows = []

    best_path = (
        out_dir / "best_model.pt"
    )

    for epoch in range(
        1, args.epochs + 1
    ):
        model.train()
        train_losses = []

        for batch in train_loader:
            batch = move_batch(
                batch, device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            pred = model(
                batch["price_seq"],
                batch["news_seq"],
                batch["market_state"],
                batch["market_id"],
                batch["base_close"],
            )

            loss, _ = criterion(
                batch, pred
            )
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )

            optimizer.step()

            train_losses.append(
                float(
                    loss.detach().cpu()
                )
            )

        _, val_diag, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        val_loss = val_diag["loss"]
        train_loss = float(
            np.mean(train_losses)
        )

        scheduler.step(val_loss)

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
                "val_gate_mean": (
                    val_diag[
                        "gate_mean"
                    ]
                ),
                "val_gate_std": (
                    val_diag[
                        "gate_std"
                    ]
                ),
            }
        )

        if (
            epoch == 1
            or epoch
            % args.print_every
            == 0
        ):
            print(
                f"H={horizon} "
                f"epoch={epoch:03d} "
                f"train={train_loss:.6f} "
                f"val={val_loss:.6f} "
                f"lr={current_lr:.2e} "
                f"gate="
                f"{val_diag['gate_mean']:.3f}"
                f"±"
                f"{val_diag['gate_std']:.3f}"
            )

        if (
            val_loss
            < best_val - 1e-10
        ):
            best_val = val_loss
            best_epoch = epoch
            wait = 0

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "horizon": horizon,
                    "args": vars(args),
                    "train_end": str(
                        prepared
                        .train_end
                        .date()
                    ),
                    "strict_train_asof_end": str(
                        prepared
                        .strict_train_asof_end
                        .date()
                    ),
                    "val_end": str(
                        prepared
                        .val_end
                        .date()
                    ),
                    "price_cols": PRICE_COLS,
                    "news_cols": (
                        prepared.news_cols
                    ),
                    "state_cols": STATE_COLS,
                    "pca_dim": (
                        args.pca_dim
                    ),
                    "pca_explained_variance_ratio_sum": (
                        prepared
                        .pca_explained_variance_ratio_sum
                    ),
                },
                best_path,
            )
        else:
            wait += 1

            if wait >= args.patience:
                print(
                    f"[H={horizon}] "
                    f"Early stopping: "
                    f"epoch={epoch}, "
                    f"best_epoch="
                    f"{best_epoch}, "
                    f"best_val="
                    f"{best_val:.6f}"
                )
                break

    pd.DataFrame(
        log_rows
    ).to_csv(
        out_dir / "train_log.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 载入最佳验证 checkpoint
    try:
        ckpt = torch.load(
            best_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        ckpt = torch.load(
            best_path,
            map_location=device,
        )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    val_pred, val_diag, val_attn = (
        evaluate(
            model,
            val_loader,
            criterion,
            device,
        )
    )
    test_pred, test_diag, test_attn = (
        evaluate(
            model,
            test_loader,
            criterion,
            device,
        )
    )

    (
        val_overall,
        val_by_step,
    ) = compute_metrics(val_pred)

    (
        test_overall,
        test_by_step,
    ) = compute_metrics(test_pred)

    val_pred.to_csv(
        out_dir
        / "val_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_pred.to_csv(
        out_dir
        / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    val_overall.to_csv(
        out_dir
        / "val_metrics_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_overall.to_csv(
        out_dir
        / "test_metrics_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    val_by_step.to_csv(
        out_dir
        / "val_metrics_by_step.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_by_step.to_csv(
        out_dir
        / "test_metrics_by_step.csv",
        index=False,
        encoding="utf-8-sig",
    )

    val_attn.to_csv(
        out_dir
        / "val_attention_lag_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_attn.to_csv(
        out_dir
        / "test_attention_lag_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "experiment": args.exp_name,
        "horizon": horizon,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "train_samples": len(
            train_ds
        ),
        "val_samples": len(
            val_ds
        ),
        "test_samples": len(
            test_ds
        ),
        "train_end_split_boundary": str(
            prepared.train_end.date()
        ),
        "strict_train_asof_end_after_purge": str(
            prepared
            .strict_train_asof_end
            .date()
        ),
        "val_end_split_boundary": str(
            prepared.val_end.date()
        ),
        "train_news_end_lag1": str(
            prepared.train_news_end.date()
        ),
        "pca_dim": args.pca_dim,
        "pca_explained_variance_ratio_sum": (
            prepared
            .pca_explained_variance_ratio_sum
        ),
        "news_input_dim": len(
            prepared.news_cols
        ),
        "hyperparameters": vars(args),
        "val_gate": val_diag,
        "test_gate": test_diag,
        "feature_protocol": {
            "price_cols": PRICE_COLS,
            "news_cols": (
                prepared.news_cols
            ),
            "market_state_cols": (
                STATE_COLS
            ),
            "news_fill_rule": (
                "previous-trading-day-last-news"
            ),
            "time_alignment": {
                "price": "t-L+1 ... t",
                "news": "t-L ... t-1",
                "target": "t+1 ... t+H",
            },
            "leakage_controls": {
                "no_bfill": True,
                "pca_train_only": True,
                "scalers_train_only": True,
                "purge_target_overlap": True,
            },
        },
    }

    with open(
        out_dir / "run_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        out_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            vars(args),
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\n[H={horizon}] "
        f"BEST VAL = {best_val:.6f} "
        f"@ epoch {best_epoch}"
    )
    print(
        f"[H={horizon}] "
        f"Validation metrics:"
    )
    print(
        val_overall.to_string(
            index=False
        )
    )
    print(
        f"\n[H={horizon}] "
        f"Test metrics:"
    )
    print(
        test_overall.to_string(
            index=False
        )
    )

    return test_overall.assign(
        H=horizon
    )


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--root",
        type=str,
        default=str(ROOT_DEFAULT),
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="outputs_tuning",
    )
    p.add_argument(
        "--exp_name",
        type=str,
        default="A0",
    )

    # --seq_len 与 --lookback 等价
    p.add_argument(
        "--lookback",
        "--seq_len",
        dest="lookback",
        type=int,
        default=20,
    )

    p.add_argument(
        "--pca_dim",
        type=int,
        default=16,
    )
    p.add_argument(
        "--horizons",
        type=str,
        default="1",
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
        "--num_layers",
        type=int,
        default=1,
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.20,
    )
    p.add_argument(
        "--attn_heads",
        type=int,
        default=4,
    )

    p.add_argument(
        "--lambda_boundary",
        type=float,
        default=1.0,
    )
    p.add_argument(
        "--lambda_center_radius",
        type=float,
        default=0.5,
    )

    # 两个命令名等价，方便与你之前的命令保持兼容
    p.add_argument(
        "--lambda_news_smooth",
        "--lambda_smooth",
        dest="lambda_news_smooth",
        type=float,
        default=0.005,
    )

    p.add_argument(
        "--huber_delta",
        type=float,
        default=0.01,
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

    # 注意：你当前 run_stockv2.py 的默认值已经是 5e-4
    p.add_argument(
        "--lr",
        type=float,
        default=5e-4,
    )
    p.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )
    p.add_argument(
        "--patience",
        type=int,
        default=25,
    )
    p.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
    )
    p.add_argument(
        "--print_every",
        type=int,
        default=10,
    )

    return p.parse_args()


def main():
    args = parse_args()

    if (
        args.device == "cuda"
        and not torch.cuda.is_available()
    ):
        print(
            "[WARN] CUDA 不可用，自动切换 CPU。"
        )
        device = torch.device("cpu")
    else:
        device = torch.device(
            args.device
        )

    horizons = [
        int(x.strip())
        for x in args.horizons.split(",")
        if x.strip()
    ]

    if not horizons:
        raise ValueError(
            "--horizons 不能为空。"
        )

    all_frames = []

    for h in horizons:
        all_frames.append(
            train_horizon(
                args,
                h,
                device,
            )
        )

    if all_frames:
        final = pd.concat(
            all_frames,
            ignore_index=True,
        )
        cols = [
            "H"
        ] + [
            c
            for c in final.columns
            if c != "H"
        ]
        final = final[cols]

        exp_root = (
            Path(args.root)
            / args.out_dir
            / args.exp_name
        )
        exp_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        final.to_csv(
            exp_root
            / "all_test_metrics_overall.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "\n========== "
            "Final overall test metrics "
            "=========="
        )
        print(
            final.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
