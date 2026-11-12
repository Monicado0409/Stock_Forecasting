# -*- coding: utf-8 -*-
"""
run_ablation_price_only.py

Ablation-1: Price-only

目的：
    检验当前完整模型中，新闻信息是否提供了真实的增量预测价值。

严格保持与 A1 Full Model 一致：
    - 相同价格数据
    - 相同 train / val / test 划分逻辑
    - 相同 H>1 purge 逻辑
    - 相同 lookback=20
    - 相同 price scaler（train-only）
    - 相同目标定义
    - 相同 IMAE / IRMSE / IMAPE 计算方式
    - 相同 hidden_dim=64
    - 相同 market_emb_dim=8
    - 相同 dropout=0.2
    - 相同 lr=3e-4
    - 相同 batch_size=64
    - 相同 seed=42
    - 相同 boundary + center/radius loss

唯一结构变化：
    Full:
        Price GRU
        + News GRU
        + Cross-Attention
        + Gate
        + residual news correction

    Price-only:
        Price GRU
        -> output head

注意：
    数据准备仍调用 stockv2_data_tune.py，
    因而会构建新闻/PCA，但 Price-only 模型不会读取 news_seq。
    这样做的目的是确保样本集合、日期边界和 Full Model 完全一致。

运行：
    python run_ablation_price_only.py --horizons 1,3,5 --device cuda

输出：
    E:\\Code\\stockv2\\outputs_ablation_price_only\\
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from stockv2_data_tune import (
        ROOT_DEFAULT,
        PRICE_COLS,
        prepare_for_horizon,
    )
except ImportError as exc:
    raise ImportError(
        "未找到 stockv2_data_tune.py。请将本文件放到 E:\\Code\\stockv2\\ 下。"
    ) from exc

try:
    from run_stockv2_tune import (
        StockIntervalDataset,
        collate_fn,
        move_batch,
        compute_metrics,
    )
except ImportError as exc:
    raise ImportError(
        "未找到 run_stockv2_tune.py。请确保它与本文件位于同一目录。"
    ) from exc


# ============================================================
# 1. Seed
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# 2. Price-only model
# ============================================================

class PriceOnlyIntervalModel(nn.Module):
    """
    仅保留完整模型中的 Price GRU 主干。

    h_P = GRU_P(X_P)
    h_F = h_P

    再与 market embedding 拼接后预测：
        center_rel
        radius_rel >= 0
    """

    def __init__(
        self,
        price_dim: int = 6,
        horizon: int = 1,
        hidden_dim: int = 64,
        market_emb_dim: int = 8,
        num_layers: int = 1,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        self.horizon = horizon

        gru_dropout = (
            dropout if num_layers > 1 else 0.0
        )

        self.price_encoder = nn.GRU(
            input_size=price_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )

        self.market_embedding = nn.Embedding(
            2,
            market_emb_dim,
        )

        self.fusion_norm = nn.LayerNorm(
            hidden_dim
        )
        self.output_dropout = nn.Dropout(
            dropout
        )

        self.center_head = nn.Linear(
            hidden_dim + market_emb_dim,
            horizon,
        )
        self.radius_head = nn.Linear(
            hidden_dim + market_emb_dim,
            horizon,
        )

        nn.init.zeros_(
            self.center_head.bias
        )
        nn.init.constant_(
            self.radius_head.bias,
            -4.0,
        )

    def forward(
        self,
        price_seq: torch.Tensor,
        market_id: torch.Tensor,
        base_close: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        price_hidden, _ = self.price_encoder(
            price_seq
        )
        price_state = price_hidden[:, -1, :]

        # 对应 Full Model 中 h_F，
        # 这里只是不再加入新闻修正项
        fused = self.fusion_norm(
            price_state
        )

        market_emb = self.market_embedding(
            market_id
        )

        out_input = self.output_dropout(
            torch.cat(
                [fused, market_emb],
                dim=-1,
            )
        )

        center_rel = self.center_head(
            out_input
        )

        radius_rel = F.softplus(
            self.radius_head(
                out_input
            )
        )

        base = (
            base_close.unsqueeze(-1)
            .clamp_min(1e-8)
        )

        center = base * (
            1.0 + center_rel
        )
        radius = base * radius_rel

        low = center - radius
        high = center + radius

        return {
            "center_rel": center_rel,
            "radius_rel": radius_rel,
            "center": center,
            "radius": radius,
            "low": low,
            "high": high,
            "price_hidden_seq": price_hidden,
            "price_state": price_state,
            "fused": fused,
        }


# ============================================================
# 3. 与 Full Model 相同的区间损失
#    Price-only 没有 news smooth 项
# ============================================================

class PriceOnlyIntervalLoss(nn.Module):
    def __init__(
        self,
        lambda_boundary: float = 1.0,
        lambda_center_radius: float = 0.5,
        huber_delta: float = 0.01,
    ) -> None:
        super().__init__()

        self.lambda_boundary = lambda_boundary
        self.lambda_center_radius = (
            lambda_center_radius
        )
        self.huber_delta = huber_delta

    def _huber_zero(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return F.smooth_l1_loss(
            x,
            torch.zeros_like(x),
            beta=self.huber_delta,
            reduction="mean",
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        pred: Dict[str, torch.Tensor],
    ):
        base = (
            batch["base_close"]
            .unsqueeze(-1)
            .clamp_min(1e-8)
        )

        low_err_rel = (
            pred["low"] - batch["y_low"]
        ) / base

        high_err_rel = (
            pred["high"] - batch["y_high"]
        ) / base

        center_err_rel = (
            pred["center"] - batch["y_center"]
        ) / base

        radius_err_rel = (
            pred["radius"] - batch["y_radius"]
        ) / base

        boundary_loss = 0.5 * (
            self._huber_zero(low_err_rel)
            + self._huber_zero(high_err_rel)
        )

        center_radius_loss = (
            self._huber_zero(center_err_rel)
            + self._huber_zero(radius_err_rel)
        )

        total = (
            self.lambda_boundary
            * boundary_loss
            + self.lambda_center_radius
            * center_radius_loss
        )

        parts = {
            "total": float(
                total.detach().cpu()
            ),
            "boundary": float(
                boundary_loss.detach().cpu()
            ),
            "center_radius": float(
                center_radius_loss.detach().cpu()
            ),
        }

        return total, parts


# ============================================================
# 4. Evaluation
# ============================================================

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

    for batch in loader:
        batch = move_batch(
            batch,
            device,
        )

        pred = model(
            batch["price_seq"],
            batch["market_id"],
            batch["base_close"],
        )

        _, parts = criterion(
            batch,
            pred,
        )
        losses.append(
            parts["total"]
        )

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

        B, H = arrays["pred_low"].shape

        for i in range(B):
            for h in range(H):
                rows.append({
                    "market": (
                        batch["market_name"][i]
                    ),
                    "as_of_date": (
                        batch["as_of_date"][i]
                    ),
                    "target_date": (
                        batch["target_dates"][i][h]
                    ),
                    "horizon_step": h + 1,
                    "true_low": float(
                        arrays["true_low"][i, h]
                    ),
                    "pred_low": float(
                        arrays["pred_low"][i, h]
                    ),
                    "true_high": float(
                        arrays["true_high"][i, h]
                    ),
                    "pred_high": float(
                        arrays["pred_high"][i, h]
                    ),
                    "true_center": float(
                        arrays["true_center"][i, h]
                    ),
                    "pred_center": float(
                        arrays["pred_center"][i, h]
                    ),
                    "true_radius": float(
                        arrays["true_radius"][i, h]
                    ),
                    "pred_radius": float(
                        arrays["pred_radius"][i, h]
                    ),
                })

    pred_df = pd.DataFrame(rows)

    diag = {
        "loss": (
            float(np.mean(losses))
            if losses
            else np.nan
        )
    }

    return pred_df, diag


# ============================================================
# 5. Train one horizon
# ============================================================

def train_one_horizon(
    args,
    horizon,
    device,
):
    set_seed(args.seed)

    out_dir = (
        Path(args.root)
        / args.out_dir
        / f"H{horizon}"
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 与 A1 完全相同的数据准备协议
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

    print("\n" + "=" * 72)
    print(
        f"ABLATION: Price-only | H={horizon}"
    )
    print(
        f"device            : {device}"
    )
    print(
        f"train_end         : {prepared.train_end.date()}"
    )
    print(
        f"strict_train_end  : "
        f"{prepared.strict_train_asof_end.date()}"
    )
    print(
        f"val_end           : {prepared.val_end.date()}"
    )
    print(
        f"samples           : "
        f"{len(train_ds)}/"
        f"{len(val_ds)}/"
        f"{len(test_ds)}"
    )
    print(
        f"price_dim         : {len(PRICE_COLS)}"
    )
    print(
        f"Full comparison   : "
        f"news_dim={len(prepared.news_cols)} "
        f"(prepared but NOT used)"
    )
    print(
        f"lookback          : {args.lookback}"
    )
    print(
        f"hidden_dim        : {args.hidden_dim}"
    )
    print(
        f"dropout           : {args.dropout}"
    )
    print(
        f"lr                : {args.lr}"
    )
    print("=" * 72)

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

    model = PriceOnlyIntervalModel(
        price_dim=len(PRICE_COLS),
        horizon=horizon,
        hidden_dim=args.hidden_dim,
        market_emb_dim=args.market_emb_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = PriceOnlyIntervalLoss(
        lambda_boundary=args.lambda_boundary,
        lambda_center_radius=(
            args.lambda_center_radius
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

    best_path = (
        out_dir / "best_model.pt"
    )

    train_log: List[
        Dict[str, float]
    ] = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        train_losses = []

        for batch in train_loader:
            batch = move_batch(
                batch,
                device,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            pred = model(
                batch["price_seq"],
                batch["market_id"],
                batch["base_close"],
            )

            loss, _ = criterion(
                batch,
                pred,
            )

            loss.backward()

            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )

            optimizer.step()

            train_losses.append(
                float(
                    loss.detach().cpu()
                )
            )

        _, val_diag = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        train_loss = float(
            np.mean(train_losses)
        )

        val_loss = float(
            val_diag["loss"]
        )

        scheduler.step(
            val_loss
        )

        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        train_log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
        })

        if (
            epoch == 1
            or epoch % args.print_every == 0
        ):
            print(
                f"H={horizon} "
                f"epoch={epoch:03d} "
                f"train={train_loss:.6f} "
                f"val={val_loss:.6f} "
                f"lr={current_lr:.2e}"
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
                        prepared.train_end.date()
                    ),
                    "strict_train_asof_end": str(
                        prepared
                        .strict_train_asof_end
                        .date()
                    ),
                    "val_end": str(
                        prepared.val_end.date()
                    ),
                    "price_cols": PRICE_COLS,
                    "model_name": (
                        "Ablation-PriceOnly"
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
                    f"best_epoch={best_epoch}, "
                    f"best_val={best_val:.6f}"
                )
                break

    pd.DataFrame(
        train_log
    ).to_csv(
        out_dir / "train_log.csv",
        index=False,
        encoding="utf-8-sig",
    )

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

    val_pred, val_diag = evaluate(
        model,
        val_loader,
        criterion,
        device,
    )

    test_pred, test_diag = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    (
        val_overall,
        val_by_step,
    ) = compute_metrics(
        val_pred
    )

    (
        test_overall,
        test_by_step,
    ) = compute_metrics(
        test_pred
    )

    val_pred.to_csv(
        out_dir / "val_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_pred.to_csv(
        out_dir / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    val_overall.to_csv(
        out_dir / "val_metrics_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_overall.to_csv(
        out_dir / "test_metrics_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    val_by_step.to_csv(
        out_dir / "val_metrics_by_step.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_by_step.to_csv(
        out_dir / "test_metrics_by_step.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "model_name": "Ablation-PriceOnly",
        "horizon": horizon,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
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
        "hyperparameters": vars(args),
        "price_cols": PRICE_COLS,
        "ablation": {
            "news_encoder": False,
            "cross_attention": False,
            "gate": False,
            "news_residual": False,
            "market_embedding": True,
            "price_gru": True,
            "center_radius_head": True,
        },
        "fair_comparison_protocol": {
            "same_data_pipeline_as_A1": True,
            "same_split_logic_as_A1": True,
            "same_purge_logic_as_A1": True,
            "same_price_features_as_A1": True,
            "same_metric_code_as_A1": True,
            "same_lr_as_A1": True,
            "same_hidden_dim_as_A1": True,
            "same_dropout_as_A1": True,
            "same_target_loss_except_news_smooth": True,
        },
        "val_loss": val_diag,
        "test_loss": test_diag,
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

    print(
        f"\n[H={horizon}] "
        f"BEST VAL={best_val:.6f} "
        f"@ epoch {best_epoch}"
    )

    print(
        f"\n[H={horizon}] "
        f"Price-only test metrics:"
    )

    print(
        test_overall.to_string(
            index=False
        )
    )

    return test_overall.assign(
        H=horizon
    )


# ============================================================
# 6. CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Ablation: Price-only "
            "for current stockv2 A1 model"
        )
    )

    p.add_argument(
        "--root",
        type=str,
        default=str(ROOT_DEFAULT),
    )

    p.add_argument(
        "--out_dir",
        type=str,
        default="outputs_ablation_price_only",
    )

    p.add_argument(
        "--horizons",
        type=str,
        default="1,3,5",
    )

    # A1 fixed
    p.add_argument(
        "--lookback",
        type=int,
        default=20,
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
        "--lambda_boundary",
        type=float,
        default=1.0,
    )
    p.add_argument(
        "--lambda_center_radius",
        type=float,
        default=0.5,
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

    # A1 的正式学习率
    p.add_argument(
        "--lr",
        type=float,
        default=3e-4,
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
            "[WARN] CUDA 不可用，自动切换 CPU"
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
            "--horizons 不能为空"
        )

    all_frames = []

    for H in horizons:
        all_frames.append(
            train_one_horizon(
                args,
                H,
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
            c for c in final.columns
            if c != "H"
        ]
        final = final[cols]

        out_root = (
            Path(args.root)
            / args.out_dir
        )
        out_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        final.to_csv(
            out_root
            / "all_test_metrics_overall.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "\n========== "
            "Price-only Final overall test metrics "
            "=========="
        )

        print(
            final.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
