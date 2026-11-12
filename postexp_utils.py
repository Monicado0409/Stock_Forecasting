# -*- coding: utf-8 -*-
"""
postexp_utils.py

后续稳健性与可解释性实验的共享工具。
依赖同目录/项目目录中的：
    stockv2_data_tune.py
    run_stockv2_tune.py
    run_ablation_no_cross_attention.py

最终模型默认：
    outputs_ablation_no_cross_attention/H{H}/best_model.pt
即：
    Price GRU + News GRU + Scalar Gate + Residual Fusion
    无 Cross-Attention
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from stockv2_data_tune import (
    PRICE_COLS,
    STATE_COLS,
    read_csv_smart,
    load_price_data,
    prepare_for_horizon,
)

from run_stockv2_tune import (
    StockIntervalDataset,
    collate_fn,
    move_batch,
    compute_metrics,
    metric_record,
)

from run_ablation_no_cross_attention import (
    NoCrossAttentionDualStreamIntervalModel,
)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，自动切换 CPU")
        return torch.device("cpu")
    return torch.device(name)


def load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def get_ckpt_args(ckpt) -> Dict:
    args = ckpt.get("args", {})
    if hasattr(args, "__dict__"):
        args = vars(args)
    if not isinstance(args, dict):
        args = {}
    return args


def load_final_model_context(
    root: Path,
    horizon: int = 1,
    model_dir: str = "outputs_ablation_no_cross_attention",
    device: str = "cuda",
    batch_size: int = 64,
):
    """
    从最终模型 checkpoint 自动恢复主要配置，
    重新按相同无泄漏协议构造数据。
    """
    root = Path(root)
    dev = resolve_device(device)

    ckpt_path = (
        root
        / model_dir
        / f"H{horizon}"
        / "best_model.pt"
    )

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"未找到最终模型 checkpoint：\n{ckpt_path}"
        )

    ckpt = load_checkpoint(
        ckpt_path,
        dev,
    )
    a = get_ckpt_args(ckpt)

    lookback = int(a.get("lookback", 20))
    pca_dim = int(a.get("pca_dim", 16))
    train_ratio = float(a.get("train_ratio", 0.70))
    val_ratio = float(a.get("val_ratio", 0.15))
    hidden_dim = int(a.get("hidden_dim", 64))
    market_emb_dim = int(a.get("market_emb_dim", 8))
    num_layers = int(a.get("num_layers", 1))
    dropout = float(a.get("dropout", 0.20))

    prepared = prepare_for_horizon(
        root,
        horizon=horizon,
        lookback=lookback,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        pca_dim=pca_dim,
        save_dir=None,
    )

    model = NoCrossAttentionDualStreamIntervalModel(
        price_dim=len(PRICE_COLS),
        news_dim=len(prepared.news_cols),
        state_dim=len(STATE_COLS),
        horizon=horizon,
        hidden_dim=hidden_dim,
        market_emb_dim=market_emb_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(dev)

    model.load_state_dict(
        ckpt["model_state_dict"]
    )
    model.eval()

    datasets = {
        split: StockIntervalDataset(
            prepared,
            split,
            lookback,
            horizon,
        )
        for split in ("train", "val", "test")
    }

    loaders = {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        for split in datasets
    }

    config = {
        "lookback": lookback,
        "pca_dim": pca_dim,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "hidden_dim": hidden_dim,
        "market_emb_dim": market_emb_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "checkpoint": str(ckpt_path),
    }

    return (
        model,
        prepared,
        datasets,
        loaders,
        dev,
        config,
    )


def forward_with_gate_override(
    model,
    price_seq: torch.Tensor,
    news_seq: torch.Tensor,
    market_state: torch.Tensor,
    market_id: torch.Tensor,
    base_close: torch.Tensor,
    gate_override: Optional[float] = None,
):
    """
    与最终模型 forward 完全一致。
    gate_override=None：正常 Gate
    gate_override=0.0 ：关闭新闻增量，用于模型内反事实解释
    gate_override=1.0 ：完全放开新闻增量（如需诊断）
    """
    price_hidden, _ = model.price_encoder(
        price_seq
    )
    price_state = price_hidden[:, -1, :]

    market_emb = model.market_embedding(
        market_id
    )
    market_repeat = (
        market_emb
        .unsqueeze(1)
        .expand(
            -1,
            news_seq.shape[1],
            -1,
        )
    )

    news_input = torch.cat(
        [news_seq, market_repeat],
        dim=-1,
    )
    news_hidden, _ = model.news_encoder(
        news_input
    )

    news_increment = model.news_state_norm(
        news_hidden[:, -1, :]
    )

    gate_input = torch.cat(
        [
            price_state,
            news_increment,
            market_emb,
            market_state,
        ],
        dim=-1,
    )

    learned_gate = torch.sigmoid(
        model.gate(gate_input)
    )

    if gate_override is None:
        g = learned_gate
    else:
        g = torch.full_like(
            learned_gate,
            float(gate_override),
        )

    news_increment_mapped = (
        model.news_projection(
            news_increment
        )
    )

    fused = model.fusion_norm(
        price_state
        + g * news_increment_mapped
    )

    out_input = model.output_dropout(
        torch.cat(
            [fused, market_emb],
            dim=-1,
        )
    )

    center_rel = model.center_head(
        out_input
    )
    radius_rel = F.softplus(
        model.radius_head(
            out_input
        )
    )

    base = (
        base_close
        .unsqueeze(-1)
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
        "gate": learned_gate,
        "used_gate": g,
        "price_hidden_seq": price_hidden,
        "news_hidden_seq": news_hidden,
        "price_state": price_state,
        "news_increment": news_increment,
        "fused": fused,
    }


@torch.no_grad()
def infer_loader(
    model,
    loader,
    device,
    gate_override: Optional[float] = None,
    news_transform=None,
):
    """
    news_transform:
        callable(news_seq_tensor) -> transformed tensor
        用于测试阶段新闻扰动。
    """
    model.eval()
    rows = []

    for batch in loader:
        batch = move_batch(
            batch,
            device,
        )

        news_seq = batch["news_seq"]
        if news_transform is not None:
            news_seq = news_transform(
                news_seq
            )

        pred = forward_with_gate_override(
            model=model,
            price_seq=batch["price_seq"],
            news_seq=news_seq,
            market_state=batch["market_state"],
            market_id=batch["market_id"],
            base_close=batch["base_close"],
            gate_override=gate_override,
        )

        arrays = {
            "pred_low": pred["low"]
                .detach().cpu().numpy(),
            "pred_high": pred["high"]
                .detach().cpu().numpy(),
            "pred_center": pred["center"]
                .detach().cpu().numpy(),
            "pred_radius": pred["radius"]
                .detach().cpu().numpy(),
            "true_low": batch["y_low"]
                .detach().cpu().numpy(),
            "true_high": batch["y_high"]
                .detach().cpu().numpy(),
            "true_center": batch["y_center"]
                .detach().cpu().numpy(),
            "true_radius": batch["y_radius"]
                .detach().cpu().numpy(),
            "base_close": batch["base_close"]
                .detach().cpu().numpy(),
            "gate": pred["gate"]
                .detach().cpu().numpy().reshape(-1),
        }

        B, H = arrays["pred_low"].shape

        for i in range(B):
            for h in range(H):
                rows.append({
                    "market": batch["market_name"][i],
                    "as_of_date": batch["as_of_date"][i],
                    "target_date": batch["target_dates"][i][h],
                    "horizon_step": h + 1,
                    "base_close": float(
                        arrays["base_close"][i]
                    ),
                    "gate_news_increment": float(
                        arrays["gate"][i]
                    ),
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

    return pd.DataFrame(rows)


def normalize_market(s: pd.Series) -> pd.Series:
    mapping = {
        "SH": "SSE",
        "SSE": "SSE",
        "SZ": "SZI",
        "SZI": "SZI",
    }
    raw = s.astype(str).str.upper()
    return raw.map(mapping).fillna(raw)


def load_raw_market_states(
    root: Path,
) -> pd.DataFrame:
    """
    读取包含收益方向的原始市场状态。
    load_price_data() 正式面板中只保留 abs_return，
    所以这里额外保留 change_pct 用于原稿“上行/下行”分析。
    """
    root = Path(root)
    frames = []

    for filename, market in [
        ("sh_index.csv", "SSE"),
        ("sz_index.csv", "SZI"),
    ]:
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(
                f"未找到 {path}"
            )

        df = read_csv_smart(path)
        df.columns = [
            str(c).strip().replace("\ufeff", "")
            for c in df.columns
        ]

        need = [
            "date",
            "amplitude_pct",
            "change_pct",
            "turnover_rate_pct",
        ]
        missing = [
            c for c in need
            if c not in df.columns
        ]
        if missing:
            raise ValueError(
                f"{filename} 缺少字段：{missing}"
            )

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
        )
        for c in need[1:]:
            df[c] = pd.to_numeric(
                df[c],
                errors="coerce",
            )

        df = df.dropna(
            subset=need
        ).copy()

        df["market"] = market
        df["abs_return_pct"] = (
            df["change_pct"].abs()
        )

        frames.append(
            df[
                [
                    "date",
                    "market",
                    "amplitude_pct",
                    "change_pct",
                    "abs_return_pct",
                    "turnover_rate_pct",
                ]
            ]
        )

    out = pd.concat(
        frames,
        ignore_index=True,
    )
    return out.sort_values(
        ["market", "date"]
    ).reset_index(drop=True)


def merge_predictions_states(
    pred_df: pd.DataFrame,
    state_df: pd.DataFrame,
) -> pd.DataFrame:
    x = pred_df.copy()
    x["market"] = normalize_market(
        x["market"]
    )
    x["as_of_date"] = pd.to_datetime(
        x["as_of_date"]
    )

    s = state_df.copy()
    s["date"] = pd.to_datetime(
        s["date"]
    )

    out = x.merge(
        s,
        left_on=["market", "as_of_date"],
        right_on=["market", "date"],
        how="left",
        validate="many_to_one",
    )

    if out[
        [
            "amplitude_pct",
            "change_pct",
            "abs_return_pct",
            "turnover_rate_pct",
        ]
    ].isna().any().any():
        raise ValueError(
            "预测记录与市场状态合并后存在缺失，请检查日期。"
        )

    return out


def grouped_metrics(
    df: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    rows = []
    for (group, market), g in df.groupby(
        [group_col, "market"]
    ):
        r = metric_record(
            market,
            g,
        )
        r[group_col] = group
        rows.append(r)

    return pd.DataFrame(rows)


def setup_matplotlib_fonts():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {
        f.name for f
        in font_manager.fontManager.ttflist
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
            x for x in candidates
            if x in available
        ),
        None,
    )

    if chinese:
        plt.rcParams["font.sans-serif"] = [
            chinese,
            "Times New Roman",
        ]

    plt.rcParams["axes.unicode_minus"] = False
