# -*- coding: utf-8 -*-
"""
stockv2_model_tune.py

模型结构保持与当前最终版一致，仅让 news_dim 可随 PCA 维度动态变化：
- Price GRU
- News GRU + market embedding
- Price current state -> Query
- Historical news hidden sequence -> Key / Value
- Cross-Attention -> news increment z
- scalar information-absorption gate
- residual fusion: hF = hP + g * W_N z
- center/radius heads; radius uses Softplus
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class FinalDualStreamIntervalModel(nn.Module):
    def __init__(
        self,
        price_dim: int = 6,
        news_dim: int = 19,
        state_dim: int = 3,
        horizon: int = 1,
        hidden_dim: int = 64,
        market_emb_dim: int = 8,
        num_layers: int = 1,
        dropout: float = 0.20,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()

        if hidden_dim % attn_heads != 0:
            raise ValueError(
                "hidden_dim 必须能被 attn_heads 整除。"
            )

        self.horizon = horizon
        self.hidden_dim = hidden_dim

        self.market_embedding = nn.Embedding(
            2, market_emb_dim
        )

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

        self.news_encoder = nn.GRU(
            input_size=news_dim + market_emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        self.news_projection = nn.Linear(
            hidden_dim, hidden_dim
        )

        # scalar gate
        self.gate = nn.Linear(
            hidden_dim
            + hidden_dim
            + market_emb_dim
            + state_dim,
            1,
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
            self.radius_head.bias, -4.0
        )

    def forward(
        self,
        price_seq: torch.Tensor,
        news_seq: torch.Tensor,
        market_state: torch.Tensor,
        market_id: torch.Tensor,
        base_close: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        market_emb = self.market_embedding(
            market_id
        )

        price_hidden, _ = self.price_encoder(
            price_seq
        )
        price_state = price_hidden[:, -1, :]

        q_repeat = market_emb.unsqueeze(1).expand(
            -1,
            news_seq.shape[1],
            -1,
        )
        news_input = torch.cat(
            [news_seq, q_repeat],
            dim=-1,
        )
        news_hidden, _ = self.news_encoder(
            news_input
        )

        # current price state = Q
        # historical news states = K/V
        query = price_state.unsqueeze(1)
        attn_out, attn_weights = (
            self.cross_attention(
                query=query,
                key=news_hidden,
                value=news_hidden,
                need_weights=True,
                average_attn_weights=True,
            )
        )

        news_increment = (
            self.attn_norm(attn_out)
            .squeeze(1)
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
        g = torch.sigmoid(
            self.gate(gate_input)
        )

        news_increment_mapped = (
            self.news_projection(
                news_increment
            )
        )

        fused = (
            price_state
            + g * news_increment_mapped
        )
        fused = self.fusion_norm(fused)

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
            self.radius_head(out_input)
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
            "gate": g,
            "attention_weights": (
                attn_weights.squeeze(1)
            ),
            "price_hidden_seq": price_hidden,
            "news_hidden_seq": news_hidden,
            "price_state": price_state,
            "news_increment": news_increment,
            "fused": fused,
        }


class FinalIntervalLoss(nn.Module):
    def __init__(
        self,
        lambda_boundary: float = 1.0,
        lambda_center_radius: float = 0.5,
        lambda_news_smooth: float = 0.005,
        huber_delta: float = 0.01,
    ) -> None:
        super().__init__()
        self.lambda_boundary = (
            lambda_boundary
        )
        self.lambda_center_radius = (
            lambda_center_radius
        )
        self.lambda_news_smooth = (
            lambda_news_smooth
        )
        self.huber_delta = huber_delta

    def _huber_zero(
        self, x: torch.Tensor
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
            pred["center"]
            - batch["y_center"]
        ) / base
        radius_err_rel = (
            pred["radius"]
            - batch["y_radius"]
        ) / base

        boundary_loss = 0.5 * (
            self._huber_zero(
                low_err_rel
            )
            + self._huber_zero(
                high_err_rel
            )
        )

        center_radius_loss = (
            self._huber_zero(
                center_err_rel
            )
            + self._huber_zero(
                radius_err_rel
            )
        )

        news_hidden = pred[
            "news_hidden_seq"
        ]
        if (
            news_hidden.shape[1] > 1
            and self.lambda_news_smooth > 0
        ):
            hidden_diff = (
                news_hidden[:, 1:, :]
                - news_hidden[:, :-1, :]
            )
            smooth_loss = torch.mean(
                hidden_diff.pow(2)
            )
        else:
            smooth_loss = torch.zeros(
                (),
                dtype=pred["center"].dtype,
                device=pred["center"].device,
            )

        total = (
            self.lambda_boundary
            * boundary_loss
            + self.lambda_center_radius
            * center_radius_loss
            + self.lambda_news_smooth
            * smooth_loss
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
            "news_smooth": float(
                smooth_loss.detach().cpu()
            ),
        }
        return total, parts
