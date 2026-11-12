# -*- coding: utf-8 -*-
r"""
stockv2_data_tune.py

调参版数据处理模块，基于当前 stockv2 最终模型协议：
1) 价格输入：open, high, low, close, volume, turnover
2) 市场状态只进入 gate：amplitude_pct, abs_return_pct, turnover_rate_pct
3) 新闻输入：sentiment + impact + relevance + PCA semantics
4) PCA 维度通过 --pca_dim 动态设置
5) PCA / scaler 仅在严格训练可见区间拟合
6) 时间协议：
      Price : t-L+1 ... t
      News  : t-L   ... t-1
      Target: t+1 ... t+H
7) 不使用 bfill；保留“前一交易日最后一条新闻”填充规则
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT_DEFAULT = Path(r"E:\Code\stockv2")

PRICE_COLS = ["open", "high", "low", "close", "volume", "turnover"]
STATE_SOURCE_COLS = ["amplitude_pct", "change_pct", "turnover_rate_pct"]
STATE_COLS = ["amplitude_pct", "abs_return_pct", "turnover_rate_pct"]
NEWS_BASE_COLS = ["sentiment", "impact", "relevance"]

MARKETS = ("SSE", "SZI")
MARKET_TO_ID = {"SSE": 0, "SZI": 1}


def get_news_cols(pca_dim: int) -> List[str]:
    if pca_dim < 1:
        raise ValueError("pca_dim 必须 >= 1")
    return NEWS_BASE_COLS + [
        f"semantic_{i:02d}" for i in range(1, pca_dim + 1)
    ]


def read_csv_smart(path: Path) -> pd.DataFrame:
    last = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            last = exc
    raise RuntimeError(f"无法读取 CSV：{path}；最后错误：{last}")


def find_existing(root: Path, names: Sequence[str]) -> Path:
    for name in names:
        p = root / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"在 {root} 下未找到以下任一文件：{list(names)}"
    )


def load_price_data(root: Path) -> pd.DataFrame:
    sh_path = find_existing(root, ["sh_index.csv"])
    sz_path = find_existing(root, ["sz_index.csv"])

    frames = []
    for path, market in [(sh_path, "SSE"), (sz_path, "SZI")]:
        df = read_csv_smart(path)
        df.columns = [
            str(c).strip().replace("\ufeff", "") for c in df.columns
        ]
        if "date" not in df.columns:
            raise ValueError(f"{path.name} 缺少字段：date")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in PRICE_COLS + STATE_SOURCE_COLS:
            if c not in df.columns:
                raise ValueError(f"{path.name} 缺少字段：{c}")
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(
            subset=["date"] + PRICE_COLS + STATE_SOURCE_COLS
        )
        df = (
            df.sort_values("date")
            .drop_duplicates("date", keep="last")
            .reset_index(drop=True)
        )
        df["market"] = market
        df["abs_return_pct"] = df["change_pct"].abs()

        # 监督目标，不进入价格分支
        df["interval_low"] = df["low"]
        df["interval_high"] = df["high"]
        df["interval_center"] = (df["high"] + df["low"]) / 2.0
        df["interval_radius"] = (df["high"] - df["low"]) / 2.0

        if (df["interval_radius"] < 0).any():
            raise ValueError(f"{path.name} 存在 high < low 的异常记录")

        frames.append(
            df[
                [
                    "date",
                    "market",
                    *PRICE_COLS,
                    *STATE_COLS,
                    "interval_low",
                    "interval_high",
                    "interval_center",
                    "interval_radius",
                ]
            ]
        )

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["market", "date"]).reset_index(drop=True)

    dates_sse = set(panel.loc[panel["market"] == "SSE", "date"])
    dates_szi = set(panel.loc[panel["market"] == "SZI", "date"])
    if dates_sse != dates_szi:
        raise ValueError("SSE 与 SZI 交易日期集合不一致，请先核验行情数据。")
    return panel


@dataclass
class NewsRaw:
    items: pd.DataFrame
    embeddings: np.ndarray


def load_news_raw(root: Path) -> NewsRaw:
    structured_path = find_existing(
        root,
        [
            "news_qwen_structured_features(3).csv",
            "news_qwen_structured_features(2).csv",
            "news_qwen_structured_features(1).csv",
            "news_qwen_structured_features.csv",
        ],
    )
    npz_path = find_existing(
        root,
        [
            "news_qwen_embeddings_1024(1).npz",
            "news_qwen_embeddings_1024.npz",
        ],
    )

    df = read_csv_smart(structured_path)
    df.columns = [
        str(c).strip().replace("\ufeff", "") for c in df.columns
    ]

    required = [
        "news_id",
        "date",
        "rank_in_day",
        "sentiment_score",
        "impact_strength",
        "sse_relevance_score",
        "szi_relevance_score",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"{structured_path.name} 缺少字段：{c}")

    df["news_id"] = df["news_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in [
        "rank_in_day",
        "sentiment_score",
        "impact_strength",
        "sse_relevance_score",
        "szi_relevance_score",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "status" in df.columns:
        bad = df["status"].astype(str).str.lower().ne("success")
        if bad.any():
            raise ValueError(
                f"结构化新闻存在 {int(bad.sum())} 条非 success 记录。"
            )

    npz = np.load(npz_path, allow_pickle=True)
    if "news_id" not in npz or "embeddings" not in npz:
        raise ValueError(f"{npz_path.name} 必须包含 news_id 与 embeddings")

    ids = np.asarray(npz["news_id"]).astype(str)
    emb = np.asarray(npz["embeddings"], dtype=np.float32)

    if emb.ndim != 2 or emb.shape[1] != 1024:
        raise ValueError(
            f"Embedding 维度异常：{emb.shape}，期望 [N,1024]"
        )
    if len(ids) != len(emb):
        raise ValueError("NPZ 中 news_id 与 embeddings 行数不一致。")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("NPZ 中 news_id 存在重复。")

    id_to_row = {nid: i for i, nid in enumerate(ids.tolist())}
    missing = [
        nid for nid in df["news_id"] if nid not in id_to_row
    ]
    if missing:
        raise ValueError(
            f"有 {len(missing)} 条结构化新闻在 NPZ 中找不到 embedding。"
        )

    order = np.asarray(
        [id_to_row[nid] for nid in df["news_id"]],
        dtype=np.int64,
    )
    emb_aligned = emb[order]

    keep = df["date"].notna()
    df = df.loc[keep].reset_index(drop=True)
    emb_aligned = emb_aligned[keep.to_numpy()]

    for c in [
        "sentiment_score",
        "impact_strength",
        "sse_relevance_score",
        "szi_relevance_score",
    ]:
        if df[c].isna().any():
            raise ValueError(f"结构化新闻字段 {c} 存在 NaN。")

    if not np.isfinite(emb_aligned).all():
        raise ValueError("Embedding 中存在 NaN/Inf。")

    return NewsRaw(items=df, embeddings=emb_aligned)


def preserve_or_fill_trading_day_news(
    news_raw: NewsRaw,
    trading_dates: Sequence[pd.Timestamp],
) -> NewsRaw:
    """
    保留已有填充记录；如果整个交易日无新闻，则复制最近已有交易日
    rank 最大（最后一条）的新闻，保持既有填充协议。
    """
    df = news_raw.items.copy().reset_index(drop=True)
    emb = news_raw.embeddings.copy()

    date_to_idx: Dict[pd.Timestamp, List[int]] = {}
    for i, d in enumerate(pd.to_datetime(df["date"])):
        date_to_idx.setdefault(pd.Timestamp(d), []).append(i)

    extra_rows = []
    extra_emb = []
    available_dates = set(date_to_idx.keys())

    for d in sorted(pd.Timestamp(x) for x in trading_dates):
        if d in available_dates:
            continue

        prev_dates = [x for x in available_dates if x < d]
        if not prev_dates:
            raise ValueError(
                f"{d.date()} 无新闻且不存在更早交易日新闻，无法执行前日填充。"
            )

        src_date = max(prev_dates)
        src_candidates = date_to_idx[src_date]
        src_i = max(
            src_candidates,
            key=lambda j: (
                float(df.loc[j, "rank_in_day"])
                if pd.notna(df.loc[j, "rank_in_day"])
                else -1.0
            ),
        )

        row = df.loc[src_i].copy()
        original_id = str(row["news_id"])
        row["news_id"] = (
            f"{original_id}_fill_{d.strftime('%Y%m%d')}"
        )
        row["date"] = d
        row["rank_in_day"] = 1

        if "record_status" in df.columns:
            row["record_status"] = (
                "filled_runtime_previous_trading_day_last_news"
            )
        if "fill_from_date" in df.columns:
            row["fill_from_date"] = src_date.strftime("%Y-%m-%d")

        extra_rows.append(row)
        extra_emb.append(emb[src_i].copy())

        new_index = len(df) + len(extra_rows) - 1
        date_to_idx[d] = [new_index]
        available_dates.add(d)

    if extra_rows:
        df = pd.concat(
            [df, pd.DataFrame(extra_rows)],
            ignore_index=True,
        )
        emb = np.vstack(
            [emb, np.asarray(extra_emb, dtype=np.float32)]
        )

    sort_order = np.lexsort(
        (
            pd.to_numeric(
                df["rank_in_day"], errors="coerce"
            )
            .fillna(999999)
            .to_numpy(),
            pd.to_datetime(df["date"]).astype("int64"),
        )
    )
    df = df.iloc[sort_order].reset_index(drop=True)
    emb = emb[sort_order]
    return NewsRaw(df, emb)


def get_split_dates(
    trading_dates: Sequence[pd.Timestamp],
    lookback: int,
    horizon: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    dates = [pd.Timestamp(d) for d in sorted(trading_dates)]

    # i=lookback 才同时具备：
    # Price t-L+1...t 与 News t-L...t-1
    sample_dates = dates[lookback : len(dates) - horizon]
    if len(sample_dates) < 30:
        raise ValueError("可用样本太少。")

    train_idx = max(
        0,
        min(
            int(len(sample_dates) * train_ratio) - 1,
            len(sample_dates) - 3,
        ),
    )
    val_idx = max(
        train_idx + 1,
        min(
            int(
                len(sample_dates)
                * (train_ratio + val_ratio)
            )
            - 1,
            len(sample_dates) - 2,
        ),
    )
    return sample_dates[train_idx], sample_dates[val_idx]


@dataclass
class HorizonPrepared:
    horizon: int
    pca_dim: int
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    strict_train_asof_end: pd.Timestamp
    train_news_end: pd.Timestamp
    price_panel: pd.DataFrame
    news_daily: pd.DataFrame
    news_cols: List[str]
    pca: PCA
    price_scaler: StandardScaler
    news_scaler: StandardScaler
    state_scaler: StandardScaler
    pca_explained_variance_ratio_sum: float


def aggregate_news_daily(
    news_items: pd.DataFrame,
    embeddings_reduced: np.ndarray,
    trading_dates: Sequence[pd.Timestamp],
    eps: float = 1e-12,
) -> pd.DataFrame:
    if len(news_items) != len(embeddings_reduced):
        raise ValueError(
            "news_items 与 embeddings_reduced 行数不一致。"
        )

    tmp = news_items.reset_index(drop=True).copy()
    rows = []

    for date, idx in tmp.groupby("date").groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        sub = tmp.iloc[idx]
        e = embeddings_reduced[idx]

        s = sub["sentiment_score"].astype(float).to_numpy()
        a = sub["impact_strength"].astype(float).to_numpy()

        for market, rel_col in [
            ("SSE", "sse_relevance_score"),
            ("SZI", "szi_relevance_score"),
        ]:
            rho = sub[rel_col].astype(float).to_numpy()

            # 与当前模型一致：impact * relevance 加权
            w = (
                np.clip(a, 0.0, None)
                * np.clip(rho, 0.0, None)
            )
            # 保持与当前正式代码完全一致：
            # alpha = w / (sum(w) + eps)
            # 若所有权重均为0，则该日加权表示自然为0。
            denom = float(w.sum()) + eps
            alpha = w / denom

            sentiment = float(np.sum(alpha * s))
            impact = float(np.sum(alpha * a))
            relevance = float(np.sum(alpha * rho))
            semantic = np.sum(alpha[:, None] * e, axis=0)

            row = {
                "date": pd.Timestamp(date),
                "market": market,
                "sentiment": sentiment,
                "impact": impact,
                "relevance": relevance,
            }
            for j, value in enumerate(semantic, start=1):
                row[f"semantic_{j:02d}"] = float(value)
            rows.append(row)

    daily = pd.DataFrame(rows)
    daily = daily.sort_values(
        ["market", "date"]
    ).reset_index(drop=True)

    expected = {
        (m, pd.Timestamp(d))
        for m in MARKETS
        for d in trading_dates
    }
    actual = {
        (r.market, pd.Timestamp(r.date))
        for r in daily.itertuples()
    }
    missing = expected - actual
    if missing:
        example = sorted(
            list(missing), key=lambda x: (x[0], x[1])
        )[:5]
        raise ValueError(
            f"日级新闻仍缺少 {len(missing)} 个 market-date，例如：{example}"
        )

    return daily


def prepare_for_horizon(
    root: Path,
    horizon: int,
    lookback: int = 20,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    pca_dim: int = 16,
    save_dir: Optional[Path] = None,
) -> HorizonPrepared:
    if horizon < 1:
        raise ValueError("horizon 必须 >= 1")
    if lookback < 2:
        raise ValueError("lookback 必须 >= 2")
    if pca_dim < 1 or pca_dim > 1024:
        raise ValueError("pca_dim 必须在 [1,1024] 内")

    news_cols = get_news_cols(pca_dim)

    price = load_price_data(root)
    trading_dates = sorted(
        price.loc[price["market"] == "SSE", "date"].unique()
    )
    trading_dates_ts = [
        pd.Timestamp(d) for d in trading_dates
    ]

    train_end, val_end = get_split_dates(
        trading_dates_ts,
        lookback,
        horizon,
        train_ratio,
        val_ratio,
    )

    # --------------------------------------------------------
    # 严格 purge：H>1 时，训练/验证边界处不允许 target window 重叠。
    #
    # train 最后 as-of 允许的 target 最后一天 = val 第一个 as-of 日；
    # 因而需从 train_end 向前 purge H-1 个 as-of。
    # H=1 时 strict_train_asof_end == train_end，不改变当前 H=1 结果协议。
    # --------------------------------------------------------
    train_end_pos = trading_dates_ts.index(
        pd.Timestamp(train_end)
    )
    strict_train_pos = train_end_pos - (horizon - 1)
    if strict_train_pos < lookback:
        raise ValueError(
            "严格 purge 后训练区间过短。"
        )
    strict_train_asof_end = trading_dates_ts[
        strict_train_pos
    ]

    # 新闻输入严格 lag-1，因此训练样本可见的最后新闻日
    # 是 strict_train_asof_end 的前一交易日。
    if strict_train_pos < 1:
        raise ValueError(
            "严格训练截止日前不存在可用新闻交易日。"
        )
    train_news_end = trading_dates_ts[
        strict_train_pos - 1
    ]

    news_raw = load_news_raw(root)
    news_raw = preserve_or_fill_trading_day_news(
        news_raw, trading_dates_ts
    )

    news_dates = pd.to_datetime(news_raw.items["date"])
    train_mask = (
        news_dates <= train_news_end
    ).to_numpy()

    if int(train_mask.sum()) < pca_dim:
        raise ValueError(
            f"训练期新闻样本 {int(train_mask.sum())} "
            f"不足以拟合 PCA{pca_dim}。"
        )

    pca = PCA(
        n_components=pca_dim,
        random_state=42,
    )
    pca.fit(news_raw.embeddings[train_mask])
    e_reduced = pca.transform(
        news_raw.embeddings
    ).astype(np.float32)

    news_daily = aggregate_news_daily(
        news_raw.items,
        e_reduced,
        trading_dates_ts,
    )

    # 所有 scaler 仅用“实际训练样本可见”的区间拟合
    price_train = price[
        price["date"] <= strict_train_asof_end
    ]
    news_train = news_daily[
        news_daily["date"] <= train_news_end
    ]

    price_scaler = StandardScaler().fit(
        price_train[PRICE_COLS].astype(float)
    )
    news_scaler = StandardScaler().fit(
        news_train[news_cols].astype(float)
    )
    state_scaler = StandardScaler().fit(
        price_train[STATE_COLS].astype(float)
    )

    prepared = HorizonPrepared(
        horizon=horizon,
        pca_dim=pca_dim,
        train_end=train_end,
        val_end=val_end,
        strict_train_asof_end=strict_train_asof_end,
        train_news_end=train_news_end,
        price_panel=price,
        news_daily=news_daily,
        news_cols=news_cols,
        pca=pca,
        price_scaler=price_scaler,
        news_scaler=news_scaler,
        state_scaler=state_scaler,
        pca_explained_variance_ratio_sum=float(
            pca.explained_variance_ratio_.sum()
        ),
    )

    if save_dir is not None:
        save_dir.mkdir(
            parents=True, exist_ok=True
        )
        news_daily.to_csv(
            save_dir
            / "news_daily_market_features_tune.csv",
            index=False,
            encoding="utf-8-sig",
        )
        price.to_csv(
            save_dir / "price_panel_tune.csv",
            index=False,
            encoding="utf-8-sig",
        )

        audit = {
            "horizon": horizon,
            "lookback": lookback,
            "pca_dim": pca_dim,
            "train_end_split_boundary": str(
                train_end.date()
            ),
            "val_end_split_boundary": str(
                val_end.date()
            ),
            "strict_train_asof_end_after_purge": str(
                strict_train_asof_end.date()
            ),
            "train_news_end_lag1": str(
                train_news_end.date()
            ),
            "price_rows": int(len(price)),
            "news_item_rows": int(
                len(news_raw.items)
            ),
            "news_daily_rows": int(
                len(news_daily)
            ),
            "price_cols": PRICE_COLS,
            "market_state_cols": STATE_COLS,
            "news_cols": news_cols,
            "pca_fit_news_count": int(
                train_mask.sum()
            ),
            "pca_explained_variance_ratio_sum": float(
                pca.explained_variance_ratio_.sum()
            ),
            "filled_record_status_counts": (
                news_raw.items["record_status"]
                .astype(str)
                .value_counts()
                .to_dict()
                if "record_status"
                in news_raw.items.columns
                else {}
            ),
            "time_protocol": {
                "price": "t-L+1 ... t",
                "news": "t-L ... t-1",
                "target": "t+1 ... t+H",
            },
            "leakage_controls": {
                "no_bfill": True,
                "pca_train_only": True,
                "scalers_train_only": True,
                "news_lag_1": True,
                "purge_target_overlap": True,
            },
        }
        with open(
            save_dir / "data_audit.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                audit,
                f,
                ensure_ascii=False,
                indent=2,
            )

    return prepared
