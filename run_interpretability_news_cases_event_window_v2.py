# -*- coding: utf-8 -*-
"""
run_interpretability_news_cases_event_window.py

图7（改进版）：单一典型事件窗口的相关新闻影响强度分析
========================================================

设计目标
--------
相较于“直接选 Gate 最高的 3 个样本”，本脚本改为优先选择：

    1) 同一市场；
    2) 连续 3 个交易日；
    3) 每天至少有 min_news 条相关新闻；
    4) 窗口内 Gate 水平较高；
    5) 窗口内新闻影响强度较高。

展示规则（V2）
-------------
窗口筛选、窗口评分和绘图展示不再共用同一个 top_k：
    - min_news：候选日最低新闻数，默认 3；
    - score_top_k：窗口评分使用的高影响新闻条数，默认 3；
    - display_top_k：图中最多展示的新闻条数，默认 5。

因此某日有 4/3/5 条新闻时，图中会分别展示 4/3/5 条，
但窗口评分仍统一基于每日 Top-3 新闻，避免新闻数量多的日期天然占优。

从而让图7真正回答：
    “在一个连续典型事件窗口中，模型实际接收了什么新闻？”

核心口径
--------
新闻影响强度定义为：

    impact_relevance_score = impact_strength × market_relevance_score

注意：
    该指标用于刻画日级新闻聚合中的相对重要程度，
    不代表神经网络 attention 权重。

案例选择方式
------------
A. 自动选择（默认）
   在每个市场内部，搜索连续 3 个交易日窗口。
   对每个窗口计算：

       window_score
         = mean(gate_percentile_within_market)
         + mean(news_impact_percentile_within_market)

   选取得分最高的窗口作为“典型事件窗口”。

B. 手工指定（推荐在论文定稿时使用）
   可以指定：
       --market SSE / SZI
       --event_start_date YYYY-MM-DD
   或：
       --market SSE / SZI
       --event_center_date YYYY-MM-DD

   这样可以强制使用你认为最有金融解释力的事件窗口。

输出
----
    outputs_interpretability/
        fig7_event_window_combined.png
        fig7_event_window_combined.pdf
        fig7_event_window_selected_window.csv
        fig7_event_window_case_news_details.csv
        fig7_event_window_candidate_windows.csv
        fig7_event_window_daily_scores.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stockv2_data_tune import load_news_raw
from postexp_utils import (
    load_final_model_context,
    infer_loader,
    setup_matplotlib_fonts,
)


TITLE_CANDIDATES = [
    "title",
    "news_title",
    "headline",
    "text",
    "content",
    "raw_title",
]

MARKET_CN = {
    "SSE": "上证指数",
    "SZI": "深证成指",
}


def find_title_col(df: pd.DataFrame) -> Optional[str]:
    for c in TITLE_CANDIDATES:
        if c in df.columns:
            return c
    return None


def shorten(text, max_chars=48):
    s = str(text).replace("\n", " ").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars - 1] + "…"


def rank_pct(s: pd.Series) -> pd.Series:
    if len(s) == 1:
        return pd.Series([1.0], index=s.index)
    return s.rank(method="average", pct=True)


def build_market_trading_dates(
    prepared
) -> Dict[str, List[pd.Timestamp]]:
    out = {}
    panel = prepared.price_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    for market, g in panel.groupby("market"):
        out[market] = sorted(pd.to_datetime(g["date"].unique()))
    return out


def get_previous_trading_day(
    trading_dates: List[pd.Timestamp],
    as_of_date: pd.Timestamp,
) -> Optional[pd.Timestamp]:
    pos_map = {d: i for i, d in enumerate(trading_dates)}
    pos = pos_map.get(as_of_date, None)
    if pos is None or pos < 1:
        return None
    return trading_dates[pos - 1]


def get_window_dates_from_start(
    trading_dates: List[pd.Timestamp],
    start_date: pd.Timestamp,
    n_days: int = 3,
) -> Optional[List[pd.Timestamp]]:
    pos_map = {d: i for i, d in enumerate(trading_dates)}
    pos = pos_map.get(start_date, None)
    if pos is None:
        return None
    if pos + n_days - 1 >= len(trading_dates):
        return None
    return trading_dates[pos : pos + n_days]


def get_window_dates_from_center(
    trading_dates: List[pd.Timestamp],
    center_date: pd.Timestamp,
    n_days: int = 3,
) -> Optional[List[pd.Timestamp]]:
    if n_days != 3:
        raise ValueError("当前仅支持3日窗口。")
    pos_map = {d: i for i, d in enumerate(trading_dates)}
    pos = pos_map.get(center_date, None)
    if pos is None or pos - 1 < 0 or pos + 1 >= len(trading_dates):
        return None
    return trading_dates[pos - 1 : pos + 2]


def prepare_daily_case_table(
    pred: pd.DataFrame,
    raw: pd.DataFrame,
    title_col: Optional[str],
    market_trading_dates: Dict[str, List[pd.Timestamp]],
    min_news: int,
    score_top_k: int,
    display_top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    为每个测试样本（日级）生成：
        - 对应 t-1 新闻日
        - 当日 Gate
        - 用于窗口评分的 Top-score_top_k 新闻统计
        - 用于绘图展示的 Top-display_top_k 新闻详情

    V2 关键变化：
        1) 候选日只要求 n_available_news >= min_news；
        2) 窗口评分固定使用每日 Top-score_top_k 新闻；
        3) 图中最多展示 display_top_k 条新闻，实际有几条就展示几条；
        4) 因此 4/3/5 条新闻的三日窗口会显示为 4/3/5，而不是 3/3/3。
    """
    if min_news < 1:
        raise ValueError("min_news 必须 >= 1")
    if score_top_k < 1:
        raise ValueError("score_top_k 必须 >= 1")
    if display_top_k < 1:
        raise ValueError("display_top_k 必须 >= 1")
    if score_top_k > display_top_k:
        raise ValueError("建议 score_top_k <= display_top_k，以保证评分新闻都能在图中看到。")

    daily_rows = []
    news_rows = []

    for row in pred.itertuples():
        market = row.market
        as_of_date = pd.Timestamp(row.as_of_date)

        prev_day = get_previous_trading_day(
            market_trading_dates[market],
            as_of_date,
        )
        if prev_day is None:
            continue

        day_news = raw[raw["date"] == prev_day].copy()
        if len(day_news) < min_news:
            continue

        rel_col = (
            "sse_relevance_score"
            if market == "SSE"
            else "szi_relevance_score"
        )

        day_news["market_relevance"] = pd.to_numeric(
            day_news[rel_col],
            errors="coerce",
        )
        day_news["impact_strength"] = pd.to_numeric(
            day_news["impact_strength"],
            errors="coerce",
        )
        day_news["sentiment_score"] = pd.to_numeric(
            day_news["sentiment_score"],
            errors="coerce",
        )

        day_news["impact_relevance_score"] = (
            day_news["impact_strength"].clip(lower=0)
            * day_news["market_relevance"].clip(lower=0)
        )

        ranked = day_news.sort_values(
            "impact_relevance_score",
            ascending=False,
        ).copy()

        # 窗口评分：固定使用 Top-score_top_k，避免“新闻多”本身抬高窗口分数。
        score_top = ranked.head(score_top_k).copy()

        # 绘图展示：最多展示 display_top_k，某日不足则全部展示。
        display_top = ranked.head(display_top_k).copy()

        if title_col is None:
            display_top["display_title"] = display_top["news_id"].astype(str)
        else:
            display_top["display_title"] = display_top[title_col].astype(str)

        total_score = float(score_top["impact_relevance_score"].sum())
        max_score = float(score_top["impact_relevance_score"].max())
        mean_score = float(score_top["impact_relevance_score"].mean())

        daily_rows.append({
            "market": market,
            "as_of_date": as_of_date,
            "news_date": prev_day,
            "gate": float(row.gate_news_increment),
            # 为兼容后续窗口选择函数，沿用 topk_* 字段名；其含义为 score_top_k。
            "topk_news_total_score": total_score,
            "top1_news_score": max_score,
            "topk_news_mean_score": mean_score,
            "n_available_news": int(len(day_news)),
            "n_score_news": int(len(score_top)),
            "n_display_news": int(len(display_top)),
        })

        for rank, r in enumerate(display_top.itertuples(), start=1):
            news_rows.append({
                "market": market,
                "as_of_date": as_of_date,
                "news_date": prev_day,
                "gate": float(row.gate_news_increment),
                "rank": rank,
                "news_id": getattr(r, "news_id"),
                "title": getattr(r, "display_title"),
                "sentiment": float(getattr(r, "sentiment_score")),
                "impact": float(getattr(r, "impact_strength")),
                "relevance": float(getattr(r, "market_relevance")),
                "impact_relevance_score": float(
                    getattr(r, "impact_relevance_score")
                ),
                "used_for_window_score": bool(rank <= score_top_k),
            })

    daily_df = pd.DataFrame(daily_rows)
    news_df = pd.DataFrame(news_rows)

    if len(daily_df) == 0:
        raise RuntimeError("没有生成任何满足最低新闻数量要求的日级样本。")

    daily_df = daily_df.sort_values(
        ["market", "as_of_date"]
    ).reset_index(drop=True)

    # 市场内百分位：避免 SZI 因 Gate 基准更高而天然霸榜
    daily_df["gate_pct_within_market"] = (
        daily_df.groupby("market")["gate"].transform(rank_pct)
    )
    daily_df["news_score_pct_within_market"] = (
        daily_df.groupby("market")["topk_news_total_score"].transform(rank_pct)
    )

    return daily_df, news_df


def enumerate_candidate_windows(
    daily_df: pd.DataFrame,
    market_trading_dates: Dict[str, List[pd.Timestamp]],
    n_days: int = 3,
    min_peak_news: int = 0,
) -> pd.DataFrame:
    rows = []

    for market, g in daily_df.groupby("market"):
        g = g.sort_values("as_of_date").copy()
        date_set = set(g["as_of_date"])
        score_map = g.set_index("as_of_date").to_dict(orient="index")
        trading_dates = market_trading_dates[market]

        for start_date in trading_dates:
            window_dates = get_window_dates_from_start(
                trading_dates,
                start_date,
                n_days=n_days,
            )
            if window_dates is None:
                continue

            if not all(d in date_set for d in window_dates):
                continue

            sub = pd.DataFrame(
                [score_map[d] for d in window_dates]
            ).copy()
            sub["as_of_date"] = window_dates

            # 可选约束：窗口内至少有一天达到指定新闻数量。
            # 例如 --min_peak_news 5 可要求三日中至少一天有 5 条新闻。
            if min_peak_news > 0:
                if int(sub["n_available_news"].max()) < min_peak_news:
                    continue

            mean_gate = float(sub["gate"].mean())
            mean_gate_pct = float(sub["gate_pct_within_market"].mean())
            mean_news_score = float(sub["topk_news_total_score"].mean())
            mean_news_pct = float(
                sub["news_score_pct_within_market"].mean()
            )

            # 默认窗口得分：兼顾高 Gate 与高新闻影响
            window_score = mean_gate_pct + mean_news_pct

            rows.append({
                "market": market,
                "market_cn": MARKET_CN.get(market, market),
                "start_date": window_dates[0],
                "center_date": window_dates[1],
                "end_date": window_dates[-1],
                "window_dates": "|".join(
                    [str(d.date()) for d in window_dates]
                ),
                "mean_gate": mean_gate,
                "mean_gate_pct": mean_gate_pct,
                "mean_news_total_score": mean_news_score,
                "mean_news_score_pct": mean_news_pct,
                "window_score": window_score,
                "available_news_counts": "|".join(
                    [str(int(v)) for v in sub["n_available_news"].tolist()]
                ),
                "max_available_news": int(sub["n_available_news"].max()),
            })

    cand = pd.DataFrame(rows)
    if len(cand) == 0:
        raise RuntimeError("未找到任何连续3日候选窗口。")

    cand = cand.sort_values(
        ["window_score", "mean_gate_pct", "mean_news_score_pct"],
        ascending=False,
    ).reset_index(drop=True)

    return cand


def resolve_selected_window(
    candidate_windows: pd.DataFrame,
    market_trading_dates: Dict[str, List[pd.Timestamp]],
    market: str,
    event_start_date: Optional[str],
    event_center_date: Optional[str],
    n_days: int = 3,
) -> pd.DataFrame:
    if event_start_date and event_center_date:
        raise ValueError("event_start_date 与 event_center_date 只能二选一。")

    # 手工指定窗口
    if event_start_date is not None or event_center_date is not None:
        if market not in ["SSE", "SZI"]:
            raise ValueError("手工指定日期时，必须同时指定 --market SSE 或 --market SZI。")

        trading_dates = market_trading_dates[market]

        if event_start_date is not None:
            start_dt = pd.Timestamp(event_start_date)
            dates = get_window_dates_from_start(trading_dates, start_dt, n_days=n_days)
        else:
            center_dt = pd.Timestamp(event_center_date)
            dates = get_window_dates_from_center(trading_dates, center_dt, n_days=n_days)

        if dates is None:
            raise RuntimeError("指定的窗口日期无效，无法构成连续3个交易日。")

        key = "|".join([str(d.date()) for d in dates])

        hit = candidate_windows[
            (candidate_windows["market"] == market)
            & (candidate_windows["window_dates"] == key)
        ].copy()

        if len(hit) == 0:
            raise RuntimeError(
                "指定窗口未进入候选集。可能其中某一天没有足够数量的相关新闻。"
            )

        return hit.head(1)

    # 自动选择
    if market in ["SSE", "SZI"]:
        sub = candidate_windows[candidate_windows["market"] == market].copy()
        if len(sub) == 0:
            raise RuntimeError(f"市场 {market} 没有可用候选窗口。")
        return sub.head(1)

    # market == auto
    return candidate_windows.head(1)


def draw_combined_figure(
    selected_window: pd.DataFrame,
    daily_df: pd.DataFrame,
    news_df: pd.DataFrame,
    out_dir: Path,
    score_top_k: int,
    display_top_k: int,
):
    row = selected_window.iloc[0]
    market = row["market"]
    market_cn = row["market_cn"]
    window_dates = [
        pd.Timestamp(x) for x in str(row["window_dates"]).split("|")
    ]

    # 汇总三个面板数据。window_dates 是预测基准日 t；news_date 是实际输入新闻日 t-1。
    panel_rows = []
    news_dates = []

    for i, as_of_date in enumerate(window_dates, start=1):
        day_meta = daily_df[
            (daily_df["market"] == market)
            & (daily_df["as_of_date"] == as_of_date)
        ].copy()

        if len(day_meta) != 1:
            raise RuntimeError(f"{market} {as_of_date.date()} 找不到唯一日级记录。")

        gate = float(day_meta["gate"].iloc[0])
        news_date = pd.Timestamp(day_meta["news_date"].iloc[0])
        n_available = int(day_meta["n_available_news"].iloc[0])
        news_dates.append(news_date)

        top = news_df[
            (news_df["market"] == market)
            & (news_df["as_of_date"] == as_of_date)
        ].copy()

        # barh 从下往上画，因此用升序让最高影响新闻位于最上方。
        top = top.sort_values(
            "impact_relevance_score",
            ascending=True,
        )

        for _, r in top.iterrows():
            panel_rows.append({
                "panel_id": i,
                "market": market,
                "market_cn": market_cn,
                "as_of_date": as_of_date,
                "news_date": news_date,
                "gate": gate,
                "n_available_news": n_available,
                "n_display_news": int(len(top)),
                "title": r["title"],
                "rank": r["rank"],
                "impact_relevance_score": r["impact_relevance_score"],
                "used_for_window_score": r.get("used_for_window_score", False),
            })

    panel_df = pd.DataFrame(panel_rows)
    panel_df.to_csv(
        out_dir / "fig7_event_window_case_news_details.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 统一 x 轴上限，方便 3 天横向比较
    global_xmax = float(panel_df["impact_relevance_score"].max())
    x_hi = global_xmax * 1.08 if global_xmax > 0 else 1.0

    setup_matplotlib_fonts()

    # 根据展示新闻总数自适应图高；4/3/5 条时比原图稍高，避免标题拥挤。
    fig_height = max(9.8, 4.0 + 0.55 * len(panel_df))
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(12.4, fig_height),
        squeeze=False,
    )
    axes = axes[:, 0]

    for panel_id, as_of_date in enumerate(window_dates, start=1):
        ax = axes[panel_id - 1]
        g = panel_df[panel_df["panel_id"] == panel_id].copy()

        labels = [
            shorten(x, max_chars=56)
            for x in g["title"]
        ]

        ax.barh(
            labels,
            g["impact_relevance_score"].to_numpy(),
            alpha=0.92,
        )

        gate = float(g["gate"].iloc[0])
        news_date = pd.Timestamp(g["news_date"].iloc[0])
        n_available = int(g["n_available_news"].iloc[0])
        n_display = int(g["n_display_news"].iloc[0])

        # 明确区分“预测基准日 t”和“新闻日期 t-1”，避免日期口径误解。
        ax.set_title(
            (
                f"预测基准日 {as_of_date.date()}｜新闻日期 {news_date.date()}｜"
                f"{market_cn} Gate={gate:.3f}｜展示 {n_display}/{n_available} 条"
            ),
            fontsize=11.5,
            pad=8,
        )
        ax.set_xlabel("新闻影响强度")
        ax.set_ylabel("新闻标题")
        ax.set_xlim(0, x_hi)
        ax.grid(axis="x", alpha=0.18, linewidth=0.6)
        ax.tick_params(axis="both", labelsize=9)

    fig.suptitle(
        f"典型事件窗口相关新闻影响强度对比（{market_cn}）",
        fontsize=16,
        y=0.995,
    )

    prediction_window = f"{window_dates[0].date()} — {window_dates[-1].date()}"
    news_window = f"{news_dates[0].date()} — {news_dates[-1].date()}"

    fig.text(
        0.5,
        0.971,
        (
            f"预测基准日窗口：{prediction_window}；新闻输入窗口：{news_window}；"
            f"窗口评分统一使用每日 Top-{score_top_k} 新闻，图中最多展示 {display_top_k} 条"
        ),
        ha="center",
        va="top",
        fontsize=9.8,
    )

    fig.text(
        0.5,
        0.008,
        (
            "注：新闻影响强度 = impact × market relevance；"
            "该指标用于刻画日级新闻聚合中的相对重要程度，不代表神经网络注意力权重。"
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )

    fig.tight_layout(rect=[0.03, 0.038, 1.0, 0.955])

    png = out_dir / "fig7_event_window_combined.png"
    pdf = out_dir / "fig7_event_window_combined.pdf"

    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    selected_window.to_csv(
        out_dir / "fig7_event_window_selected_window.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return png, pdf, panel_df


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--root", default=r"E:\Code\stockv2")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--step", type=int, default=1)
    p.add_argument(
        "--min_news",
        type=int,
        default=3,
        help="候选日最低新闻数，默认3。",
    )
    p.add_argument(
        "--score_top_k",
        type=int,
        default=3,
        help="窗口评分使用的高影响新闻条数，默认3。",
    )
    p.add_argument(
        "--display_top_k",
        type=int,
        default=5,
        help="图中每天最多展示的新闻条数，默认5。",
    )
    p.add_argument(
        "--min_peak_news",
        type=int,
        default=0,
        help="自动候选窗口可选约束：三日中至少一天达到该新闻数；0表示不限制。",
    )
    p.add_argument("--model_dir", default="outputs_ablation_no_cross_attention")
    p.add_argument("--out_dir", default="outputs_interpretability")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")

    # 新增：更合理的案例选择控制
    p.add_argument(
        "--market",
        choices=["auto", "SSE", "SZI"],
        default="auto",
        help="auto: 在两市场候选窗口中自动选择；SSE/SZI: 仅在指定市场内选择。",
    )
    p.add_argument(
        "--event_start_date",
        default=None,
        help="手工指定3日窗口起始交易日，如 2025-05-06。",
    )
    p.add_argument(
        "--event_center_date",
        default=None,
        help="手工指定3日窗口中间交易日，如 2025-05-07。",
    )

    args = p.parse_args()

    root = Path(args.root)
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

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

    pred = infer_loader(model, loaders["test"], device)
    pred = pred[pred["horizon_step"] == args.step].copy()
    pred["as_of_date"] = pd.to_datetime(pred["as_of_date"])

    raw = load_news_raw(root).items.copy()
    raw["date"] = pd.to_datetime(raw["date"])

    title_col = find_title_col(raw)
    market_trading_dates = build_market_trading_dates(prepared)

    daily_df, news_df = prepare_daily_case_table(
        pred=pred,
        raw=raw,
        title_col=title_col,
        market_trading_dates=market_trading_dates,
        min_news=args.min_news,
        score_top_k=args.score_top_k,
        display_top_k=args.display_top_k,
    )

    daily_df.to_csv(
        out_dir / "fig7_event_window_daily_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )

    candidate_windows = enumerate_candidate_windows(
        daily_df=daily_df,
        market_trading_dates=market_trading_dates,
        n_days=3,
        min_peak_news=args.min_peak_news,
    )

    candidate_windows.to_csv(
        out_dir / "fig7_event_window_candidate_windows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selected_window = resolve_selected_window(
        candidate_windows=candidate_windows,
        market_trading_dates=market_trading_dates,
        market=args.market,
        event_start_date=args.event_start_date,
        event_center_date=args.event_center_date,
        n_days=3,
    )

    png, pdf, panel_df = draw_combined_figure(
        selected_window=selected_window,
        daily_df=daily_df,
        news_df=news_df,
        out_dir=out_dir,
        score_top_k=args.score_top_k,
        display_top_k=args.display_top_k,
    )

    print("\n" + "=" * 80)
    print("FIG.7 | EVENT-WINDOW NEWS CASE ANALYSIS")
    print("=" * 80)
    print(f"Checkpoint: {config['checkpoint']}")
    print(f"H={args.horizon}, step={args.step}")
    print(f"Selection market: {args.market}")
    print(
        f"News rule: min_news={args.min_news}, "
        f"score_top_k={args.score_top_k}, "
        f"display_top_k={args.display_top_k}, "
        f"min_peak_news={args.min_peak_news}"
    )
    print("\n========== Selected window ==========")
    print(selected_window.to_string(index=False))

    print("\n========== Top candidate windows (Top 10) ==========")
    print(candidate_windows.head(10).to_string(index=False))

    print("\n========== Figure 7 news details ==========")
    print(
        panel_df[
            [
                "panel_id",
                "market",
                "as_of_date",
                "news_date",
                "gate",
                "rank",
                "title",
                "impact_relevance_score",
            ]
        ].to_string(index=False)
    )

    print("\n[SAVE]")
    for path in [
        out_dir / "fig7_event_window_combined.png",
        out_dir / "fig7_event_window_combined.pdf",
        out_dir / "fig7_event_window_selected_window.csv",
        out_dir / "fig7_event_window_case_news_details.csv",
        out_dir / "fig7_event_window_candidate_windows.csv",
        out_dir / "fig7_event_window_daily_scores.csv",
    ]:
        print(path)

    print("\nTips:")
    print("1) 若自动选择的窗口不够有金融解释力，可用 --market + --event_start_date 手工锁定事件窗口。")
    print("2) 建议正式稿优先选择具有清晰政策/监管/突发事件背景的连续3日窗口。")
    print("3) 若希望自动候选窗口至少有一天达到5条新闻，可追加 --min_peak_news 5。")
    print("4) 默认窗口评分仍按每日Top-3新闻，但图中最多展示5条；因此4/3/5会画成4/3/5。")


if __name__ == "__main__":
    main()
