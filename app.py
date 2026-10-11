from __future__ import annotations

from datetime import date, timedelta
import time
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_URL = "https://fapi.binance.com"
INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]

st.set_page_config(page_title="Binance Futures Backtester", page_icon="📈", layout="wide")
st.title("📈 Binance USDⓈ-M 合约回测平台")
st.caption("网页端策略研究工具 · 使用币安公开行情接口 · 不需要 API Key")

with st.sidebar:
    st.header("回测参数")
    symbol = st.text_input("交易对", value="BTCUSDT").strip().upper()
    interval = st.selectbox("K 线周期", INTERVALS, index=3)
    default_end = date.today()
    default_start = default_end - timedelta(days=90)
    start_date = st.date_input("开始日期（UTC）", value=default_start)
    end_date = st.date_input("结束日期（UTC，不含当天结束之后的数据）", value=default_end)
    st.divider()
    st.subheader("示例策略：双均线")
    fast = st.number_input("快均线周期", min_value=2, max_value=500, value=10, step=1)
    slow = st.number_input("慢均线周期", min_value=3, max_value=1000, value=30, step=1)
    st.caption("快均线高于慢均线做多，低于慢均线做空；信号下一根 K 线生效。")
    st.divider()
    st.subheader("账户与交易成本")
    initial_equity = st.number_input("初始资金（USDT）", min_value=100.0, max_value=100000000.0, value=10000.0, step=1000.0)
    leverage = st.number_input("名义杠杆倍数", min_value=1.0, max_value=20.0, value=1.0, step=0.5)
    fee_rate_pct = st.number_input("单边手续费（%）", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.3f")
    slippage_bps = st.number_input("单边滑点（基点）", min_value=0.0, max_value=100.0, value=2.0, step=0.5)
    run = st.button("▶ 运行回测", type="primary", use_container_width=True)

def request_json(endpoint: str, params: dict, retries: int = 4):
    url = BASE_URL + endpoint
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=25)
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(min(2 ** attempt, 8))
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"币安接口请求失败：{last_error}")

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_klines(symbol: str, interval: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    start_ms = int(pd.Timestamp(start_iso, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_iso, tz="UTC").timestamp() * 1000)
    cursor, rows = start_ms, []
    while cursor < end_ms:
        batch = request_json("/fapi/v1/klines", {
            "symbol": symbol, "interval": interval, "startTime": cursor,
            "endTime": end_ms - 1, "limit": 1500
        })
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < 1500:
            break
        time.sleep(0.12)
    columns = ["open_time", "open", "high", "low", "close", "volume", "close_time",
               "quote_volume", "trade_count", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df["open_time"] >= pd.Timestamp(start_ms, unit="ms", tz="UTC")) &
            (df["open_time"] < pd.Timestamp(end_ms, unit="ms", tz="UTC"))]
    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_funding(symbol: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    start_ms = int(pd.Timestamp(start_iso, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_iso, tz="UTC").timestamp() * 1000)
    cursor, rows = start_ms, []
    while cursor < end_ms:
        batch = request_json("/fapi/v1/fundingRate", {
            "symbol": symbol, "startTime": cursor, "endTime": end_ms - 1, "limit": 1000
        })
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1]["fundingTime"]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < 1000:
            break
        time.sleep(0.12)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["funding_time", "funding_rate"])
    df["funding_time"] = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    return df[["funding_time", "funding_rate"]].drop_duplicates("funding_time").sort_values("funding_time").reset_index(drop=True)

def backtest(klines: pd.DataFrame, funding: pd.DataFrame, fast: int, slow: int,
             initial_equity: float, leverage: float, fee_rate: float, slippage_bps: float):
    if fast >= slow:
        raise ValueError("快均线周期必须小于慢均线周期。")
    df = klines.copy().sort_values("open_time").reset_index(drop=True)
    if len(df) < slow + 5:
        raise ValueError(f"历史 K 线不足：当前 {len(df)} 根，至少需要 {slow + 5} 根。")
    df["fast_ma"] = df["close"].rolling(int(fast), min_periods=int(fast)).mean()
    df["slow_ma"] = df["close"].rolling(int(slow), min_periods=int(slow)).mean()
    raw = np.where(df["fast_ma"] > df["slow_ma"], 1.0,
          np.where(df["fast_ma"] < df["slow_ma"], -1.0, 0.0))
    df["signal"] = raw
    # Avoid look-ahead: today's completed bar signal applies to next bar.
    df["position"] = pd.Series(df["signal"]).shift(1).fillna(0.0)
    df["asset_return"] = df["close"].pct_change().fillna(0.0)
    prev = df["position"].shift(1).fillna(0.0)
    df["position_change"] = (df["position"] - prev).abs()
    # Turnover proxy: a reversal from -1 to +1 trades two units of notional.
    df["turnover"] = df["position_change"] * leverage
    df["fee_cost"] = df["turnover"] * fee_rate
    df["slippage_cost"] = df["turnover"] * (slippage_bps / 10000.0)
    df["funding_cost"] = 0.0
    if not funding.empty:
        close_ns = pd.to_datetime(df["close_time"], utc=True).astype("int64").to_numpy()
        f_ns = pd.to_datetime(funding["funding_time"], utc=True).astype("int64").to_numpy()
        indices = np.searchsorted(close_ns, f_ns, side="left")
        for i, idx in enumerate(indices):
            if idx >= len(df):
                continue
            pos = float(df.at[idx, "position"])
            rate = float(funding.iloc[i]["funding_rate"])
            # Positive rate means longs pay shorts; this is a simplified approximation.
            df.at[idx, "funding_cost"] += leverage * pos * rate
    df["gross_return"] = df["position"] * df["asset_return"] * leverage
    df["net_return"] = df["gross_return"] - df["fee_cost"] - df["slippage_cost"] - df["funding_cost"]
    df["equity"] = initial_equity * (1 + df["net_return"]).cumprod()
    df["peak"] = df["equity"].cummax()
    df["drawdown"] = df["equity"] / df["peak"] - 1
    df["daily_date"] = df["close_time"]
    daily = df.set_index("daily_date")["net_return"].resample("1D").apply(lambda x: (1 + x).prod() - 1).dropna()
    sharpe = np.nan
    if len(daily) > 1 and daily.std(ddof=1) > 0:
        sharpe = np.sqrt(365) * daily.mean() / daily.std(ddof=1)
    total_return = df["equity"].iloc[-1] / initial_equity - 1
    elapsed_days = max((df["close_time"].iloc[-1] - df["close_time"].iloc[0]).total_seconds() / 86400, 1/24)
    cagr = (df["equity"].iloc[-1] / initial_equity) ** (365 / elapsed_days) - 1 if df["equity"].iloc[-1] > 0 else np.nan
    events = df[df["position_change"] > 0].copy()
    summary = {
        "最终权益": float(df["equity"].iloc[-1]),
        "总收益率": float(total_return * 100),
        "年化收益率（估算）": float(cagr * 100),
        "最大回撤": float(df["drawdown"].min() * 100),
        "年化夏普（按日收益估算）": float(sharpe),
        "持仓变更次数": int(len(events)),
        "K线数量": int(len(df)),
        "资金费率事件数": int(len(funding)),
    }
    trades = events[["open_time", "close", "position", "position_change", "fee_cost", "slippage_cost", "funding_cost"]].copy()
    return df, trades, summary

def fmt(value, suffix=""):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.2f}{suffix}"

if run:
    if not symbol or not symbol.isalnum():
        st.error("请填写有效交易对，例如 BTCUSDT。")
        st.stop()
    if start_date >= end_date:
        st.error("开始日期必须早于结束日期。")
        st.stop()
    start_iso, end_iso = start_date.isoformat(), (end_date + timedelta(days=1)).isoformat()
    try:
        with st.spinner("正在从币安下载历史 K 线与资金费率，并执行回测……"):
            klines = fetch_klines(symbol, interval, start_iso, end_iso)
            funding = fetch_funding(symbol, start_iso, end_iso)
            if klines.empty:
                st.error("未获取到 K 线数据。请检查交易对、日期范围，或确认该合约在所选时期已经上线。")
                st.stop()
            df, trades, summary = backtest(
                klines, funding, int(fast), int(slow), float(initial_equity),
                float(leverage), float(fee_rate_pct) / 100.0, float(slippage_bps)
            )
        st.success(f"回测完成：{symbol} · {interval} · {len(df):,} 根 K 线")
        cols = st.columns(5)
        metrics = [
            ("最终权益", fmt(summary["最终权益"], " USDT")),
            ("总收益率", fmt(summary["总收益率"], "%")),
            ("最大回撤", fmt(summary["最大回撤"], "%")),
            ("年化夏普", fmt(summary["年化夏普（按日收益估算）"])),
            ("持仓变更", f'{summary["持仓变更次数"]:,} 次'),
        ]
        for col, (label, value) in zip(cols, metrics):
            col.metric(label, value)
        tab1, tab2, tab3 = st.tabs(["净值曲线", "价格与均线", "交易记录 / 数据"])
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["close_time"], y=df["equity"], name="策略权益", mode="lines"))
            fig.update_layout(height=480, xaxis_title="UTC 时间", yaxis_title="权益（USDT）", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            dd = go.Figure()
            dd.add_trace(go.Scatter(x=df["close_time"], y=df["drawdown"] * 100, name="回撤", fill="tozeroy"))
            dd.update_layout(height=260, xaxis_title="UTC 时间", yaxis_title="回撤（%）")
            st.plotly_chart(dd, use_container_width=True)
        with tab2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["close_time"], y=df["close"], name="收盘价", mode="lines"))
            fig.add_trace(go.Scatter(x=df["close_time"], y=df["fast_ma"], name=f"快均线 {int(fast)}", mode="lines"))
            fig.add_trace(go.Scatter(x=df["close_time"], y=df["slow_ma"], name=f"慢均线 {int(slow)}", mode="lines"))
            fig.update_layout(height=480, xaxis_title="UTC 时间", yaxis_title="价格", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        with tab3:
            st.write("### 汇总指标")
            st.dataframe(pd.DataFrame([summary]).T.rename(columns={0: "数值"}), use_container_width=True)
            st.write("### 持仓变更事件（不是逐笔成交撮合记录）")
            st.dataframe(trades.tail(200), use_container_width=True)
            st.download_button("下载完整权益与行情 CSV", df.to_csv(index=False).encode("utf-8-sig"), file_name=f"{symbol}_{interval}_equity.csv", mime="text/csv")
            st.download_button("下载持仓变更 CSV", trades.to_csv(index=False).encode("utf-8-sig"), file_name=f"{symbol}_{interval}_events.csv", mime="text/csv")
            st.download_button("下载原始 K 线 CSV", klines.to_csv(index=False).encode("utf-8-sig"), file_name=f"{symbol}_{interval}_klines.csv", mime="text/csv")
        with st.expander("回测假设与限制（请先阅读）"):
            st.markdown("""
- 本版本是**研究原型**，不是实盘交易系统。
- 示例策略为双均线多空切换；策略信号使用下一根 K 线生效，避免最直接的未来数据泄漏。
- 手续费和滑点以权益收益的近似扣减方式计算；成交价格、盘口深度、部分成交和延迟未建模。
- 资金费率按事件附近持仓做简化估算，未完整模拟交易所的精确结算快照规则。
- **没有模拟维持保证金、风险限额、强平价格、自动减仓（ADL）或爆仓后资金归零规则**。高杠杆结果可能严重失真。
- 回测结果对手续费、滑点、策略参数、数据完整性很敏感。请做样本外测试，不应将结果视为未来收益保证。
            """)
    except Exception as exc:
        st.exception(exc)
else:
    st.info("在左侧设置参数，然后点击「运行回测」。第一次运行会从币安公开接口下载数据。")
    st.markdown("""
### 当前原型包含
- 币安 USDⓈ-M 合约历史 K 线下载
- 历史资金费率下载（简化入账）
- 双均线多空策略与可调参数
- 手续费、滑点、杠杆和初始资金配置
- 净值曲线、回撤曲线、价格均线图
- CSV 导出

**无需 Binance API Key。** 公开历史行情接口通常不需要密钥，但部署平台必须能够访问 `fapi.binance.com`。
""")
