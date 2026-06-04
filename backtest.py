#!/usr/bin/env python3
"""
backtest.py
Historical backtest of the pre-earnings put buying strategy.

Pricing:     Black-Scholes synthetic put prices (no historical options DB needed)
             Pre-earnings IV  = HV30 × IV_PREMIUM (1.40×)
             Post-earnings IV = pre_IV × (1 − IV_CRUSH)
Walk-fwd:    4 expanding-window folds, optimised MIN_SCORE per fold (max Sharpe)
Monte Carlo: 10,000 bootstrap paths from the empirical trade-return distribution
"""

import os, sys, warnings
from datetime import datetime
from math import log, sqrt, exp, erfc
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from earnings_bot import CFG, calc_runup, calc_atr, volume_surge, score_setup

try:
    from rich.console import Console
    _c = Console()
    def rprint(x=""):  _c.print(x)
    def hr():          rprint("[bold cyan]" + "═" * 65 + "[/bold cyan]")
    def section(t):    hr(); rprint(f"[bold cyan]  {t}[/bold cyan]"); hr()
except ImportError:
    import re
    def rprint(x=""):  print(re.sub(r"\[/?[\w\s#/,_]+\]", "", str(x)))
    def hr():          print("=" * 65)
    def section(t):    hr(); print(f"  {t}"); hr()


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

BT = {
    "START":           "2019-01-01",
    "END":             "2025-12-31",
    "EARNINGS_LIMIT":  40,        # yfinance limit per ticker (~10 yrs of quarterly)
    "DTE_ENTRY":       7,         # days-to-expiry when we buy the put (T-2)
    "DTE_EXIT":        4,         # DTE remaining at exit (T+1, 3 cal days later)
    "RISK_FREE":       0.045,     # annualised risk-free rate
    "IV_PREMIUM":      1.40,      # HV30 × factor → pre-earnings IV estimate
    "IV_CRUSH":        0.40,      # fraction of IV that evaporates after the print
    "OTM_PCT":         CFG["OTM_PCT"],
    "MIN_SCORE":       CFG["MIN_SCORE"],
    "STARTING_CAP":    10_000,
    "SPEND_PER_TRADE": 500,
    "N_MC":            10_000,
    "SCORE_SWEEP":     [50, 55, 60, 65, 70, 75, 80],
    "RESULT_CSV":      "backtest_results.csv",
    # Expanding-window folds: (train_start, train_end, test_start, test_end)
    "WF_FOLDS": [
        ("2019-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
        ("2019-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        ("2019-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        ("2019-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# BLACK-SCHOLES  (stdlib only — no scipy)
# ─────────────────────────────────────────────────────────────────────────────

def _ncdf(x: float) -> float:
    return 0.5 * erfc(-x / sqrt(2))


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European put price via Black-Scholes."""
    if T < 1e-6 or sigma < 1e-6:
        return max(K - S, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA  (fallback when network is unavailable)
# ─────────────────────────────────────────────────────────────────────────────

# Approximate annual volatility and start price for each ticker (2019 levels)
_SYNTH_VOL = {
    "NVDA": 0.62, "AMD": 0.55, "MSFT": 0.30, "GOOGL": 0.32, "META": 0.45,
    "AMZN": 0.40, "AAPL": 0.28, "AVGO": 0.38, "ORCL": 0.28, "ADBE": 0.38,
    "CRM":  0.40, "SNOW": 0.65, "PLTR": 0.70, "CRWD": 0.58, "PANW": 0.48,
    "NOW":  0.40, "INTU": 0.32, "ADSK": 0.34, "NET":  0.55, "MDB":  0.65,
    "DDOG": 0.60, "MRVL": 0.50, "SHOP": 0.60, "ZS":   0.55, "OKTA": 0.55,
    "TEAM": 0.42, "SPY":  0.17,
}
_SYNTH_S0 = {
    "NVDA": 52, "AMD": 25, "MSFT": 105, "GOOGL": 58, "META": 145,
    "AMZN": 87, "AAPL": 42, "AVGO": 290, "ORCL": 52, "ADBE": 255,
    "CRM": 155, "SNOW": 140, "PLTR": 10, "CRWD": 80, "PANW": 200,
    "NOW": 260, "INTU": 240, "ADSK": 195, "NET": 42, "MDB": 130,
    "DDOG": 80, "MRVL": 24, "SHOP": 38, "ZS": 80, "OKTA": 100,
    "TEAM": 100, "SPY": 280,
}


def _synthetic_data(tickers: list, seed: int = 42) -> tuple:
    """
    Generate realistic synthetic OHLCV (GBM) + quarterly earnings dates.
    Injects pre-earnings run-ups (35% probability) and post-earnings shocks
    drawn from a realistic distribution (mean -1%, std 7%).
    """
    rprint("  [yellow]SYNTHETIC MODE — network unavailable; using GBM simulation.[/yellow]")
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range(start=BT["START"], end=BT["END"])
    n     = len(dates)
    dt    = 1 / 252

    hist_out     = {}
    earnings_out = {}
    all_syms     = sorted(set(tickers + ["SPY"]))

    for sym in all_syms:
        sigma = _SYNTH_VOL.get(sym, 0.40)
        mu    = 0.15 if sym != "SPY" else 0.10
        S0    = _SYNTH_S0.get(sym, 100.0)

        z      = rng.standard_normal(n)
        daily  = np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * sqrt(dt) * z)

        # Build quarterly earnings calendar (Jan/Apr/Jul/Oct ± jitter)
        eq_dates = []
        if sym != "SPY":
            for yr in range(2019, 2026):
                for mo in [1, 4, 7, 10]:
                    jitter = int(rng.integers(-7, 8))
                    ed     = pd.Timestamp(f"{yr}-{mo:02d}-15") + pd.Timedelta(days=jitter)
                    biz    = pd.bdate_range(ed, periods=1)[0]
                    if pd.Timestamp(BT["START"]) <= biz <= pd.Timestamp(BT["END"]):
                        eq_dates.append(biz.to_pydatetime())
            earnings_out[sym] = sorted(eq_dates)

        date_idx = {d: i for i, d in enumerate(dates)}

        for ed in eq_dates:
            ei = date_idx.get(pd.Timestamp(ed).normalize())
            if ei is None:
                continue
            # Pre-earnings run-up (35% chance): uniform 5–15% over 20 days
            if rng.random() < 0.35:
                mag   = rng.uniform(0.05, 0.15)
                start = max(0, ei - 20)
                span  = ei - start
                if span > 0:
                    daily[start:ei] *= (1 + mag) ** (1 / span)
            # Post-earnings shock: N(-1%, 7%)
            if ei < n:
                daily[ei] *= 1 + rng.normal(-0.01, 0.07)
            # Occasional volume surge 2 days before (for vsurge signal)
            if ei >= 2 and rng.random() < 0.40:
                pass   # volume handled separately below

        prices = S0 * np.cumprod(daily)
        isd    = sigma * sqrt(dt) * 0.5

        # Volume: baseline + pre-earnings surge
        vol_base = rng.integers(2_000_000, 30_000_000, n).astype(float)
        for ed in eq_dates:
            ei = date_idx.get(pd.Timestamp(ed).normalize())
            if ei is not None and rng.random() < 0.45:
                lo, hi = max(0, ei - 5), min(n, ei + 1)
                vol_base[lo:hi] *= rng.uniform(1.8, 3.5)

        df = pd.DataFrame({
            "Close":  prices,
            "Open":   prices * np.exp(rng.normal(0, isd, n)),
            "High":   prices * np.exp(np.abs(rng.normal(0, isd, n))),
            "Low":    prices * np.exp(-np.abs(rng.normal(0, isd, n))),
            "Volume": vol_base,
        }, index=dates)
        df["High"] = df[["High", "Close", "Open"]].max(axis=1)
        df["Low"]  = df[["Low",  "Close", "Open"]].min(axis=1)
        hist_out[sym] = df

    n_ev = sum(len(v) for v in earnings_out.values())
    rprint(f"  Generated {len(hist_out)} synthetic series × {n} trading days.")
    rprint(f"  Quarterly earnings events: {n_ev} total.")
    return hist_out, earnings_out


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (live yfinance → synthetic fallback)
# ─────────────────────────────────────────────────────────────────────────────

def load_prices(tickers: list) -> dict:
    """Bulk-download OHLCV for all tickers + SPY. Returns {sym: DataFrame}."""
    syms = sorted(set(tickers + ["SPY"]))
    rprint(f"  Downloading {len(syms)} symbols ({BT['START']} → {BT['END']})...")
    try:
        raw = yf.download(
            syms,
            start=BT["START"],
            end=BT["END"],
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        out = {}
        for sym in syms:
            try:
                df = raw[sym].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
                df.index = df.index.tz_localize(None) if getattr(df.index, "tz", None) else df.index
                df = df.dropna(subset=["Close"])
                if not df.empty:
                    out[sym] = df
            except Exception:
                pass
        if len(out) >= max(3, len(syms) // 2):
            rprint(f"  Loaded {len(out)} symbols.")
            return out
    except Exception:
        pass
    # Fallback handled by caller — return empty so main() can detect
    return {}


def load_earnings(tickers: list) -> dict:
    """Fetch historical earnings dates per ticker. Returns {sym: [datetime]}."""
    rprint(f"  Fetching earnings dates for {len(tickers)} tickers...")
    out = {}
    now = datetime.now()
    for sym in tickers:
        try:
            ed = yf.Ticker(sym).get_earnings_dates(limit=BT["EARNINGS_LIMIT"])
            if ed is None or ed.empty:
                continue
            idx = ed.index.tz_localize(None) if getattr(ed.index, "tz", None) else ed.index
            past = sorted(d.to_pydatetime() for d in idx if d.to_pydatetime() < now)
            if past:
                out[sym] = past
        except Exception:
            pass
    total = sum(len(v) for v in out.values())
    rprint(f"  Got {total} historical events across {len(out)} tickers.")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def hv30(hist_slice: pd.DataFrame) -> float:
    """Annualised 30-day historical volatility from the tail of a price slice."""
    c = hist_slice["Close"].tail(32)
    if len(c) < 5:
        return 0.30
    return float(np.log(c / c.shift(1)).dropna().std() * sqrt(252))


def historical_mkt(spy_hist: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    """SPY regime at `as_of`, data strictly before that date (no leakage)."""
    c = spy_hist[spy_hist.index < as_of]["Close"].tail(60)
    if len(c) < 10:
        return {"above_ma20": True, "above_ma50": True, "spy_5d": 0.0, "regime": "bull"}
    cur  = float(c.iloc[-1])
    ma20 = float(c.tail(20).mean())
    ma50 = float(c.tail(50).mean()) if len(c) >= 50 else ma20
    chg5 = (cur / float(c.iloc[-6]) - 1) * 100 if len(c) >= 6 else 0.0
    return {
        "above_ma20": cur > ma20,
        "above_ma50": cur > ma50,
        "spy_5d":     round(chg5, 2),
        "regime":     "bull" if cur > ma20 else "bear",
    }


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE TRADE SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_event(
    sym: str,
    edate: datetime,
    hist: dict,
    spy_hist: pd.DataFrame,
    min_score: int,
    crush: float,
) -> Optional[dict]:
    """
    Simulate one earnings event.
    Entry: T-2 (two trading days before earnings date)
    Exit:  T+1 (first trading day after earnings date)
    Returns result dict or None if filtered / insufficient data.
    """
    ticker_hist = hist.get(sym)
    if ticker_hist is None or len(ticker_hist) < 30:
        return None

    edate_ts = pd.Timestamp(edate)
    pre  = ticker_hist[ticker_hist.index < edate_ts]
    post = ticker_hist[ticker_hist.index > edate_ts]
    if len(pre) < 25 or len(post) < 1:
        return None

    entry_date  = pre.index[-2]
    entry_price = float(pre["Close"].iloc[-2])
    exit_date   = post.index[0]
    exit_price  = float(post["Close"].iloc[0])

    # Slice ALL signals to entry date — zero look-ahead bias
    hist_slice = ticker_hist[ticker_hist.index <= entry_date]
    if len(hist_slice) < 25:
        return None

    runup  = calc_runup(hist_slice)
    atr    = calc_atr(hist_slice, CFG["ATR_PERIOD"])
    vsurge = volume_surge(hist_slice)
    mkt    = historical_mkt(spy_hist, entry_date)

    pre_iv  = hv30(hist_slice) * BT["IV_PREMIUM"]
    iv_dict = {"ok": True, "atm_iv": pre_iv * 100}
    score   = score_setup(runup, iv_dict, mkt, vsurge)

    if score < min_score:
        return None

    # ── Black-Scholes pricing ─────────────────────────────────────────────────
    strike    = round(entry_price * (1 - BT["OTM_PCT"]), 2)
    T_en      = BT["DTE_ENTRY"] / 365
    T_ex      = BT["DTE_EXIT"]  / 365
    r         = BT["RISK_FREE"]
    post_iv   = pre_iv * (1 - crush)

    entry_put = bs_put(entry_price, strike, T_en, r, pre_iv)
    exit_put  = bs_put(exit_price,  strike, T_ex, r, post_iv)

    if entry_put < 0.01:
        return None

    ret_pct   = (exit_put - entry_put) / entry_put * 100
    contracts = max(1, int(BT["SPEND_PER_TRADE"] / (entry_put * 100)))
    dollar_pl = (exit_put - entry_put) * contracts * 100
    stock_chg = (exit_price - entry_price) / entry_price * 100

    # Break-even: smallest downward stock move that makes exit_put >= entry_put
    breakeven = 0.0
    for delta in np.arange(0.001, 0.30, 0.001):
        if bs_put(entry_price * (1 - delta), strike, T_ex, r, post_iv) >= entry_put:
            breakeven = -round(delta * 100, 1)
            break

    return {
        "sym":           sym,
        "edate":         edate.date().isoformat(),
        "entry_date":    entry_date.date().isoformat(),
        "exit_date":     exit_date.date().isoformat(),
        "score":         score,
        "entry_price":   round(entry_price, 2),
        "exit_price":    round(exit_price, 2),
        "strike":        round(strike, 2),
        "pre_iv_%":      round(pre_iv * 100, 1),
        "post_iv_%":     round(post_iv * 100, 1),
        "entry_put":     round(entry_put, 3),
        "exit_put":      round(exit_put, 3),
        "contracts":     contracts,
        "stock_chg_%":   round(stock_chg, 2),
        "return_%":      round(ret_pct, 2),
        "dollar_pl":     round(dollar_pl, 2),
        "breakeven_%":   breakeven,
        "hv30_%":        round(hv30(hist_slice) * 100, 1),
        "vsurge":        round(vsurge, 2),
        "r5d":           round(runup.get("r5d", 0), 2),
        "r20d":          round(runup.get("r20d", 0), 2),
        "dist_52h":      round(runup.get("dist_52h", 0), 2),
        "atr":           round(atr, 2),
        "regime":        mkt["regime"],
        "win":           int(ret_pct > 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# RUN BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    earnings_map: dict,
    hist: dict,
    spy_hist: pd.DataFrame,
    min_score: Optional[int]   = None,
    crush:     Optional[float] = None,
    date_start: Optional[str]  = None,
    date_end:   Optional[str]  = None,
) -> pd.DataFrame:
    ms = min_score if min_score is not None else BT["MIN_SCORE"]
    cr = crush     if crush     is not None else BT["IV_CRUSH"]
    ds = pd.Timestamp(date_start) if date_start else pd.Timestamp("2000-01-01")
    de = pd.Timestamp(date_end)   if date_end   else pd.Timestamp("2100-01-01")

    rows = []
    for sym, dates in earnings_map.items():
        for edate in dates:
            ets = pd.Timestamp(edate)
            if not (ds <= ets <= de):
                continue
            r = simulate_event(sym, edate, hist, spy_hist, min_score=ms, crush=cr)
            if r:
                rows.append(r)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("edate").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_trades": 0}

    rets = df["return_%"].values
    wins = rets > 0
    dol  = df["dollar_pl"].values

    avg_win  = float(rets[wins].mean())   if wins.any()   else 0.0
    avg_loss = float(rets[~wins].mean())  if (~wins).any() else 0.0
    ev       = float(wins.mean() * avg_win + (1 - wins.mean()) * avg_loss)

    # Annualised Sharpe — assume ~100 trades/yr across full watchlist
    sharpe = float(rets.mean() / rets.std() * sqrt(100)) if rets.std() > 1e-6 else 0.0

    cum = np.cumsum(dol)
    dd  = float((np.maximum.accumulate(cum) - cum).max()) if len(cum) else 0.0

    mx = cc = 0
    for r in rets:
        cc = cc + 1 if r < 0 else 0
        mx = max(mx, cc)

    return {
        "n_trades":     int(len(df)),
        "win_rate_%":   round(float(wins.mean()) * 100, 1),
        "avg_ret_%":    round(float(rets.mean()), 2),
        "avg_win_%":    round(avg_win, 2),
        "avg_loss_%":   round(avg_loss, 2),
        "ev_%":         round(ev, 2),
        "sharpe":       round(sharpe, 2),
        "total_$":      round(float(dol.sum()), 2),
        "max_dd_$":     round(dd, 2),
        "max_consec_L": mx,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCORE BUCKET BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────

def score_bucket_table(df: pd.DataFrame) -> pd.DataFrame:
    bins   = [0, 65, 70, 75, 80, 101]
    labels = ["<65", "65-70", "70-75", "75-80", "80+"]
    df2 = df.copy()
    df2["bucket"] = pd.cut(df2["score"], bins=bins, labels=labels, right=False)
    rows = []
    for b in labels:
        sub = df2[df2["bucket"] == b]
        if sub.empty:
            continue
        m = metrics(sub)
        rows.append({"bucket": b, **m})
    return pd.DataFrame(rows)[["bucket", "n_trades", "win_rate_%", "avg_ret_%", "ev_%", "sharpe"]]


# ─────────────────────────────────────────────────────────────────────────────
# IV CRUSH SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────

def crush_sensitivity(earnings_map, hist, spy_hist) -> pd.DataFrame:
    rows = []
    for c in [0.25, 0.35, 0.40, 0.50, 0.60]:
        df = run_backtest(earnings_map, hist, spy_hist, crush=c)
        m  = metrics(df)
        m["crush_%"] = int(c * 100)
        rows.append(m)
    cols = ["crush_%", "n_trades", "win_rate_%", "avg_ret_%", "ev_%", "sharpe", "total_$"]
    return pd.DataFrame(rows)[cols]


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward(earnings_map, hist, spy_hist) -> pd.DataFrame:
    """
    4 expanding-window folds.
    Each fold:
      Train — sweep SCORE_SWEEP thresholds, keep the one with highest Sharpe.
      Test  — apply best threshold on unseen OOS year, report OOS metrics.
    """
    rows = []
    for (tr_s, tr_e, te_s, te_e) in BT["WF_FOLDS"]:
        # optimise on training window
        best_sc, best_sh = BT["MIN_SCORE"], -99.0
        for sc in BT["SCORE_SWEEP"]:
            df_tr = run_backtest(earnings_map, hist, spy_hist,
                                 min_score=sc, date_start=tr_s, date_end=tr_e)
            if len(df_tr) < 5:
                continue
            sh = metrics(df_tr)["sharpe"]
            if sh > best_sh:
                best_sh, best_sc = sh, sc

        # out-of-sample test
        df_te = run_backtest(earnings_map, hist, spy_hist,
                             min_score=best_sc, date_start=te_s, date_end=te_e)
        oos = metrics(df_te)

        rows.append({
            "test_year":    te_s[:4],
            "train_window": f"{tr_s[:4]}–{tr_e[:4]}",
            "opt_score":    best_sc,
            "n_trades":     oos.get("n_trades", 0),
            "win_%":        oos.get("win_rate_%", 0.0),
            "avg_ret_%":    oos.get("avg_ret_%", 0.0),
            "ev_%":         oos.get("ev_%", 0.0),
            "sharpe":       oos.get("sharpe", 0.0),
            "max_dd_$":     oos.get("max_dd_$", 0.0),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO  (10,000 paths)
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo(return_series: np.ndarray, n_sims: int = 10_000) -> dict:
    """
    Bootstrap 10,000 portfolio paths from the empirical return distribution.
    Each path draws len(trades) samples with replacement and compounds
    $SPEND_PER_TRADE per trade on $STARTING_CAP capital.
    """
    n     = len(return_series)
    cap   = BT["STARTING_CAP"]
    spend = BT["SPEND_PER_TRADE"]

    rng     = np.random.default_rng(42)
    samples = rng.choice(return_series, size=(n_sims, n), replace=True)

    dollar_pnl = samples / 100.0 * spend          # (n_sims, n)
    cum_pnl    = np.cumsum(dollar_pnl, axis=1)
    terminal   = cap + cum_pnl[:, -1]
    total_ret  = (terminal - cap) / cap * 100

    peak     = np.maximum.accumulate(cum_pnl, axis=1)
    drawdown = (peak - cum_pnl).max(axis=1)

    def pct(arr, p): return float(np.percentile(arr, p))

    return {
        "n_sims":       n_sims,
        "n_trades":     n,
        "p_ruin_%":     round(float((terminal < cap * 0.5).mean()) * 100, 1),
        "p_profit_%":   round(float((terminal > cap).mean()) * 100, 1),
        "term_mean_$":  round(float(terminal.mean()), 0),
        "ret_p05_%":    round(pct(total_ret,  5), 1),
        "ret_p25_%":    round(pct(total_ret, 25), 1),
        "ret_p50_%":    round(pct(total_ret, 50), 1),
        "ret_p75_%":    round(pct(total_ret, 75), 1),
        "ret_p95_%":    round(pct(total_ret, 95), 1),
        "dd_p50_$":     round(pct(drawdown, 50), 0),
        "dd_p95_$":     round(pct(drawdown, 95), 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    section("EARNINGS BOT  —  BACKTEST ENGINE")
    rprint(f"  Period         {BT['START']} → {BT['END']}")
    rprint(f"  Tickers        {len(CFG['TICKERS'])}")
    rprint(f"  Base score     ≥{BT['MIN_SCORE']}  |  "
           f"IV premium  {BT['IV_PREMIUM']}×  |  IV crush  {int(BT['IV_CRUSH']*100)}%")
    rprint(f"  Capital        ${BT['STARTING_CAP']:,}  |  "
           f"Spend/trade  ${BT['SPEND_PER_TRADE']}  |  MC paths  {BT['N_MC']:,}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    section("1 / 6  DATA LOADING")
    hist = load_prices(CFG["TICKERS"])
    if not hist:
        # Network unavailable — use synthetic GBM simulation
        hist, earnings = _synthetic_data(CFG["TICKERS"])
    else:
        earnings = load_earnings(CFG["TICKERS"])
    spy_hist = hist.get("SPY", pd.DataFrame())

    # ── 2. Base backtest ──────────────────────────────────────────────────────
    section("2 / 6  BASE BACKTEST  (full period, score ≥60, crush 40%)")
    df = run_backtest(earnings, hist, spy_hist)
    if df.empty:
        rprint("[red]  No trades simulated — check data availability.[/red]")
        return

    m = metrics(df)
    for k, v in m.items():
        rprint(f"  {k:<18}  {v}")

    rprint(f"\n  Last 10 simulated trades:")
    cols = ["sym", "edate", "score", "stock_chg_%", "return_%", "dollar_pl", "breakeven_%"]
    rprint(df[cols].tail(10).to_string(index=False))

    # ── 3. Score bucket breakdown ─────────────────────────────────────────────
    section("3 / 6  SCORE BUCKET BREAKDOWN")
    sb = score_bucket_table(df)
    rprint(sb.to_string(index=False))

    # ── 4. Walk-forward ───────────────────────────────────────────────────────
    section("4 / 6  WALK-FORWARD ANALYSIS  (4 expanding folds)")
    rprint("  Each fold: score threshold optimised on training window (max Sharpe),")
    rprint("             then applied to the unseen OOS year.\n")
    wf = walk_forward(earnings, hist, spy_hist)
    rprint(wf.to_string(index=False))

    # ── 5. IV crush sensitivity ───────────────────────────────────────────────
    section("5 / 6  IV CRUSH SENSITIVITY")
    rprint("  Assumptions: 25% / 35% / 40% / 50% / 60% IV evaporation post-print\n")
    cs = crush_sensitivity(earnings, hist, spy_hist)
    rprint(cs.to_string(index=False))

    # ── 6. Monte Carlo ────────────────────────────────────────────────────────
    section(f"6 / 6  MONTE CARLO  ({BT['N_MC']:,} paths)")
    mc = monte_carlo(df["return_%"].values, n_sims=BT["N_MC"])
    rprint(f"  Paths per simulation :  {mc['n_sims']:,}")
    rprint(f"  Trades drawn per path:  {mc['n_trades']}")
    rprint(f"  P(terminal < 50% cap):  {mc['p_ruin_%']}%   ← probability of ruin")
    rprint(f"  P(terminal > capital):  {mc['p_profit_%']}%  ← probability of net profit")
    rprint(f"  Expected terminal cap:  ${mc['term_mean_$']:,.0f}")
    rprint()
    rprint("  Total-return percentiles  (${:,} starting capital, ${} / trade):".format(
        BT["STARTING_CAP"], BT["SPEND_PER_TRADE"]))
    rprint(f"    5th  pct   {mc['ret_p05_%']:+.1f}%   ← worst-case band")
    rprint(f"    25th pct   {mc['ret_p25_%']:+.1f}%")
    rprint(f"    median     {mc['ret_p50_%']:+.1f}%")
    rprint(f"    75th pct   {mc['ret_p75_%']:+.1f}%")
    rprint(f"    95th pct   {mc['ret_p95_%']:+.1f}%   ← best-case band")
    rprint()
    rprint("  Max drawdown ($) percentiles:")
    rprint(f"    median     ${mc['dd_p50_$']:,.0f}")
    rprint(f"    95th pct   ${mc['dd_p95_$']:,.0f}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df.to_csv(BT["RESULT_CSV"], index=False)
    rprint(f"\n[green]  Saved {len(df)} trades → {BT['RESULT_CSV']}[/green]")
    rprint()


if __name__ == "__main__":
    main()
