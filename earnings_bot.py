#!/usr/bin/env python3
"""
earnings_bot.py
Pre-earnings put buying strategy.
ATR-based stops · IV analysis · Ollama AI verdict · Alpaca options execution
"""

import os, re, csv, logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import warnings; warnings.filterwarnings("ignore")

# Load .env from the same directory as this file (keeps secrets out of CFG)
try:
    from dotenv import load_dotenv as _ld
    _ld(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass  # python-dotenv optional — fall back to system env vars

import yfinance as yf
import pandas as pd
import numpy as np

try:
    import anthropic as _anthropic
    HAS_CLAUDE = True
except ImportError:
    HAS_CLAUDE = False

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, ContractType

try:
    from rich.console import Console
    from rich.panel import Panel
    _c = Console()
    def cprint(msg, **kw): _c.print(msg, **kw)
    def cinput(p=""): return _c.input(p)
except ImportError:
    def cprint(msg, **kw): print(re.sub(r"\[/?[\w\s#/,_]+\]", "", str(msg)))
    def cinput(p=""): return input(re.sub(r"\[/?[\w\s#/,_]+\]", "", str(p)))

logging.basicConfig(level=logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — edit here or set env vars: ALPACA_KEY, ALPACA_SECRET, OLLAMA_MODEL
# ─────────────────────────────────────────────────────────────────────────────

CFG = {
    # Alpaca credentials
    "ALPACA_KEY":    os.getenv("ALPACA_KEY",    ""),
    "ALPACA_SECRET": os.getenv("ALPACA_SECRET", ""),
    "PAPER":         True,          # always start on paper — flip to False for live

    # AI backend (Claude preferred, Ollama fallback)
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
    "CLAUDE_MODEL":      os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5"),
    "OLLAMA_MODEL":      os.getenv("OLLAMA_MODEL", "llama3.1"),

    # Scanner window
    "DAYS_WINDOW":   7,             # look for earnings within next N days
    "MIN_SCORE":     80,            # 0-100 threshold to qualify a trade

    # Risk management
    "MAX_SPEND":     500,           # max $ premium per trade
    "MAX_POSITIONS": 3,             # max concurrent earnings trades

    # ATR stop reference
    "ATR_PERIOD":    14,
    "ATR_MULT":      1.5,           # stop ref = entry + (ATR x mult) — for direct shorts

    # Put option targeting
    "OTM_PCT":       0.02,          # target strike = price x (1 - 0.02)
    "MIN_OI":        50,            # minimum open interest
    "MAX_SPREAD":    0.30,          # max bid/ask spread as % of ask

    # Logging
    "LOG_FILE":      "earnings_trades.csv",

    # Default watchlist
    "TICKERS": [
        "NVDA", "AMD",  "MSFT", "GOOGL", "META",  "AMZN", "AAPL",
        "AVGO", "ORCL", "ADBE", "CRM",   "SNOW",  "PLTR", "CRWD",
        "PANW", "NOW",  "INTU", "ADSK",  "NET",   "MDB",  "DDOG",
        "MRVL", "SHOP", "ZS",   "OKTA",  "TEAM",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def market_conditions() -> Dict[str, Any]:
    """SPY regime: trend, 5d change, above/below MA"""
    df   = yf.Ticker("SPY").history(period="60d")
    c    = df["Close"]
    cur  = float(c.iloc[-1])
    ma20 = float(c.tail(20).mean())
    ma50 = float(c.tail(50).mean())
    chg5 = (cur / float(c.iloc[-6]) - 1) * 100
    return {
        "above_ma20": cur > ma20,
        "above_ma50": cur > ma50,
        "spy_5d":     round(chg5, 2),
        "regime":     "bull" if cur > ma20 else "bear",
    }


def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    """14-period Average True Range via EWM"""
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    return round(float(tr.ewm(span=period, adjust=False).mean().iloc[-1]), 2)


def calc_runup(df: pd.DataFrame) -> Dict[str, float]:
    """Multi-timeframe run-up + proximity to 52-week high"""
    close = df["Close"]
    cur   = float(close.iloc[-1])
    h52   = float(df["High"].tail(252).max())
    out   = {"dist_52h": round((cur / h52 - 1) * 100, 2)}
    for d in [5, 10, 20]:
        if len(close) > d:
            out[f"r{d}d"] = round((cur / float(close.iloc[-(d + 1)]) - 1) * 100, 2)
        else:
            out[f"r{d}d"] = 0.0
    return out


def volume_surge(df: pd.DataFrame) -> float:
    """Ratio of 5-day avg volume vs 20-day avg — detects unusual buying pressure"""
    vol = df["Volume"]
    v5  = float(vol.tail(5).mean())
    v20 = float(vol.tail(20).mean())
    return round(v5 / v20, 2) if v20 > 0 else 1.0


def get_iv_and_chain(sym: str, earnings_dt: datetime) -> Dict[str, Any]:
    """
    Pull options chain from yfinance.
    Returns ATM put IV, expected move, and the puts DataFrame.
    Picks the first expiry after earnings within 21 days.
    """
    t = yf.Ticker(sym)
    exps = t.options
    if not exps:
        return {"ok": False}

    target_date = earnings_dt.date()
    chosen = None
    for exp in sorted(exps):
        exp_d = datetime.strptime(exp, "%Y-%m-%d").date()
        delta = (exp_d - target_date).days
        if 0 <= delta <= 21:
            chosen = exp
            break
    if not chosen:
        chosen = sorted(exps)[0]

    try:
        chain = t.option_chain(chosen)
        puts  = chain.puts.copy()
    except Exception:
        return {"ok": False}

    if puts.empty:
        return {"ok": False}

    # Current price
    try:
        cur = float(t.fast_info.last_price)
    except Exception:
        cur = float(t.history(period="1d")["Close"].iloc[-1])

    # ATM IV (average of 3 closest strikes)
    atm = puts.iloc[(puts["strike"] - cur).abs().argsort()[:3]]
    atm_iv = float(atm["impliedVolatility"].mean()) * 100

    # Expected move priced in: IV x sqrt(DTE/365)
    dte = max(1, (datetime.strptime(chosen, "%Y-%m-%d") - datetime.now()).days)
    exp_move = round(atm_iv * (dte / 365) ** 0.5, 1)

    return {
        "ok":       True,
        "expiry":   chosen,
        "atm_iv":   round(atm_iv, 1),
        "exp_move": exp_move,     # % move priced in by market
        "puts":     puts,
        "spot":     round(cur, 2),
    }


def score_setup(runup: dict, iv: dict, mkt: dict, vsurge: float) -> int:
    """
    Score 0-100. Weights (tuned from backtest data):
      Run-up (30 pts) | Near 52w high (25 pts) | IV sweet spot (25 pts)
      Market regime (10 pts) | Volume surge (10 pts)

    Key findings: dist_52h and IV range (50-70%) are the strongest predictors.
    The 65-69 score range was the worst-performing bucket — min_score is now 70.
    Runs >100% IV are penalised: crush destroys gains even on big drops.
    """
    s   = 0
    r5  = runup.get("r5d",   0)
    r20 = runup.get("r20d",  0)
    d52 = runup.get("dist_52h", -100)
    iv_ = iv.get("atm_iv", 0) if iv.get("ok") else 0

    # Run-up: 5d (0-15 pts) — favour moderate acceleration, not extremes
    s += 15 if r5 >= 8 else 10 if r5 >= 5 else 5 if r5 >= 3 else 0

    # Run-up: 20d (0-15 pts) — 10-25% sweet spot from data
    s += 15 if 10 <= r20 < 25 else 10 if r20 >= 5 else 5 if r20 >= 3 else 0

    # Proximity to 52w high (0-25 pts) — strongest predictor; avoid extended pull-backs
    s += 25 if d52 >= -3 else 15 if d52 >= -8 else 5 if d52 >= -15 else 0

    # IV: sweet spot 50-75% (0-25 pts); penalise extreme IV (>100%) for crush risk
    s += 25 if 50 <= iv_ < 75 else 15 if 75 <= iv_ < 100 else 10 if iv_ >= 100 else 10 if iv_ >= 35 else 5 if iv_ >= 20 else 0

    # Market conditions (0-10 pts)
    spy = mkt.get("spy_5d", 0)
    if not mkt.get("above_ma20"):
        s += 10 if spy < 0 else 7
    else:
        s += 7 if spy > 2 else 5

    # Volume surge: unusual buying pressure = crowded longs (0-10 pts)
    s += 10 if vsurge >= 2.0 else 7 if vsurge >= 1.5 else 3 if vsurge >= 1.2 else 0

    return min(s, 100)


# ─────────────────────────────────────────────────────────────────────────────
# EARNINGS DATE FETCH
# ─────────────────────────────────────────────────────────────────────────────

def get_earnings_date(sym: str) -> Optional[datetime]:
    """Multi-method earnings date fetch. yfinance can be inconsistent so we try both."""
    t   = yf.Ticker(sym)
    now = datetime.now()

    # Method 1: calendar dict/dataframe
    try:
        cal = t.calendar
        if isinstance(cal, dict):
            for dates in cal.get("Earnings Date", []):
                d = pd.Timestamp(dates).to_pydatetime().replace(tzinfo=None)
                if d > now:
                    return d
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            for idx in cal.index:
                if "earnings" in str(idx).lower():
                    v = cal.loc[idx]
                    v = v.iloc[0] if isinstance(v, pd.Series) else v
                    if pd.notna(v):
                        d = pd.Timestamp(v).to_pydatetime().replace(tzinfo=None)
                        if d > now:
                            return d
    except Exception:
        pass

    # Method 2: earnings_dates index
    try:
        ed = t.earnings_dates
        if ed is not None and not ed.empty:
            idx = ed.index.tz_localize(None) if ed.index.tz else ed.index
            future = ed[idx > now]
            if not future.empty:
                return idx[ed.index.get_loc(future.index[-1])].to_pydatetime()
    except Exception:
        pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# SCANNER
# ─────────────────────────────────────────────────────────────────────────────

def scan() -> List[Dict]:
    """Scan watchlist. Returns candidates above score threshold, sorted by score."""
    cprint("\n[bold cyan]Scanning for candidates...[/bold cyan]")

    mkt      = market_conditions()
    now      = datetime.now()
    deadline = now + timedelta(days=CFG["DAYS_WINDOW"])
    results  = []

    cprint(f"  Market: SPY 5d [{'+' if mkt['spy_5d'] > 0 else ''}{mkt['spy_5d']}%]  "
           f"regime: {mkt['regime']}  above MA20: {mkt['above_ma20']}\n")

    for sym in CFG["TICKERS"]:
        try:
            edate = get_earnings_date(sym)
            if not edate or not (now < edate <= deadline):
                continue

            df = yf.Ticker(sym).history(period="1y")
            if df.empty or len(df) < 25:
                continue

            runup   = calc_runup(df)
            atr     = calc_atr(df, CFG["ATR_PERIOD"])
            vsurge  = volume_surge(df)
            iv_data = get_iv_and_chain(sym, edate)
            sc      = score_setup(runup, iv_data, mkt, vsurge)
            cur     = float(df["Close"].iloc[-1])

            mark = "[green]✓[/green]" if sc >= CFG["MIN_SCORE"] else "[dim]✗[/dim]"
            cprint(f"  {mark} {sym:<6}  score={sc:3d}  "
                   f"earn={edate.strftime('%m/%d')}  "
                   f"5d={runup['r5d']:+.1f}%  "
                   f"iv={iv_data.get('atm_iv', 0):.0f}%")

            if sc >= CFG["MIN_SCORE"]:
                results.append({
                    "sym":    sym,
                    "edate":  edate,
                    "days":   (edate - now).days,
                    "price":  round(cur, 2),
                    "atr":    atr,
                    "stop":   round(cur + atr * CFG["ATR_MULT"], 2),
                    "runup":  runup,
                    "iv":     iv_data,
                    "mkt":    mkt,
                    "vsurge": vsurge,
                    "score":  sc,
                })
        except Exception as e:
            cprint(f"  [dim]✗ {sym}: {str(e)[:70]}[/dim]")

    return sorted(results, key=lambda x: x["score"], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# AI ANALYSIS  —  Claude primary · Ollama fallback
# ─────────────────────────────────────────────────────────────────────────────

# System prompt is stable across all calls — cached by the API after the first hit.
_SYSTEM_PROMPT = """\
You are a quantitative trading analyst specialising in pre-earnings options strategies.
Your job: evaluate whether a setup warrants buying put options before the earnings print.

Strategy thesis: stocks that ran up hard into earnings often "sell the news" — especially
when guided by IV crush, crowded longs, and proximity to 52-week highs.
Key risks: surprise beat, short squeeze, and IV crush overwhelming any price drop.

Respond ONLY in this exact format — no extra text, no preamble:
VERDICT: BUY PUTS | SKIP | WEAK
KEY REASON: <one sentence, specific to the numbers>
MAIN RISK: <one sentence, specific to the setup>
NOTE: <one stock-specific observation about this ticker's earnings history or sector>\
"""


def _build_user_message(c: dict) -> str:
    r  = c["runup"]
    iv = c["iv"]
    m  = c["mkt"]
    return f"""\
TICKER: {c['sym']}
Earnings: {c['edate'].strftime('%Y-%m-%d')} ({c['days']} days away)
Price: ${c['price']}  |  Score: {c['score']}/100

PRICE ACTION
  Run-up  5d: {r['r5d']:+.1f}%   10d: {r['r10d']:+.1f}%   20d: {r['r20d']:+.1f}%
  Distance from 52w high: {r['dist_52h']:+.1f}%
  Volume surge: {c['vsurge']:.1f}x 20d average

OPTIONS
  ATM put IV: {iv.get('atm_iv', 0):.0f}%
  Market-priced expected move: +/-{iv.get('exp_move', 0):.1f}%

TECHNICALS
  ATR(14): ${c['atr']}   |   ATR-stop ref: ${c['stop']}

MARKET
  SPY 5d: {m['spy_5d']:+.1f}%   |   Regime: {m['regime']}   |   Above MA20: {m['above_ma20']}

Evaluate this setup.\
"""


def _ai_via_claude(c: dict) -> str:
    """Call Claude API with cached system prompt. Returns formatted verdict string."""
    client = _anthropic.Anthropic(api_key=CFG["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=CFG["CLAUDE_MODEL"],
        max_tokens=256,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},   # cache across scans
            }
        ],
        messages=[{"role": "user", "content": _build_user_message(c)}],
    )
    return msg.content[0].text.strip()


def _ai_via_ollama(c: dict) -> str:
    """Fallback: local Ollama model. Same prompt, no caching."""
    prompt = _SYSTEM_PROMPT + "\n\n" + _build_user_message(c)
    resp = ollama.chat(
        model=CFG["OLLAMA_MODEL"],
        messages=[{"role": "user", "content": prompt}],
    )
    return resp["message"]["content"].strip()


def ai_analysis(c: dict) -> str:
    """
    Return AI verdict for a candidate setup.

    Priority:
      1. Claude API  (if ANTHROPIC_API_KEY is set)
      2. Ollama      (if installed and running locally)
      3. Plain text  (signals only, no LLM)
    """
    # ── Claude (preferred) ────────────────────────────────────────────────────
    if HAS_CLAUDE and CFG["ANTHROPIC_API_KEY"]:
        try:
            return _ai_via_claude(c)
        except Exception as e:
            cprint(f"  [yellow]Claude API error ({e}) — falling back to Ollama[/yellow]")

    # ── Ollama (fallback) ─────────────────────────────────────────────────────
    if HAS_OLLAMA:
        try:
            return _ai_via_ollama(c)
        except Exception as e:
            cprint(f"  [yellow]Ollama error ({e}) — no AI verdict available[/yellow]")

    # ── No LLM available ─────────────────────────────────────────────────────
    r  = c["runup"]
    iv = c["iv"]
    return (
        f"VERDICT: WEAK  (no AI backend — set ANTHROPIC_API_KEY or start Ollama)\n"
        f"KEY REASON: Score {c['score']}/100 | IV {iv.get('atm_iv',0):.0f}% | "
        f"dist_52h {r['dist_52h']:+.1f}%\n"
        f"MAIN RISK: Cannot assess without AI verdict.\n"
        f"NOTE: Run: export ANTHROPIC_API_KEY=your_key"
    )


# ─────────────────────────────────────────────────────────────────────────────
# PUT OPTION SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def select_put(c: dict) -> Optional[Dict]:
    """
    Pick the best put contract from yfinance chain.
    Target: 2% OTM, liquid, reasonable spread.
    Position size: max spend / (mid x 100).
    """
    iv_data = c["iv"]
    if not iv_data.get("ok") or iv_data.get("puts") is None:
        return None

    puts   = iv_data["puts"].copy()
    expiry = iv_data["expiry"]
    price  = c["price"]
    target = price * (1 - CFG["OTM_PCT"])

    # Apply liquidity filters, relax if nothing passes
    flt = puts[
        (puts["openInterest"] >= CFG["MIN_OI"]) &
        (puts["bid"] > 0) &
        (puts["ask"] > 0)
    ].copy()
    spread_mask = (flt["ask"] - flt["bid"]) / flt["ask"] <= CFG["MAX_SPREAD"]
    flt = flt[spread_mask]
    if flt.empty:
        flt = puts[(puts["bid"] > 0) & (puts["ask"] > 0)].copy()
    if flt.empty:
        return None

    # Best strike
    flt["diff"] = (flt["strike"] - target).abs()
    row = flt.nsmallest(1, "diff").iloc[0]

    bid = float(row["bid"])
    ask = float(row["ask"])
    mid = round((bid + ask) / 2, 2)
    iv_ = round(float(row["impliedVolatility"]) * 100, 1)
    if mid <= 0:
        return None

    contracts = max(1, int(CFG["MAX_SPEND"] / (mid * 100)))

    # TP: targeting ~2.5x premium (conservative for IV crush)
    # Note: for true 1:3 RR on options you need the put to 3x in value
    # IV crush means the stock needs to drop MORE than the expected move
    tp_price = round(mid * 2.5, 2)

    return {
        "expiry":      expiry,
        "strike":      float(row["strike"]),
        "bid":         bid,
        "ask":         ask,
        "mid":         mid,
        "iv":          iv_,
        "oi":          int(row.get("openInterest", 0)),
        "volume":      int(row["volume"]) if pd.notna(row.get("volume")) else 0,
        "contracts":   contracts,
        "total_cost":  round(contracts * mid * 100, 2),
        "tp":          tp_price,
        "crush_warn":  iv_ > 70,          # high IV = significant crush risk
        "contract_sym": str(row.get("contractSymbol", "")),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ALPACA EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def alpaca_client() -> TradingClient:
    if not CFG["ALPACA_KEY"]:
        raise ValueError("ALPACA_KEY not set. Add to CFG or set env var ALPACA_KEY.")
    return TradingClient(
        api_key    = CFG["ALPACA_KEY"],
        secret_key = CFG["ALPACA_SECRET"],
        paper      = CFG["PAPER"],
    )


def place_put_order(c: dict, put: dict) -> Optional[Dict]:
    """
    Search Alpaca for the matching put contract, then place a limit order.
    Uses GetOptionContractsRequest to find the OCC symbol reliably.
    """
    try:
        client = alpaca_client()
        sym    = c["sym"]

        # Search for the specific contract
        req = GetOptionContractsRequest(
            underlying_symbols = [sym],
            expiration_date    = put["expiry"],
            option_type        = ContractType.PUT,
            strike_price_gte   = str(round(put["strike"] - 1.5, 1)),
            strike_price_lte   = str(round(put["strike"] + 1.5, 1)),
        )
        found = client.get_option_contracts(req)

        if not found or not found.option_contracts:
            cprint(f"[red]No Alpaca contract found for {sym} ${put['strike']}P {put['expiry']}[/red]")
            cprint(f"[dim]Tip: options may not be approved on this account, or the contract "
                   f"doesn't exist in Alpaca's universe.[/dim]")
            return None

        contract_sym = found.option_contracts[0].symbol
        limit_px     = round(put["ask"] * 1.01, 2)   # 1% above ask for faster fill

        order = client.submit_order(LimitOrderRequest(
            symbol        = contract_sym,
            qty           = put["contracts"],
            side          = OrderSide.BUY,
            time_in_force = TimeInForce.DAY,
            limit_price   = limit_px,
        ))

        return {
            "id":     str(order.id),
            "symbol": contract_sym,
            "status": str(order.status),
            "qty":    put["contracts"],
            "limit":  limit_px,
            "ts":     datetime.now().isoformat(),
        }

    except Exception as e:
        cprint(f"[red]Order failed: {e}[/red]")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# POSITION MONITOR
# ─────────────────────────────────────────────────────────────────────────────

def show_positions():
    """Print all open positions with unrealized P&L"""
    try:
        positions = alpaca_client().get_all_positions()
        if not positions:
            cprint("  No open positions.")
            return

        cprint(f"\n  {'SYMBOL':<24} {'QTY':>5} {'ENTRY':>8} {'PRICE':>8} {'P&L':>10} {'%':>8}")
        cprint("  " + "─" * 65)
        for p in positions:
            pnl = float(p.unrealized_pl)
            pct = float(p.unrealized_plpc) * 100
            sgn = "+" if pnl >= 0 else ""
            col = "green" if pnl >= 0 else "red"
            cprint(
                f"  {p.symbol:<24} {str(p.qty):>5} "
                f"{float(p.avg_entry_price):>8.2f} "
                f"{float(p.current_price):>8.2f} "
                f"[{col}]{sgn}{pnl:>9.2f} {sgn}{pct:>6.1f}%[/{col}]"
            )
    except Exception as e:
        cprint(f"[red]Error fetching positions: {e}[/red]")


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def display_candidate(c: dict, put: Optional[dict], verdict: str):
    r  = c["runup"]
    iv = c["iv"]
    sc = c["score"]
    sc_col = "green" if sc >= 75 else "yellow" if sc >= 60 else "red"

    cprint(f"\n{'━' * 60}")
    cprint(
        f"[bold]{c['sym']}[/bold]  |  "
        f"Earnings: {c['edate'].strftime('%b %d')} ({c['days']}d)  |  "
        f"Price: ${c['price']}  |  "
        f"Score: [{sc_col}]{sc}/100[/{sc_col}]"
    )

    cprint(f"\n  Run-up     5d: {r['r5d']:+.1f}%  |  10d: {r['r10d']:+.1f}%  |  20d: {r['r20d']:+.1f}%")
    cprint(f"  52w high   {r['dist_52h']:+.1f}% from peak")
    cprint(f"  Vol surge  {c['vsurge']:.1f}x 20d average")
    cprint(f"  ATR (14)   ${c['atr']}  |  Stop ref: ${c['stop']}  (+{CFG['ATR_MULT']}x ATR)")

    if iv.get("ok"):
        cprint(f"  IV         ATM {iv['atm_iv']:.0f}%  |  Expected move priced in: +/-{iv['exp_move']:.1f}%")
    else:
        cprint(f"  IV         [dim]Options data unavailable[/dim]")

    if put:
        cprint(f"\n  [bold]Best put:[/bold]  ${put['strike']} strike  |  Exp: {put['expiry']}  |  IV: {put['iv']:.0f}%")
        cprint(f"  Bid/Ask: ${put['bid']} / ${put['ask']}  |  Mid: ${put['mid']}")
        cprint(f"  {put['contracts']} contract(s)  |  Max spend: ${put['total_cost']}  |  TP target: ${put['tp']}/contract")
        cprint(f"  OI: {put['oi']}  |  Volume: {put['volume']}")

        if put["crush_warn"]:
            cprint(
                f"\n  [yellow]IV crush warning: IV {put['iv']:.0f}% will likely collapse 30-50% after the print.[/yellow]\n"
                f"  [yellow]Stock needs to drop well beyond +/-{iv.get('exp_move', 0):.1f}% to overcome the decay.[/yellow]\n"
                f"  [yellow]Consider sizing smaller or using a wider OTM strike.[/yellow]"
            )
    else:
        cprint("  [yellow]No qualifying put found in options chain.[/yellow]")

    cprint(f"\n  [bold cyan]AI verdict:[/bold cyan]")
    for line in verdict.strip().splitlines():
        cprint(f"  {line}")


# ─────────────────────────────────────────────────────────────────────────────
# TRADE LOG
# ─────────────────────────────────────────────────────────────────────────────

def log_trade(c: dict, put: dict, order: Optional[dict], verdict: str):
    row = {
        "ts":           datetime.now().isoformat(),
        "ticker":       c["sym"],
        "earn_date":    c["edate"].date().isoformat(),
        "price":        c["price"],
        "score":        c["score"],
        "r5d":          c["runup"]["r5d"],
        "r20d":         c["runup"]["r20d"],
        "atr":          c["atr"],
        "atm_iv":       c["iv"].get("atm_iv", 0),
        "exp_move_pct": c["iv"].get("exp_move", 0),
        "vol_surge":    c["vsurge"],
        "strike":       put["strike"],
        "expiry":       put["expiry"],
        "put_mid":      put["mid"],
        "contracts":    put["contracts"],
        "total_spend":  put["total_cost"],
        "tp_target":    put["tp"],
        "crush_warn":   put["crush_warn"],
        "order_id":     order.get("id", "") if order else "NOT_PLACED",
        "order_status": order.get("status", "") if order else "",
        "ai_verdict":   verdict.replace("\n", " ")[:400],
    }
    exists = os.path.exists(CFG["LOG_FILE"])
    with open(CFG["LOG_FILE"], "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            w.writeheader()
        w.writerow(row)
    cprint(f"  [dim]Logged to {CFG['LOG_FILE']}[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    mode_label = "[yellow]PAPER[/yellow]" if CFG["PAPER"] else "[red bold]LIVE[/red bold]"
    if HAS_CLAUDE and CFG["ANTHROPIC_API_KEY"]:
        ai_label = f"Claude ({CFG['CLAUDE_MODEL']})"
    elif HAS_OLLAMA:
        ai_label = f"Ollama ({CFG['OLLAMA_MODEL']})"
    else:
        ai_label = "no AI"
    cprint(f"\n[bold cyan]Earnings Short Bot[/bold cyan]  |  "
           f"Mode: {mode_label}  |  "
           f"AI: {ai_label}  |  "
           f"Max spend: ${CFG['MAX_SPEND']}")

    cprint("\n[1] Scan + Trade   [2] Monitor positions   [3] Custom tickers")
    mode = cinput("Choose [1/2/3]: ").strip()

    if mode == "2":
        show_positions()
        return

    if mode == "3":
        raw = cinput("Tickers (comma-separated): ").upper()
        CFG["TICKERS"] = [t.strip() for t in raw.split(",") if t.strip()]

    # Capacity check
    try:
        open_count = len(alpaca_client().get_all_positions())
    except Exception:
        open_count = 0

    if open_count >= CFG["MAX_POSITIONS"]:
        cprint(f"\n[yellow]Already at max {CFG['MAX_POSITIONS']} positions. Monitor first.[/yellow]")
        show_positions()
        return

    # Scan
    candidates = scan()

    if not candidates:
        cprint("\n[yellow]No qualifying setups found in current window.[/yellow]")
        return

    cprint(f"\n[green]{len(candidates)} candidate(s) above score {CFG['MIN_SCORE']}.[/green]")

    for c in candidates:
        put     = select_put(c)
        verdict = ai_analysis(c)
        display_candidate(c, put, verdict)

        if not put:
            cinput("\n[dim]Press Enter to continue...[/dim]")
            continue

        action = cinput("\n[bold]Execute? (y / n / q to quit):[/bold] ").strip().lower()

        if action == "q":
            break
        elif action == "y":
            cprint(f"\nPlacing order for {c['sym']} puts...")
            order = place_put_order(c, put)
            if order:
                cprint(f"[green]Order submitted | id: {order['id']} | status: {order['status']}[/green]")
            log_trade(c, put, order, verdict)
            open_count += 1
            if open_count >= CFG["MAX_POSITIONS"]:
                cprint(f"[yellow]Max positions ({CFG['MAX_POSITIONS']}) reached.[/yellow]")
                break
        else:
            cprint("[dim]Skipped.[/dim]")

    cprint("\n[bold]Current positions:[/bold]")
    show_positions()
    cprint("\nDone.\n")


if __name__ == "__main__":
    main()
