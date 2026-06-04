# Earnings Short Bot

Pre-earnings put buying strategy. Scans for tech stocks with:
- Strong pre-earnings run-up across multiple timeframes
- Elevated IV (crowded long positioning)
- Volume surge into earnings
- ATR-based stop reference
- Ollama AI qualitative verdict
- Alpaca options execution

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API keys (or paste directly into CFG in earnings_bot.py)
export ALPACA_KEY="your_alpaca_api_key"
export ALPACA_SECRET="your_alpaca_secret_key"

# 3. Make sure Ollama is running locally with a model pulled
ollama pull llama3.1

# 4. Run
python earnings_bot.py
```

---

## Modes

| Mode | What it does |
|------|-------------|
| 1    | Scan watchlist, analyze candidates, execute puts |
| 2    | Show all open positions and P&L |
| 3    | Same as 1 but with custom tickers you enter |

---

## Key config values (earnings_bot.py -> CFG)

| Key | Default | What it controls |
|-----|---------|-----------------|
| PAPER | True | Paper vs live trading — always test on paper first |
| MAX_SPEND | 500 | Max $ premium per trade |
| MAX_POSITIONS | 3 | Cap on concurrent earnings trades |
| MIN_SCORE | 60 | Setup score threshold (0–100) |
| ATR_MULT | 1.5 | Stop reference = entry + (ATR x mult) |
| OTM_PCT | 0.02 | Put strike = price x (1 - 0.02), 2% OTM |
| DAYS_WINDOW | 7 | Scan for earnings within next N days |
| OLLAMA_MODEL | llama3.1 | Any model you have pulled locally |

---

## Scoring system (0-100)

- Run-up quality (40 pts): size of 5d and 20d move into earnings
- Near 52-week high (15 pts): proximity to peak = more sell-the-news risk
- IV level (20 pts): high IV = crowded expectations, more room to disappoint
- Market regime (15 pts): SPY trend and momentum
- Volume surge (10 pts): unusual buying pressure into the print

---

## IV crush warning

When ATM IV exceeds 70%, the bot will flag it. After earnings, IV typically
drops 30-50% regardless of direction. The stock needs to drop beyond the
priced-in expected move for puts to be profitable. Consider sizing smaller
on high-IV prints.

---

## Trade log

Every executed trade is appended to earnings_trades.csv. Use this to track
your real win rate, RR, and setup score correlation over time.

---

## Requirements

- Alpaca account with options trading enabled (paper or live)
- Ollama installed and running: https://ollama.com
- Python 3.10+
