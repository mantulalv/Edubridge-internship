# Assignment 3: Crypto signal app (Decision Tree / Random Forest / XGBoost)

`crypto.py` is the signal engine (given by the instructor). `app.py` is the Streamlit page built on top of it.
It trains three supervised models on Binance trade data and shows a BUY / HOLD / SELL signal.
**Paper money only. It never places real orders. Not financial advice.**

## Files
```
crypto.py          load trades -> bars -> features -> train -> live signal -> paper plan
app.py             Streamlit page (this assignment)
requirements.txt   libraries
.env.example       copy to .env (the real .env is never committed)
```

## Setup
```
python -m pip install -r requirements.txt --default-timeout=100
copy .env.example .env        (Windows)
```

## Run
```
python -m streamlit run app.py
```
(`python crypto.py` runs the terminal version and needs the trades CSV in the same folder.)

## Data
- Free daily trade files: data.binance.vision, Spot, daily, trades, BTCUSDT. Unzip and upload the CSV in the app.
- No file? Tick **Use random demo data**. It is random, so the models should not beat the baseline.

## How the app works
1. Sidebar: bar seconds, predict ahead (bars), BUY/SELL share, paper cash, risk %.
2. Upload a trades CSV (or demo data), shows the price line chart.
3. **Train models**: trains each model on the oldest 70% of bars and tests on the newest 30% (split by time, never shuffled). Shows accuracy next to the baseline ("always guess the most common label").
4. **Get live signal**: downloads the latest trades from Binance, builds the same features, shows Signal, Confidence and Price, then the paper-trade suggestion.

## Reading the result
If accuracy is not above the baseline, the model learned nothing useful and its signal should not be trusted.
