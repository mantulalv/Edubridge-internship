# Trading Data EDA (Synthetic BTCUSDT)

Internship assignment: data cleaning, feature engineering and EDA on `trades_orderbook_STUDENT_dirty.csv`.
Steps follow the instructor's PDF. All data is synthetic. Educational work, not financial advice.

## Folder structure
```
data/        raw CSV and cleaned outputs
notebooks/   eda.ipynb (the full workflow)
```

## How to run
```
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook notebooks/eda.ipynb
```
Run all cells from the top. The raw CSV must be inside `data/`.

## Problems found in the raw data
| Problem | Count |
|---|---|
| Duplicate rows (same trade_id) | 35 |
| Missing price / quantity / side / bid_qty_1 / ask_price_1 | 25 / 20 / 15 / 30 / 20 |
| Mixed timestamp formats (standard / DD/MM/YYYY / Unix) | 1917 / 78 / 40 |
| Side spellings (buy, BUY, Sell) and symbol spellings | 3 / 4 |
| Price stored as text with `$` or `,` | 50 |
| Negative prices / extreme prices (about 100x too big) | 6 / 8 |
| Zero quantity | 6 |
| Crossed order book (bid above ask) | 10 |
| Rows not in time order | yes |

## Cleaning result
2,035 rows in, 2,000 rows out, no missing values. The reason for every cleaning decision is written in the notebook next to its code.

## Outputs
- `data/trades_clean.csv` cleaned trades with spread, mid_price, order_imbalance, return, volatility
- `data/candles_1min.csv` 1-minute OHLCV candles with model features and target

## Notes
- Data is split by time (70% train, 30% test), never shuffled, and no feature uses future data.
- The data covers about 27 minutes, so the model-ready set is very small. Results are for practice only.
