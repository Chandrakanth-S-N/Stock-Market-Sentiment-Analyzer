# Equity Sentiment Analyzer

A Streamlit application that predicts the short-term directional outlook
of a stock based on lexicon-based sentiment analysis of financial news,
supported by technical features and honest model evaluation.

---

## Setup

### 1. Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Configure API keys

Copy the example secrets file and add your keys:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` and uncomment the provider you want:

```toml
# GNEWS_KEY = "your-key"
# NEWSAPI_KEY = "your-key"
```

If no key is configured, the app uses Google News RSS (no key required).

### 4. Run the application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
app.py                        Streamlit entry point
config.py                     Constants, palette, settings
data/
  news_client.py              News provider interface + implementations
  price_client.py             yfinance wrapper
nlp/
  sentiment.py                VADER + finance lexicon scorer
  lexicon/
    finance_lexicon.tsv       Finance-specific sentiment terms
features/
  builder.py                  Feature engineering, time alignment
models/
  train.py                    Training, validation, calibration
  backtest.py                 Backtesting logic
ui/
  components.py               Plotly charts, metric cards
  styles.py                   CSS injection
tests/
  test_sentiment.py           Sentiment scorer tests
  test_alignment.py           Trading-day alignment tests
  test_leakage.py             Look-ahead leakage prevention tests
.streamlit/
  config.toml                 Streamlit theme
  secrets.toml.example        API key template
requirements.txt              Pinned dependencies
```

---

## Methodology

### Sentiment Analysis

- **Engine**: VADER (Valence Aware Dictionary and sEntiment Reasoner)
  extended with a finance-specific lexicon inspired by the
  Loughran-McDonald word lists.
- **Lexicon**: ~110 finance terms (e.g., "beat", "downgrade",
  "guidance cut", "record revenue") with calibrated valence scores
  on the [-4, 4] VADER scale.
- **Scoring**: Headlines receive 65% weight, descriptions 35%.
  Daily aggregation uses exponential recency weighting (half-life: 3 days).

### Prediction Model

- **Candidates**: L2-regularised logistic regression and gradient
  boosting (scikit-learn). The model with the higher out-of-sample
  ROC-AUC is selected.
- **Features**: Lagged daily sentiment (mean, std, article count,
  pos/neg ratio), sentiment momentum (3d, 7d), plus technical
  indicators (returns, RSI-14, SMA-5/20 crossover, 10-day volatility,
  volume change).
- **Validation**: Strictly time-ordered walk-forward (TimeSeriesSplit,
  5 folds). No random shuffling. No look-ahead leakage.
- **Calibration**: Platt scaling (CalibratedClassifierCV, sigmoid).

### Baselines

Every run compares the model against:
1. **Always-up**: predicts the majority class every day.
2. **Previous-day direction**: repeats yesterday's move.
3. **Price-only model**: gradient boosting on technical features only
   (no sentiment).

If the full model does not outperform baselines, the UI states this
clearly.

### Backtest

Long/flat strategy: go long when the model predicts "up"
(probability > 0.50), otherwise cash. Transaction costs: 10 bps per
round-trip. Reports cumulative return, buy-and-hold return, Sharpe
ratio, max drawdown, and hit rate.

### How accuracy numbers are computed

All metrics (accuracy, precision, recall, F1, ROC-AUC) are computed
on **out-of-sample predictions** aggregated across all walk-forward
folds.  Each fold trains on the chronologically earlier portion of the
data and tests on the subsequent block.  No training data ever appears
in the test set.  The confusion matrix is also computed on these
aggregated out-of-sample predictions.

To reproduce: run the app with a given ticker and lookback.  The
metrics displayed in the "Model Performance" tab are the walk-forward
out-of-sample results.  The code is deterministic (random_state=42).

---

## Limitations

- Lexicon-based sentiment captures surface-level polarity but cannot
  understand sarcasm, context, or nuance.
- Google News RSS may return fewer articles for less-covered tickers.
- Short lookback windows often yield insufficient data for training.
- The model identifies statistical patterns in recent data that may
  not persist out of sample.
- Transaction costs, slippage, and market impact are approximated.
- **This tool is for educational purposes only and does not constitute
  financial advice.**
