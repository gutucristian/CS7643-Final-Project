# CS7643-Final-Project

## Project Structure

```
CS7643-Final-Project/
│
├── requirements.txt
├── README.md
│
├── SPY_ohlcv.csv                        # raw price data
├── SPY_ohlcv_with_indicators.csv        # enriched data with technical indicators
│
├── data/                                # data pipeline
│   ├── __init__.py
│   ├── build_indicators.py              # entry point — builds indicator CSV
│   ├── data_utils.py                    # indicator computation, CSV loading
│   ├── get_data.py                      # yfinance data fetching
│   ├── dataset.py                       # SPYDataset — sliding window (PyTorch)
│   └── labeling.py                      # direction_change_labels()
│
├── models/                              # model architectures
│   ├── __init__.py
│   ├── mlp/
│   │   ├── __init__.py
│   │   └── model.py                     # MLP(nn.Module)
│   ├── lstm/
│   │   ├── __init__.py
│   │   └── model.py                     # LSTMModel(nn.Module)
│   └── cnn/
│       ├── __init__.py
│       └── model.py                     # CNN1D(nn.Module)
│
├── training/                            # training utilities
│   ├── __init__.py
│   ├── trainer.py                       # Trainer — train / evaluate loop
│   └── losses.py                        # cross_entropy_loss + sharpe_loss
│
├── backtest/                            # portfolio simulation & metrics
│   ├── __init__.py
│   └── simulator.py                     # Backtester — run / metrics
│
├── configs/                             # hyperparameter YAML configs
│   └── mlp.yaml
│
├── checkpoints/                         # saved model weights
│   └── mlp_best.pt
│
├── results/                             # outputs from training + backtest runs
│   ├── {run_tag}_test_predictions.csv
│   ├── {run_tag}_backtest_results.txt
│   └── {run_tag}_backtest_plot.png
│
└── experiments/                         # per-model training scripts
    ├── mlp/
    │   └── test_predictions.csv         # saved after training (true + pred labels)
    ├── backtest_cnn.py                  # placeholder
    ├── backtest_lstm.py                 # placeholder
    ├── backtest_mlp.py                  # run backtest from saved predictions
    ├── count_labels.py                  # inspect label distribution
    ├── plot.py                          # plot portfolio curves from results
    ├── train_cnn.py
    ├── train_lstm.py
    └── train_mlp.py                     # train MLP, save checkpoint + predictions
```

## Sliding Window

Each model receives a **window of the last N days** of indicator values as input, rather than a single row. This gives the model context about how indicators are trending over time.

```
window_size = 20  (default)

Day:   1  2  3 ... 19  20  21  22 ...
                        ↑
              first prediction here
              input = [day1 ... day20]

Sample 1:  [day1,  day2,  ..., day20]  → predict label for day20
Sample 2:  [day2,  day3,  ..., day21]  → predict label for day21
...
```

- The first 19 days produce no prediction (window not yet full)
- **MLP:** window is flattened → `(window_size × num_features,)` vector
- **LSTM / CNN:** window is kept as `(window_size, num_features)` tensor

> `window_size` is a hyperparameter and can be tuned. Larger windows give more temporal context at the cost of a longer warm-up period and higher input dimensionality.

## Labeling Strategy

Labels are generated via the triple barrier method. 

## MLP Training Adjustments

### 1. Class-weighted loss

Raw OHLCV data produces imbalanced labels: Hold is roughly twice as frequent as Long or Short (direction changes only happen on reversal days). Without correction, the model collapses to predicting Hold 100% of the time, which maximizes accuracy by exploiting the majority class.

**Fix:** `CrossEntropyLoss` is initialized with per-class weights proportional to `1 / class_count`, computed from the training split. This equalizes the gradient contribution of each class so the model is penalized equally for missing a rare Long/Short and a common Hold.

Checkpoint selection also uses **lowest validation loss** rather than highest validation accuracy, since accuracy still rewards Hold-collapse when the val set is imbalanced.

### 2. Log-return features

Raw OHLCV prices are non-stationary — SPY trended from ~130 to ~450 over the 10-year dataset. Z-score normalization using train statistics does not fix this: test prices fall far outside the value range seen during training, so the model generalizes poorly.

**Fix:** Each price column (`Open`, `High`, `Low`, `Close`) and `Volume` are converted to log returns before normalization:

```
log_return_t = log(price_t / price_{t-1})
```

Log returns are stationary (mean ≈ 0, stable variance regardless of absolute price level), so train normalization statistics transfer cleanly to the val and test splits. This also removes the bull-market directional bias that caused Short predictions to be suppressed entirely.

---

## MLP Workflow

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train
Trains the MLP, saves the best checkpoint, and writes test predictions to `experiments/mlp/test_predictions.csv`.
```bash
python experiments/train_mlp.py --config configs/mlp.yaml
```

Labels are generated from `SPY_ohlcv.csv` on first run and cached to `SPY_ohlcv_labels_mlp.csv` for reuse.

### 3. Backtest
Loads saved predictions and runs the portfolio simulator in both `long_only` and `long_short` modes.
```bash
python experiments/backtest_mlp.py --config configs/mlp.yaml
```

### 4. Results
All outputs are saved to the `results/` directory:

| File | Description |
|---|---|
| `{run_tag}_test_predictions.csv` | Per-day true labels and model predictions over the test split |
| `{run_tag}_backtest_results.txt` | Full metrics report: prediction distribution, classification report, confusion matrix, and financial backtest metrics for each mode |
| `{run_tag}_backtest_plot.png` | Portfolio value curve vs buy-and-hold benchmark over the test period |

### Configuration
All hyperparameters live in `configs/mlp.yaml`:

| Section | Key | Description |
|---|---|---|
| `data` | `csv_path` | Path to raw OHLCV CSV |
| `data` | `feature_cols` | Feature columns fed to the model |
| `data` | `window_size` | Look-back window in days (default 20) |
| `data` | `train_frac` / `val_frac` | Chronological split fractions |
| `model` | `hidden_sizes` | MLP hidden layer widths (default [256, 128, 64]) |
| `model` | `dropout` | Dropout probability |
| `training` | `epochs` / `learning_rate` | Training hyperparameters |
| `training` | `loss` | `cross_entropy` or `sharpe` |
| `training` | `device` | `auto`, `cpu`, or `cuda` |
| `backtest` | `modes` | List of backtest modes to run |
| `checkpoint` | `dir` / `filename` | Where to save the best model |

---

## Backtest Metrics

### Classification metrics (per-class)

| Metric | Definition | What it tells you |
|---|---|---|
| **Precision** | Of all days predicted as class X, how many were actually X | How trustworthy is a signal when it fires |
| **Recall** | Of all days that were truly class X, how many did we catch | How often we miss a signal |
| **F1-score** | Harmonic mean of precision and recall | Single balanced score; useful when classes are imbalanced |
| **Support** | Number of true samples in that class | Context for how rare/common each class is in the test set |
| **Macro avg** | Unweighted average across classes | Treats each class equally regardless of frequency |
| **Weighted avg** | Average weighted by support | Reflects overall performance weighted by class frequency |

The **confusion matrix** shows the full breakdown of true vs predicted classes. Rows are the true label, columns are the predicted label. Diagonal entries are correct predictions; off-diagonal entries are mistakes (e.g., a true Hold predicted as Long means an unnecessary trade was triggered).

### Financial metrics (backtest)

| Metric | Definition | What it tells you |
|---|---|---|
| **Total return** | `(final_value - initial_capital) / initial_capital` | Raw percentage gain/loss over the test period |
| **Benchmark return** | Buy-and-hold return over the same period | Baseline to beat; computed as `(last_price - first_price) / first_price` |
| **Sharpe ratio** | `mean(daily_returns) / std(daily_returns) × √252` | Risk-adjusted return; >1 is generally considered good, >2 is strong |
| **Max drawdown** | Largest peak-to-trough decline in portfolio value | Worst-case loss from a high point; measures downside risk — reported separately for the strategy and the buy-and-hold benchmark |
| **Win rate** | Fraction of trading days with a positive daily return | How often the strategy makes money day-to-day |
| **Profit factor** | `sum(gains) / sum(losses)` | >1 means total gains exceed total losses; >1.5 is a useful threshold |

### Backtest modes

- **`long_only`**: enters a long position on a Long signal, exits to cash on a Short signal, does nothing on Hold. Never shorts the market.
- **`long_short`**: also takes short positions on Short signals. Always acts on a flip between Long and Short; Hold maintains the current position.

---

# Conda Set Up Instructions

1. Create the Conda environment with a Python version:

    `conda create -n cs7643-final-project python=3.11 -y`

2. Activate it: 

    `conda activate cs7643-final-project`

3. Install `requirements.txt` packages with pip:

    `pip install -r requirements.txt`