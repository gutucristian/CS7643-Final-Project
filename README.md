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
└── experiments/                         # per-model training scripts
    ├── mlp/
    │   └── test_predictions.csv         # saved after training (true + pred labels)
    ├── backtest_cnn.py                  # placeholder
    ├── backtest_lstm.py                 # placeholder
    ├── backtest_mlp.py                  # run backtest from saved predictions
    ├── count_labels.py                  # inspect label distribution
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

Labels are generated from **close-to-close direction changes**, producing three classes: `Long (1)`, `Short (-1)`, and `Hold (0)`.

**Logic:**

Given a price series, compute the daily direction:
```
Price:      1    3    4    7    5    4
Direction:  —    up   up   up   dn   dn
Label:      L    H    H    S    H    (NaN at end)
```

- Direction **changes to up** → `Long` — enter or switch to long position
- Direction **stays up** → `Hold` — maintain current position
- Direction **changes to down** → `Short` — enter or switch to short position
- Direction **stays down** → `Hold` — maintain current position
- Last row → `NaN` (no future price available)

**Execution:** Labels are generated from `sign(Close_t - Close_{t-1})`. The predicted action on day T is executed at **day T+1 open price**, reflecting a realistic signal-after-close, execute-next-open workflow.

This labeling encodes the theoretically optimal strategy — capturing every directional move — and the model is trained to predict these direction changes from technical indicators observed at day T's close.

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

### 4. Inspect labels (optional)
```bash
python experiments/count_labels.py SPY_ohlcv_labels_mlp.csv
```

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

# Conda Set Up Instructions

1. Create the Conda environment with a Python version:

    `conda create -n cs7643-final-project python=3.11 -y`

2. Activate it: 

    `conda activate cs7643-final-project`

3. Install `requirements.txt` packages with pip:

    `pip install -r requirements.txt`