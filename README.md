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
│   └── labeling.py                      # triple_barrier_labels()
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
└── experiments/                         # per-model training scripts
    ├── train_mlp.py
    ├── train_lstm.py
    └── train_cnn.py
```

# Conda Set Up Instructions

1. Create the Conda environment with a Python version:

    `conda create -n cs7643-final-project python=3.11 -y`

2. Activate it: 

    `conda activate cs7643-final-project`

3. Install `requirements.txt` packages with pip:

    `pip install -r requirements.txt`