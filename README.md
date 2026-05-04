# CS7643-Final-Project

To run CNN training pipeline:

`python experiments/train_cnn.py --config configs/cnn_cross_entropy_best.yaml`

To run CNN backtest pipeline and print statistics:

`python experiments/backtest_cnn.py --config configs/cnn_cross_entropy_best.yaml`

# Conda Set Up Instructions

1. Create the Conda environment with a Python version:

    `conda create -n cs7643-final-project python=3.11 -y`

2. Activate it: 

    `conda activate cs7643-final-project`

3. Install `requirements.txt` packages with pip:

    `pip install -r requirements.txt`