# file used to run a suite of experiements for lstm hyper paramater tuning
# brute for pairwise combinations of different hyper paramater values
# to run, use the following command
#   python experiments/run_lstm_experiments.py --suite configs/lstm_experiments.yaml

import argparse
import json
import os
import subprocess
import sys
import tempfile
from copy import deepcopy

import pandas as pd
import yaml


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_yaml(path):
    # load experiments configuration
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(path, payload):
    # helper function to save a yaml file
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def build_run_tag(feature_set_name, loss_name, label_variant_name):
    # generate a unique directory name for each combination of hyper paramaters

    run_tag = f"lstm_{feature_set_name}_{loss_name}_{label_variant_name}"

    return run_tag


def generate_config(
    base_cfg, feature_set_name, feature_cols, loss_name, 
    label_col, training_overrides,
): # Function to generate a configuration for the particular model run
    
    cfg = deepcopy(base_cfg)
    cfg["data"]["feature_cols"] = feature_cols
    cfg["training"]["loss"] = loss_name
    cfg["data"]["label_col"] = label_col


    # populate training params
    for key, value in training_overrides.items():
        cfg["training"][key] = value

    return cfg


def run_command(cmd, cwd):
   # helper to run commands in terminal
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(
        cmd,
        cwd = cwd,
        check = True
    )


def load_summary(run_tag):
    # helper function to load training/backtest results
    summary_path = os.path.join(
        PROJECT_ROOT, 
        "results", 
        "lstm_experiments", 
        run_tag,
        "backtest_summary.json"
    )

    with open(summary_path) as f:
        # return json results
        return json.load(f)


def main():
    # main function


    # setup args
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default = "configs/lstm_experiments.yaml")
    parser.add_argument("--base-config", default = None)
    parser.add_argument("--skip-train", action = "store_true")
    parser.add_argument("--skip-backtest", action = "store_true")
    args = parser.parse_args()


    # generate paths for saving results
    suite_path = os.path.join(PROJECT_ROOT, args.suite)
    suite = load_yaml(suite_path)

    # genertae configs
    base_config_rel = args.base_config or suite["base_config"]
    base_config_path = os.path.join(
        PROJECT_ROOT, base_config_rel
    )
    base_cfg = load_yaml(base_config_path)


    training_overrides = suite.get("training_overrides", {})

    default_label_variants = { # default ot regular label col
        "default": base_cfg["data"].get("label_col", "label")
    }
    label_variants = suite.get("label_variants", default_label_variants)

    # load other training params
    feature_sets = suite["feature_sets"]
    losses = suite["losses"]


    # generate results path
    results_dir = os.path.join(PROJECT_ROOT, "results", "lstm_experiments")
    os.makedirs(results_dir, exist_ok = True)

    rows = []
    # iterate through all possible combinations of hyper paramaters
    # created df row for every combination for summary statistics
    with tempfile.TemporaryDirectory(prefix = "lstm_configs_") as temp_config_dir:
        # different feature sets
        for feature_set_name, feature_cols in feature_sets.items():
            # different labeling srategies
            for label_variant_name, label_col in label_variants.items():
                # types of losses
                for loss_name in losses:
                    # get directory name for model variant

                    run_tag = build_run_tag(
                        feature_set_name, 
                        loss_name,
                        label_variant_name
                    )

                    config_path = os.path.join(temp_config_dir, f"{run_tag}.yaml")
                    
                    # generate the training config
                    cfg = generate_config(
                        base_cfg = base_cfg,
                        feature_set_name = feature_set_name,
                        feature_cols = feature_cols,
                        loss_name = loss_name,
                        label_col = label_col,
                        training_overrides = training_overrides
                    )

                    # save the yaml config for training
                    save_yaml(config_path, cfg)


                    if not args.skip_train:
                        # run the train lstm script in the terminal
                        command_args = [sys.executable, "experiments/train_lstm.py", "--config", config_path]
                        run_command(command_args, cwd = PROJECT_ROOT)

                    if not args.skip_backtest:
                        # run the backtests
                        command_args = [sys.executable, "experiments/backtest_lstm.py", "--config", config_path]
                        run_command(command_args, cwd = PROJECT_ROOT)

                    # get the training/backtest summary
                    summary = load_summary(run_tag)
                    
                    # gather results summary
                    cls, bt = summary["classification"], summary["backtest"]


                    experimetn_summary = {
                        "run_tag": run_tag,
                        "feature_set": feature_set_name,
                        "num_features": len(feature_cols),
                        "loss": loss_name,
                        "label_variant": label_variant_name,
                        "label_col": label_col,
                        "classification_accuracy": cls["accuracy"],
                        "classification_macro_f1": cls["macro_f1"],
                        "classification_weighted_f1": cls["weighted_f1"],
                        "total_return": bt["total_return"],
                        "sharpe": bt["sharpe_ratio"],
                        "max_drawdown": bt["max_drawdown"],
                        "trade_count": bt["trade_count"]
                    }

                    
                    rows.append(experimetn_summary)

    rows.sort( # order results on classification accuracy
        key = lambda row: (row["label_variant"], row["loss"], -row["classification_accuracy"], -row["total_return"])
    )

    csv_path = os.path.join(results_dir, "lstm_experiment_summary.csv")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index = False)

    print(f"\nSaved aggregate CSV -> {csv_path}")


if __name__ == "__main__":
    main()

    
