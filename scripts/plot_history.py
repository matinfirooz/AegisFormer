from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("history")
    p.add_argument("--output", default="training_curves.png")
    args = p.parse_args()

    df = pd.read_csv(args.history)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    if "train_accuracy" in df:
        ax.plot(df["epoch"], df["train_accuracy"], label="train accuracy")
    if "val_accuracy" in df:
        ax.plot(df["epoch"], df["val_accuracy"], label="validation accuracy")
    if "contrastive_loss" in df:
        ax.plot(df["epoch"], df["contrastive_loss"], label="contrastive loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
