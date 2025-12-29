import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path

script_path = os.path.abspath(__file__)
root_dir = Path(script_path).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.utils import load_json, save_figure, setup_visual_style

DATA_PATH = (
    Path(__file__).parents[2]
    / "results"
    / "stat_analysis"
    / "boots.json"
)


def plot_bootstrap_ci():
    setup_visual_style()
    data = load_json(DATA_PATH)

    rows = []

    for clf_data in data["classifiers"].values():
        for biome_data in clf_data.values():
            for method, stats in biome_data.items():
                rows.append({
                    "method": method,
                    "mean": stats["median_rank"]["mean"],
                    "low": stats["median_rank"]["ci_lower"],
                    "high": stats["median_rank"]["ci_upper"]
                })

    df = (
        pd.DataFrame(rows)
        .groupby("method", as_index=False)
        .mean()
        .sort_values("mean")
    )

    plt.figure(figsize=(10, 6))

    plt.hlines(
        y=df["method"],
        xmin=df["low"],
        xmax=df["high"],
        linewidth=3,
        alpha=0.8
    )

    plt.plot(
        df["mean"],
        df["method"],
        "o",
        markersize=7
    )

    plt.xlabel("Медианный ранг")
    plt.ylabel("Методы")
    plt.title("Bootstrap доверительные интервалы (median rank)")

    # plt.gca().invert_yaxis()

    plt.grid(axis="x", linestyle="--", alpha=0.5)

    save_figure("bootstrap_ci.png")


def plot_topk_probability_by_biome():
    setup_visual_style()
    data = load_json(DATA_PATH)

    rows = []

    for clf_name, clf_data in data["classifiers"].items():
        for biome_name, biome_data in clf_data.items():
            for method, stats in biome_data.items():
                rows.append({
                    "classifier": clf_name,
                    "biome": biome_name,
                    "method": method,
                    "prob_top1": stats["prob_top1"],
                    "prob_top3": stats["prob_top3"]
                })

    df = pd.DataFrame(rows)

    # ===== Top-3 =====
    plt.figure(figsize=(14, 6))
    sns.barplot(
        data=df,
        x="method",
        y="prob_top3",
        errorbar = None,
        hue="biome"
    )
    plt.ylabel("Вероятность")
    plt.title("Вероятность попасть в Топ-3 (bootstrap)")
    plt.xticks(rotation=90)
    plt.legend(title="Биом")
    save_figure("prob_top3_by_biome.png")

    # ===== Top-1 =====
    plt.figure(figsize=(14, 6))
    sns.barplot(
        data=df,
        x="method",
        y="prob_top1",
        errorbar = None,
        hue="biome"
    )
    plt.ylabel("Вероятность")
    plt.title("Вероятность попасть в Топ-1 (bootstrap)")
    plt.xticks(rotation=90)
    plt.legend(title="Биом")
    save_figure("prob_top1_by_biome.png")



if __name__ == "__main__":
    plot_bootstrap_ci()
    plot_topk_probability_by_biome()
