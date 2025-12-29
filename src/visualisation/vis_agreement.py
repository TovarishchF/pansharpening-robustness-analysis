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
    / "agr.json"
)


def plot_kendall_w():
    setup_visual_style()
    data = load_json(DATA_PATH)

    rows = []

    for clf, clf_data in data["classifiers"].items():
        for biome, stats in clf_data.items():
            rows.append({
                "classifier": clf,
                "biome": biome,
                "kendall_w": stats["kendall_w"]
            })

    df = pd.DataFrame(rows)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="classifier",
        y="kendall_w",
        hue="biome"
    )
    plt.title("Согласованность метрик (Kendall W)")
    plt.ylabel("Значение")
    plt.xlabel("Методы классификации")
    plt.legend(title="Биом")
    save_figure("kendall_w.png")


if __name__ == "__main__":
    plot_kendall_w()