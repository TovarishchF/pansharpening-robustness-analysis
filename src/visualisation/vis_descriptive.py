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
    / "decs.json"
)


def plot_boxplots():
    setup_visual_style()
    data = load_json(DATA_PATH)

    rows = []

    for clf, clf_data in data["classifiers"].items():
        for biome, biome_data in clf_data.items():
            for method, method_data in biome_data.items():
                metrics = method_data["classification_metrics"]
                for metric_name, stats in metrics.items():
                    rows.append({
                        "classifier": clf,
                        "biome": biome,
                        "method": method,
                        "metric": metric_name,
                        "value": stats["median"]
                    })

    df = pd.DataFrame(rows)

    for metric in df["metric"].unique():
        plt.figure(figsize=(18, 10))
        sns.boxplot(
            data=df[df["metric"] == metric],
            x="biome",
            y="value",
            hue="method"
        )
        plt.title(f"{metric} — распределение по биомам")
        plt.ylabel("Значение")
        plt.xlabel("Биом")
        plt.legend()
        plt.xticks(rotation=30)
        save_figure(f"boxplot_{metric}.png")

if __name__ == "__main__":
    plot_boxplots()