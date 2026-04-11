import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import yaml
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

    # Загрузка цветовой схемы и меток из config.yaml
    config_path = root_dir / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        colors = config.get('colors', [])
        biome_labels = config.get('biome_labels')
        classifier_labels = config.get('classifier_labels')
        biome_order = ['urban', 'forest', 'agriculture']
        biome_colors_en = dict(zip(biome_order, colors))
        biome_colors_ru = {biome_labels[k]: v for k, v in biome_colors_en.items()}

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
    df['biome'] = df['biome'].map(biome_labels)
    df['classifier'] = df['classifier'].map(classifier_labels)

    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=df,
        x="classifier",
        y="kendall_w",
        width=0.4,
        hue="biome",
        palette=biome_colors_ru
    )
    plt.title("Согласованность метрик (Kendall's W)")
    plt.ylabel("Значение")
    plt.xlabel("Алгоритмы классификации")
    plt.legend(title="Тип покрытия земель")
    save_figure("kendall_w.png")


if __name__ == "__main__":
    plot_kendall_w()