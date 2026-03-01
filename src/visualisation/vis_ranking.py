import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
    / "ranking.json"
)


def plot_rank_heatmap():
    """Тепловая карта медианных рангов по полигонам (с метками методов)"""
    setup_visual_style()
    data = load_json(DATA_PATH)

    # Загрузка меток методов
    config_path = root_dir / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        method_labels = config.get('method_labels')
    rows = []
    for clf_name, clf_data in data["classifiers"].items():
        for biome_name, biome_data in clf_data.items():
            for poly, poly_data in biome_data["per_polygon"].items():
                for method, rank in poly_data["median_rank"].items():
                    rows.append({
                        "classifier": clf_name,
                        "biome": biome_name,
                        "polygon": poly,
                        "method": method,
                        "median_rank": rank
                    })

    df = pd.DataFrame(rows)

    df["poly_id"] = df["polygon"].str.extract(r"(\d+)").astype(int)
    order = df.sort_values("poly_id")["polygon"].unique()

    pivot = pd.pivot_table(
        df,
        index="polygon",
        columns="method",
        values="median_rank",
        aggfunc="mean"
    ).reindex(order)

    # Переименовываем колонки в соответствии с method_labels
    pivot.rename(columns=method_labels, inplace=True)

    plt.figure(figsize=(18, 10))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="viridis_r",
        annot_kws={"size": 10},
        linewidths=0.5,
        linecolor="white"
    )
    plt.title("Медианный ранг по полигонам", fontsize=18)
    plt.yticks(fontsize=10)
    plt.ylabel("Полигоны", fontsize=14)
    plt.xticks(fontsize=10)
    plt.xlabel("Алгоритмы паншарпенинга", fontsize=14)
    save_figure("heatmap_median_rank.png")


def plot_borda_scores():
    setup_visual_style()

    # Загрузка цветовой схемы и меток
    config_path = root_dir / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        colors = config.get('colors', [])
        biome_labels = config.get('biome_labels')
        method_labels = config.get('method_labels')
        biome_order = ['urban', 'forest', 'agriculture']
        biome_colors_en = dict(zip(biome_order, colors))
        biome_colors_ru = {biome_labels[k]: v for k, v in biome_colors_en.items()}
    data = load_json(DATA_PATH)

    rows = []
    for clf_name, clf_data in data["classifiers"].items():
        for biome_name, biome_data in clf_data.items():
            for poly_data in biome_data["per_polygon"].values():
                for method in poly_data["borda"].keys():
                    rows.append({
                        "classifier": clf_name,
                        "biome": biome_name,
                        "method": method,
                        "borda": poly_data["borda"][method],
                        "median_rank": poly_data["median_rank"][method]
                    })

    df = pd.DataFrame(rows)
    df['biome'] = df['biome'].map(biome_labels)
    df['method'] = df['method'].map(method_labels)

    # ===== Borda score =====
    plt.figure(figsize=(18, 10))
    sns.barplot(
        data=df,
        x="method",
        y="borda",
        hue="biome",
        errorbar=None,
        palette=biome_colors_ru
    )
    plt.title("Оценка Борды по типам территории")
    plt.ylabel("Оценка Борды")
    plt.xlabel("Алгоритмы паншарпенинга")
    plt.xticks(rotation=60)
    plt.legend(title="Тип покрытия земель")
    save_figure("borda_by_biome.png")

    # ===== Median rank =====
    plt.figure(figsize=(18, 10))
    sns.barplot(
        data=df,
        x="method",
        y="median_rank",
        hue="biome",
        errorbar=None,
        palette=biome_colors_ru
    )
    plt.title("Медианный ранг по типам территории")
    plt.ylabel("Медианный ранг")
    plt.xlabel("Алгоритмы паншарпенинга")
    plt.xticks(rotation=60)
    plt.legend(title="Тип покрытия земель")
    save_figure("median_rank_by_biome.png")

if __name__ == "__main__":
    plot_rank_heatmap()
    plot_borda_scores()
