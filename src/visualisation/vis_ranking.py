import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
    / "ranking.json"
)


def plot_rank_heatmap():
    setup_visual_style()
    data = load_json(DATA_PATH)

    rows = []

    # собираем ВСЕ median_rank со всех классификаторов и биомов
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
    
    # сортируем для красивой легенды heatmap
    df["poly_id"] = df["polygon"].str.extract(r"(\d+)").astype(int)
    order = df.sort_values("poly_id")["polygon"].unique()

    # КЛЮЧЕВОЙ МОМЕНТ:
    # агрегируем дубликаты polygon+method
    pivot = pd.pivot_table(
        df,
        index="polygon",
        columns="method",
        values="median_rank",
        aggfunc="mean"
    ).reindex(order)

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
    plt.title("Средний median rank по полигонам", fontsize=18)
    plt.yticks(fontsize=10)
    plt.ylabel("Полигоны", fontsize=14)
    plt.xticks(fontsize=10)
    plt.xlabel("Методы ПШ", fontsize=14)
    save_figure("heatmap_median_rank.png")



def plot_borda_scores():
    """
    Строит barplot:
    1) Borda score по биомам
    2) Median rank по биомам
    """
    setup_visual_style()
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

    # ===== Borda score =====
    plt.figure(figsize=(18, 10))
    sns.barplot(
        data=df,
        x="method",
        y="borda",
        hue="biome",
        errorbar=None
    )
    plt.title("Borda score по биомам")
    plt.ylabel("Borda score")
    plt.xlabel("Методы ПШ")
    plt.xticks(rotation=90)
    plt.legend(title="Биом")
    save_figure("borda_by_biome.png")

    # ===== Median rank =====
    plt.figure(figsize=(18, 10))
    sns.barplot(
        data=df,
        x="method",
        y="median_rank",
        hue="biome",
        errorbar=None
    )
    plt.title("Median rank по биомам")
    plt.ylabel("Median rank (меньше — лучше)")
    plt.xlabel("Методы ПШ")
    plt.xticks(rotation=90)
    plt.legend(title="Биом")
    save_figure("median_rank_by_biome.png")

if __name__ == "__main__":
    plot_rank_heatmap()
    plot_borda_scores()
