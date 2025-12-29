from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Корневая директория результатов с графиками
FIGURES_DIR = Path(__file__).parents[2] / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    """Загрузка JSON-файла"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_figure(filename: str):
    """Сохранение текущей фигуры"""
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()


def setup_visual_style():
    """Единый стиль визуализаций"""
    sns.set_theme(
        style="whitegrid",
        context="talk",
        palette="Set2"
    )

