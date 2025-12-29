import sys
import os
from pathlib import Path

script_path = os.path.abspath(__file__)
root_dir = Path(script_path).parent.parent.parent
sys.path.insert(0, str(root_dir))

from collections import defaultdict
import json
import yaml
import numpy as np

from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class BootstrapRankingAnalyzer:
    """
    Bootstrap-анализ устойчивости консенсусных рейтингов с оценкой доверительных интервалов.
    """

    def __init__(self):
        self.root_dir = Path(__file__).resolve().parents[2]
        self.config = self._load_config()

        self.random_state = self.config["project"]["random_state"]
        self.n_bootstrap = self.config["project"]["bootstrap"]["n_iterations"]
        self.confidence_level = self.config["project"]["bootstrap"]["confidence_level"]

        np.random.seed(self.random_state)

        self.input_path = (
            self.root_dir / "results" / "stat_analysis" / "ranking.json"
        )
        self.output_path = (
            self.root_dir / "results" / "stat_analysis" / "boots.json"
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Инициализация BootstrapRankingAnalyzer")
        logger.info(f"Random state проекта: {self.random_state}")
        logger.info(f"Bootstrap итераций: {self.n_bootstrap}")
        logger.info(f"Уровень доверия: {self.confidence_level}")

    def _load_config(self):
        config_path = self.root_dir / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _confidence_interval(values, confidence_level):
        """
        Процентильный bootstrap доверительный интервал.
        """
        alpha = 1.0 - confidence_level
        lower = np.percentile(values, 100 * alpha / 2)
        upper = np.percentile(values, 100 * (1 - alpha / 2))
        return float(lower), float(upper)

    def run(self):
        logger.info("Запуск bootstrap-анализа устойчивости рейтингов")

        with open(self.input_path, "r", encoding="utf-8") as f:
            ranking = json.load(f)

        result = {
            "protocol": ranking.get("protocol", "unknown"),
            "bootstrap": {
                "n_iterations": self.n_bootstrap,
                "confidence_level": self.confidence_level
            },
            "classifiers": {}
        }

        for clf_name, biome_data in ranking["classifiers"].items():
            logger.info(f"Обработка классификатора: {clf_name}")
            result["classifiers"][clf_name] = {}

            for biome_name, biome_block in biome_data.items():
                logger.info(f"  Биом: {biome_name}")

                per_polygon = list(biome_block["per_polygon"].values())
                if not per_polygon:
                    logger.warning("  Нет данных по полигонам — пропуск")
                    continue

                methods = list(per_polygon[0]["borda"].keys())

                # распределения bootstrap
                borda_dist = defaultdict(list)
                median_dist = defaultdict(list)
                top1_count = defaultdict(int)
                top3_count = defaultdict(int)

                for i in range(self.n_bootstrap):
                    sample = np.random.choice(
                        per_polygon,
                        size=len(per_polygon),
                        replace=True
                    )

                    borda_acc = defaultdict(list)
                    median_acc = defaultdict(list)

                    for poly in sample:
                        for m, v in poly["borda"].items():
                            borda_acc[m].append(v)
                        for m, v in poly["median_rank"].items():
                            median_acc[m].append(v)

                    # агрегируем
                    for m in methods:
                        mean_borda = float(np.mean(borda_acc[m]))
                        mean_median = float(np.mean(median_acc[m]))

                        borda_dist[m].append(mean_borda)
                        median_dist[m].append(mean_median)

                    # вероятности топов считаем по борде
                    sorted_methods = sorted(
                        {m: np.mean(borda_acc[m]) for m in methods}.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    top1 = sorted_methods[0][0]
                    top3 = {m for m, _ in sorted_methods[:3]}

                    top1_count[top1] += 1
                    for m in top3:
                        top3_count[m] += 1

                    if (i + 1) % max(1, self.n_bootstrap // 10) == 0:
                        logger.info(f"    Bootstrap {i + 1}/{self.n_bootstrap}")

                biome_result = {}
                for m in methods:
                    borda_ci = self._confidence_interval(
                        borda_dist[m], self.confidence_level
                    )
                    median_ci = self._confidence_interval(
                        median_dist[m], self.confidence_level
                    )

                    biome_result[m] = {
                        "borda": {
                            "mean": float(np.mean(borda_dist[m])),
                            "std": float(np.std(borda_dist[m])),
                            "ci_lower": borda_ci[0],
                            "ci_upper": borda_ci[1]
                        },
                        "median_rank": {
                            "mean": float(np.mean(median_dist[m])),
                            "std": float(np.std(median_dist[m])),
                            "ci_lower": median_ci[0],
                            "ci_upper": median_ci[1]
                        },
                        "prob_top1": top1_count[m] / self.n_bootstrap,
                        "prob_top3": top3_count[m] / self.n_bootstrap
                    }

                result["classifiers"][clf_name][biome_name] = biome_result

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Bootstrap-результаты сохранены: {self.output_path}")


def main():
    BootstrapRankingAnalyzer().run()


if __name__ == "__main__":
    main()
