import sys
import os
from pathlib import Path

script_path = os.path.abspath(__file__)
root_dir = Path(script_path).parent.parent.parent
sys.path.insert(0, str(root_dir))

import json
from collections import defaultdict
import numpy as np
import yaml

from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class KendallAgreementAnalyzer:
    """
    Анализ согласованности метрик классификации
    при ранжировании методов паншарпенинга
    с использованием коэффициента Kendall W
    """

    def __init__(self, config_path: Path | None = None):
        self.root_dir = Path(__file__).parent.parent.parent

        if config_path is None:
            config_path = self.root_dir / "config.yaml"

        self.config = self._load_config(config_path)

        # МОЖНО использовать любой из этих файлов
        self.metrics_file = (
            self.root_dir
            / "results"
            / "classification_metrics.json"
        )

        self.output_file = (
            self.root_dir
            / "results"
            / "stat_analysis"
            / "agr.json"
        )
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self.metrics_for_agreement = ["F1", "IOU", "PA", "UA", "Kappa"]

    @staticmethod
    def _load_config(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def kendall_w(rank_matrix: np.ndarray) -> float:
        """
        rank_matrix: shape (n_judges, n_objects)
        """
        m, n = rank_matrix.shape

        if m < 2 or n < 2:
            return np.nan

        R = np.sum(rank_matrix, axis=0)
        R_bar = np.mean(R)

        S = np.sum((R - R_bar) ** 2)

        W = 12 * S / (m**2 * (n**3 - n))

        return float(np.clip(W, 0.0, 1.0))

    def _load_detailed_results(self) -> list[dict]:
        """Поддержка list и dict форматов"""
        with open(self.metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            logger.info("Загружен список метрик (list)")
            return data

        if isinstance(data, dict) and "detailed_results" in data:
            logger.info("Загружен unified report")
            return data["detailed_results"]

        raise ValueError("Неизвестный формат файла метрик")

    def run(self):
        logger.info("Запуск анализа согласованности Kendall W")

        detailed = self._load_detailed_results()

        # classifier → biome → polygon → metric → {method: value}
        data = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        )

        for row in detailed:
            clf = row["classification_method"]
            biome = row["biome"]
            poly = row.get("poly_id", "unknown")
            method = row["pansharpening_method"]

            for metric in self.metrics_for_agreement:
                if metric in row:
                    data[clf][biome][poly][metric][method] = row[metric]

        results = {
            "protocol": self.config.get("protocol", "unknown"),
            "classifiers": {}
        }

        for clf, biome_data in data.items():
            results["classifiers"][clf] = {}

            for biome, poly_data in biome_data.items():
                logger.info(f"Обработка: {clf} | {biome}")

                metric_ranks = defaultdict(list)

                for poly_id, metric_data in poly_data.items():
                    for metric, values in metric_data.items():
                        if len(values) < 2:
                            continue

                        scores = np.array(list(values.values()))
                        ranks = np.argsort(np.argsort(-scores)) + 1
                        metric_ranks[metric].append(ranks)

                rank_matrix = []

                for metric, ranks_list in metric_ranks.items():
                    if len(ranks_list) < 2:
                        continue

                    mean_ranks = np.mean(np.vstack(ranks_list), axis=0)
                    rank_matrix.append(mean_ranks)

                if len(rank_matrix) < 2:
                    logger.warning(
                        f"Недостаточно данных для Kendall W: {clf} | {biome}"
                    )
                    continue

                rank_matrix = np.vstack(rank_matrix)

                W = self.kendall_w(rank_matrix)

                results["classifiers"][clf][biome] = {
                    "kendall_w": float(W),
                    "n_metrics": rank_matrix.shape[0],
                    "n_methods": rank_matrix.shape[1],
                    "n_polygons": max(len(v) for v in metric_ranks.values())
                }

                logger.info(
                    f"[{clf} | {biome}] Kendall W = {W:.3f}"
                )

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Файл согласованности сохранён: {self.output_file}")


def main():
    KendallAgreementAnalyzer().run()


if __name__ == "__main__":
    main()
