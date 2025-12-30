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


class DescriptiveStatisticsAnalyzer:
    """
    Дескриптивный анализ влияния методов паншарпенинга
    на метрики классификации с добавлением интегрального рейтинга.
    """

    def __init__(self):
        self.root_dir = Path(__file__).resolve().parents[2]
        self.config = self._load_config()

        self.random_state = self.config["project"]["random_state"]
        np.random.seed(self.random_state)

        self.cls_metrics_path = (
            self.root_dir / "results" / "classification_metrics.json"
        )
        self.pansharp_metrics_path = (
            self.root_dir / "results" / "pansharpening_metrics.json"
        )
        self.output_path = (
            self.root_dir / "results" / "stat_analysis" / "decs.json"
        )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Инициализация DescriptiveStatisticsAnalyzer")
        logger.info(f"Random state проекта: {self.random_state}")

    def _load_config(self):
        config_path = self.root_dir / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _descriptive_stats(values):
        values = np.asarray(values, dtype=float)
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "iqr": [
                float(np.percentile(values, 25)),
                float(np.percentile(values, 75))
            ],
            "min": float(np.min(values)),
            "max": float(np.max(values))
        }

    def _load_classification_metrics(self):
        logger.info("Загрузка метрик классификации")
        with open(self.cls_metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Загружено записей: {len(data)}")
        return data

    def _load_integral_ratings(self):
        logger.info("Загрузка интегральных рейтингов паншарпенинга")
        with open(self.pansharp_metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        ratings = defaultdict(lambda: defaultdict(dict))
        for biome, poly_data in data.get("integral_ratings", {}).items():
            for poly_id, methods in poly_data.items():
                for method, rating in methods.items():
                    ratings[biome][method][f"poly_{poly_id}"] = float(rating)

        return ratings

    def build(self, records, integral_ratings):
        grouped = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(list)
                )
            )
        )
        per_polygon = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(dict)
            )
        )

        for rec in records:
            clf = rec["classification_method"]
            biome = rec["biome"]
            method = rec["pansharpening_method"]
            poly = f"poly_{rec['poly_id']}"

            for k, v in rec.items():
                if k in {"classification_method", "biome", "pansharpening_method", "poly_id"}:
                    continue
                if isinstance(v, (int, float)):
                    grouped[clf][biome][method][k].append(v)
                    per_polygon[clf][biome][method].setdefault(poly, {})[k] = float(v)

        result = {
            "protocol": "within_polygon_blocked_validation",
            "classifiers": {}
        }

        for clf, biome_data in grouped.items():
            result["classifiers"][clf] = {}
            for biome, method_data in biome_data.items():
                result["classifiers"][clf][biome] = {}
                for method, metrics in method_data.items():
                    result["classifiers"][clf][biome][method] = {
                        "classification_metrics": {
                            m: self._descriptive_stats(v)
                            for m, v in metrics.items()
                        },
                        "per_polygon_metrics": per_polygon[clf][biome][method],
                        "integral_rating": integral_ratings.get(biome, {}).get(method, {})
                    }

        return result

    def run(self):
        logger.info("Запуск ЭТАПА 1: дескриптивная статистика")
        records = self._load_classification_metrics()
        integral = self._load_integral_ratings()
        result = self.build(records, integral)

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Результат сохранён: {self.output_path}")


def main():
    DescriptiveStatisticsAnalyzer().run()


if __name__ == "__main__":
    main()
