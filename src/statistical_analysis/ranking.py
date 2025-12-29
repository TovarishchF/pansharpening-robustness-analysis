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


class PansharpeningRankingAnalyzer:
    """
    ЭТАП 2.
    Консенсусное ранжирование методов паншарпенинга.
    """

    def __init__(self):
        self.root_dir = Path(__file__).resolve().parents[2]
        self.config = self._load_config()

        self.random_state = self.config["project"]["random_state"]
        np.random.seed(self.random_state)

        self.input_path = self.root_dir / "results" / "stat_analysis" / "decs.json"
        self.output_path = self.root_dir / "results" / "stat_analysis" / "ranking.json"

        logger.info("Инициализация RankingAnalyzer")
        logger.info(f"Random state проекта: {self.random_state}")

    def _load_config(self):
        with open(self.root_dir / "config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _rank(values):
        return {m: r + 1 for r, (m, _) in enumerate(sorted(values.items(), key=lambda x: x[1], reverse=True))}

    @staticmethod
    def _borda(ranks):
        scores = defaultdict(int)
        for r in ranks:
            n = len(r)
            for m, k in r.items():
                scores[m] += n - k + 1
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    @staticmethod
    def _median(ranks):
        acc = defaultdict(list)
        for r in ranks:
            for m, k in r.items():
                acc[m].append(k)
        return dict(sorted({m: float(np.median(v)) for m, v in acc.items()}.items(), key=lambda x: x[1]))

    def run(self):
        logger.info("Запуск ЭТАПА 2: ранжирование")
        with open(self.input_path, "r", encoding="utf-8") as f:
            decs = json.load(f)

        result = {
            "protocol": decs["protocol"],
            "classifiers": {}
        }

        for clf, biome_data in decs["classifiers"].items():
            result["classifiers"][clf] = {}
            for biome, methods in biome_data.items():
                per_poly = defaultdict(lambda: defaultdict(dict))

                for method, block in methods.items():
                    for poly, metrics in block["per_polygon_metrics"].items():
                        for metric, val in metrics.items():
                            per_poly[poly][metric][method] = val

                poly_ranks = {}
                for poly, metrics in per_poly.items():
                    ranks = [self._rank(v) for v in metrics.values()]
                    poly_ranks[poly] = {
                        "borda": self._borda(ranks),
                        "median_rank": self._median(ranks)
                    }

                result["classifiers"][clf][biome] = {
                    "per_polygon": poly_ranks
                }

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Результат сохранён: {self.output_path}")


def main():
    PansharpeningRankingAnalyzer().run()
    logger.info("Консеснусное ранжирование завершено.")


if __name__ == "__main__":
    main()

