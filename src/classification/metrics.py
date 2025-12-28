import sys
from pathlib import Path

# root
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

import yaml
import json
import numpy as np
import rasterio
import geopandas as gpd
from collections import defaultdict
from sklearn.metrics import (
    f1_score,
    jaccard_score,
    cohen_kappa_score,
    confusion_matrix
)

from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class MetricsCalculator:
    """
    Калькулятор метрик классификации.
    Метрики БЕРУТСЯ ТОЛЬКО из config.yaml.
    """

    def __init__(self, config_path: str | None = None):
        if config_path is None:
            config_path = root_dir / "config.yaml"

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.metrics_to_calculate = self.config["class_metrics"]["metrics_to_calculate"]
        self._validate_metrics()

        self.classification_dir = (
            root_dir
            / self.config["data"]["processed"]
            / "classification"
        )

        self.polygons_path = (
            root_dir
            / self.config["data"]["class_polygons"]
        )

        self.results_dir = (
            root_dir
            / "results"
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #

    def _validate_metrics(self):
        allowed = {"F1", "IOU", "UA", "PA", "Kappa"}
        extra = set(self.metrics_to_calculate) - allowed
        if extra:
            raise ValueError(
                f"В конфиге указаны неподдерживаемые метрики: {extra}. "
                f"Разрешены ТОЛЬКО {allowed}"
            )

    # ------------------------------------------------------------------ #

    def _load_reference(self, poly_id: int):
        gdf = gpd.read_file(
            self.polygons_path,
            layer="classification_polygons"
        )
        gdf["class"] = gdf["class"].astype(int)
        gdf = gdf[gdf["poly_id"] == poly_id]
        return gdf

    # ------------------------------------------------------------------ #

    def _rasterize_reference(self, gdf, transform, shape):
        from rasterio.features import rasterize

        return rasterize(
            [(geom, cls) for geom, cls in zip(gdf.geometry, gdf["class"])],
            out_shape=shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.int32
        )

    # ------------------------------------------------------------------ #

    def _compute_metrics(self, y_true, y_pred) -> dict:
        mask = y_true > 0
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        results = {}

        average = self.config["class_metrics"].get("average", "macro")

        if "F1" in self.metrics_to_calculate:
            results["F1"] = float(
                f1_score(y_true, y_pred, average=average, zero_division=0)
            )

        if "IOU" in self.metrics_to_calculate:
            results["IOU"] = float(
                jaccard_score(y_true, y_pred, average=average, zero_division=0)
            )

        if "Kappa" in self.metrics_to_calculate:
            results["Kappa"] = float(
                cohen_kappa_score(y_true, y_pred)
            )

        if "UA" in self.metrics_to_calculate or "PA" in self.metrics_to_calculate:
            cm = confusion_matrix(y_true, y_pred)
            with np.errstate(divide="ignore", invalid="ignore"):
                if "UA" in self.metrics_to_calculate:
                    ua = np.diag(cm) / cm.sum(axis=0)
                    results["UA"] = float(np.nanmean(ua))
                if "PA" in self.metrics_to_calculate:
                    pa = np.diag(cm) / cm.sum(axis=1)
                    results["PA"] = float(np.nanmean(pa))

        return results

    # ------------------------------------------------------------------ #

    def run(self):
        all_results = []

        for clf_method in self.classification_dir.iterdir():
            if not clf_method.is_dir():
                continue

            for biome_dir in clf_method.iterdir():
                if not biome_dir.is_dir():
                    continue

                for pansharp_dir in biome_dir.iterdir():
                    if not pansharp_dir.is_dir():
                        continue

                    for tif in pansharp_dir.glob("*.tif"):
                        poly_id = int(tif.stem.split("_")[0])

                        with rasterio.open(tif) as src:
                            pred = src.read(1)
                            transform = src.transform
                            height, width = pred.shape

                        gdf = self._load_reference(poly_id)
                        ref = self._rasterize_reference(
                            gdf, transform, (height, width)
                        )

                        metrics = self._compute_metrics(
                            ref.flatten(),
                            pred.flatten()
                        )

                        result = {
                            "classification_method": clf_method.name,
                            "biome": biome_dir.name,
                            "pansharpening_method": pansharp_dir.name,
                            "poly_id": poly_id,
                            **metrics
                        }

                        all_results.append(result)

                        logger.info(
                            f"Метрики посчитаны: "
                            f"{clf_method.name} | {biome_dir.name} | "
                            f"{pansharp_dir.name} | poly {poly_id}"
                        )

        out_path = self.results_dir / "classification_metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Метрики сохранены: {out_path}")


def main():
    MetricsCalculator().run()


if __name__ == "__main__":
    main()