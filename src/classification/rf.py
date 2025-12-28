import sys
import os
from pathlib import Path

script_path = os.path.abspath(__file__)
root_dir = Path(script_path).parent.parent.parent
sys.path.insert(0, str(root_dir))

import yaml
import rasterio
import numpy as np
import geopandas as gpd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from rasterio import features

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit

from src.utils.logger import get_logger, setup_logging
from src.classification.tt_split import TTSplitter

setup_logging()
logger = get_logger(__name__)


@dataclass
class ClassificationResult:
	biome_name: str
	poly_id: int
	pansharp_method: str
	classification_method: str
	classified_data: np.ndarray
	transform: any
	crs: str
	profile: dict


class RandomForestImageClassifier:
	"""
	Random Forest pixel-based classifier
	"""

	def __init__(self, config_path: str = None):
		if config_path is None:
			config_path = root_dir / "config.yaml"

		with open(config_path, "r", encoding="utf-8") as f:
			self.config = yaml.safe_load(f)

		self.splitter = TTSplitter(config_path)
		self.classification_method = "random_forest"

		self._setup_paths()

		logger.info("Инициализирован классификатор Random Forest")

	def _setup_paths(self):
		self.pansharp_path = root_dir / self.config["data"]["processed"] / "pansharpening"
		self.polygons_path = root_dir / self.config["data"]["class_polygons"]
		self.output_path = root_dir / self.config["data"]["processed"] / "classification"
		self.output_path.mkdir(parents=True, exist_ok=True)

	def _load_polygon_data(self, poly_id: int) -> gpd.GeoDataFrame:
		gdf = gpd.read_file(
			self.polygons_path,
			layer="classification_polygons"
		)
		gdf["class"] = gdf["class"].astype(int)
		gdf["poly_id"] = gdf["poly_id"].astype(int)

		poly = gdf[gdf["poly_id"] == poly_id]
		if poly.empty:
			raise ValueError(f"Нет полигонов для poly_id={poly_id}")

		return poly

	def _load_pansharpened_data(
		self,
		poly_id: int,
		biome_name: str,
		method: str
	) -> Optional[Tuple[np.ndarray, dict, any, str]]:

		names = [
			f"{poly_id:02d}_{method}.tif",
			f"{poly_id}_{method}.tif"
		]

		for name in names:
			path = self.pansharp_path / biome_name / method / name
			if path.exists():
				with rasterio.open(path) as src:
					return src.read(), src.profile, src.transform, src.crs

		return None

	def _rasterize_classes(
		self,
		gdf: gpd.GeoDataFrame,
		transform,
		shape: Tuple[int, int]
	) -> np.ndarray:
		return features.rasterize(
			[(geom, cls) for geom, cls in zip(gdf.geometry, gdf["class"])],
			out_shape=shape,
			transform=transform,
			fill=0,
			all_touched=True,
			dtype=np.int32
		)

	def _build_rf(self) -> RandomForestClassifier:
		params = self.config["classification"].get("random_forest", {})

		return RandomForestClassifier(
			n_estimators=params.get("n_estimators", 300),
			max_depth=params.get("max_depth", None),
			min_samples_split=params.get("min_samples_split", 2),
			min_samples_leaf=params.get("min_samples_leaf", 1),
			max_features=params.get("max_features", "sqrt"),
			n_jobs=params.get("n_jobs", -1),
			random_state=self.config["project"]["random_state"]
		)

	def process_single_polygon(
		self,
		poly_id: int,
		biome_name: str,
		method: str
	) -> Optional[ClassificationResult]:

		pansharp = self._load_pansharpened_data(poly_id, biome_name, method)
		if pansharp is None:
			return None

		data, profile, transform, crs = pansharp
		_, height, width = data.shape

		gdf = self._load_polygon_data(poly_id)
		class_raster = self._rasterize_classes(gdf, transform, (height, width))
		pixel_classes = class_raster.flatten()

		X = data.reshape(data.shape[0], -1).T
		valid = pixel_classes > 0

		# ================= SPLIT =================

		try:
			split = self.splitter.split(pixel_classes, (height, width))
			X_train = X[split.train_indices]
			y_train = np.array(split.train_classes)

		except ValueError as e:
			logger.warning(
				f"poly_id={poly_id}: split невозможен ({e}). "
				f"Используется TRAIN-ONLY режим"
			)
			X_train = X[valid]
			y_train = pixel_classes[valid]

		# ================= TRAIN =================

		if len(np.unique(y_train)) < 2:
			logger.warning(
				f"poly_id={poly_id}: недостаточно классов для RF"
			)
			return None

		rf = self._build_rf()
		rf.fit(X_train, y_train)

		# ================= PREDICT =================

		pred = rf.predict(X).reshape(height, width)

		return ClassificationResult(
			biome_name=biome_name,
			poly_id=poly_id,
			pansharp_method=method,
			classification_method=self.classification_method,
			classified_data=pred.astype(np.int32),
			transform=transform,
			crs=crs,
			profile=profile
		)

	def _export(self, result: ClassificationResult):
		out_dir = (
			self.output_path /
			result.classification_method /
			result.biome_name /
			result.pansharp_method
		)
		out_dir.mkdir(parents=True, exist_ok=True)

		out_path = out_dir / f"{result.poly_id:02d}_{result.classification_method}_{result.pansharp_method}.tif"

		profile = result.profile.copy()
		profile.update(
			count=1,
			dtype="int32",
			nodata=0,
			compress="DEFLATE"
		)

		with rasterio.open(out_path, "w", **profile) as dst:
			dst.write(result.classified_data, 1)

		logger.info(f"Экспортирован файл: {out_path.name}")

	def process_all_polygons(self):
		gdf = gpd.read_file(self.polygons_path, layer="classification_polygons")

		for poly_id in sorted(gdf["poly_id"].unique()):
			biome = gdf[gdf["poly_id"] == poly_id]["biome_name"].iloc[0]

			for method_dir in (self.pansharp_path / biome).iterdir():
				if not method_dir.is_dir():
					continue

				result = self.process_single_polygon(
					poly_id=int(poly_id),
					biome_name=biome,
					method=method_dir.name
				)
				if result:
					self._export(result)


def main():
	logger.info("Запуск классификации методом Random Forest")
	classifier = RandomForestImageClassifier()
	classifier.process_all_polygons()
	logger.info("Классификация Random Forest завершена")


if __name__ == "__main__":
	main()
