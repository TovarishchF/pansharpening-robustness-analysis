import sys
from pathlib import Path

# Добавляем корневую директорию проекта в sys.path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

import yaml
import geopandas as gpd
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import json
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter, defaultdict

from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@dataclass
class TrainTestSplit:
    """Результат разделения train/test для одного полигона"""
    poly_id: int
    biome_name: str
    train_indices: List[int]
    test_indices: List[int]
    train_classes: List[int]
    test_classes: List[int]
    class_distribution: Dict[int, Dict[str, int]]


class TTSplitter:
    """
    Разделение данных на train/test (blocked pixel-based split)
    """

    def __init__(self, config_path: str = None):
        self.root_dir = Path(__file__).parent.parent.parent
        if config_path is None:
            config_path = self.root_dir / 'config.yaml'
        self.config = self._load_config(config_path)
        self._setup_paths()
        self.random_state = self.config['project']['random_state']

        # Размер блока (в пикселях)
        self.block_size = self.config['cross_validation'].get('block_size', 16)

        logger.info(
            f"Инициализирован TTSplitter (blocked pixel-split, block_size={self.block_size})"
        )

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_paths(self):
        """Настройка путей"""
        self.polygons_path = self.root_dir / self.config['data']['class_polygons']
        self.output_dir = self.root_dir / self.config['data']['processed'] / "splits"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_classification_polygons(self) -> gpd.GeoDataFrame:
        """Загрузка полигонов классификации"""
        logger.info(f"Загрузка полигонов классификации: {self.polygons_path}")

        try:
            gdf = gpd.read_file(self.polygons_path, layer='class_learning_polygons')
        except:
            gdf = gpd.read_file(self.polygons_path)

        required_fields = ['class', 'poly_id', 'biome_name']
        for field in required_fields:
            if field not in gdf.columns:
                raise ValueError(f"Отсутствует поле {field} в данных")

        gdf['class'] = gdf['class'].astype(int)
        gdf['poly_id'] = gdf['poly_id'].astype(int)
        gdf['biome_name'] = gdf['biome_name'].astype(str)

        logger.info(f"Загружено {len(gdf)} полигонов")
        return gdf

    def _build_blocks(
        self,
        pixel_classes: np.ndarray,
        width: int,
        height: int
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Формирование блоков и их доминирующих классов
        """

        blocks = []
        block_labels = []

        pixel_classes_2d = pixel_classes.reshape(height, width)

        for y0 in range(0, height, self.block_size):
            for x0 in range(0, width, self.block_size):
                y1 = min(y0 + self.block_size, height)
                x1 = min(x0 + self.block_size, width)

                block_pixels = []
                block_classes = []

                for y in range(y0, y1):
                    for x in range(x0, x1):
                        idx = y * width + x
                        cls = pixel_classes_2d[y, x]

                        if cls <= 0:
                            continue

                        block_pixels.append(idx)
                        block_classes.append(cls)

                if not block_pixels:
                    continue

                class_counts = Counter(block_classes)
                dominant_class = class_counts.most_common(1)[0][0]

                blocks.append(block_pixels)
                block_labels.append(dominant_class)

        return blocks, block_labels

    def train_test_split_single_polygon(
        self,
        poly_id: int,
        pixel_classes: np.ndarray,
        raster_shape: Tuple[int, int],
        biome_name: str,
        test_size: float
    ) -> TrainTestSplit:
        """
        Blocked pixel-based train/test split для одного макрополигона
        """

        height, width = raster_shape
        blocks, block_labels = self._build_blocks(pixel_classes, width, height)

        if len(blocks) < 2:
            logger.warning(f"Полигон {poly_id}: недостаточно блоков для split")
            return None

        label_counts = Counter(block_labels)
        if min(label_counts.values()) < 2:
            logger.warning(
                f"Полигон {poly_id}: блоки не позволяют стратификацию"
            )
            return None

        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=self.random_state
        )

        block_indices = np.arange(len(blocks))
        train_blk_idx, test_blk_idx = next(
            sss.split(block_indices.reshape(-1, 1), block_labels)
        )

        train_indices = []
        test_indices = []
        train_classes = []
        test_classes = []

        for i in train_blk_idx:
            for px in blocks[i]:
                train_indices.append(px)
                train_classes.append(int(pixel_classes[px]))

        for i in test_blk_idx:
            for px in blocks[i]:
                test_indices.append(px)
                test_classes.append(int(pixel_classes[px]))

        if set(train_classes) != set(pixel_classes[pixel_classes > 0]):
            logger.warning(
                f"Полигон {poly_id}: не все классы попали в train"
            )
            return None

        class_distribution = {}
        for cls in sorted(set(pixel_classes[pixel_classes > 0])):
            cls = int(cls)
            class_distribution[cls] = {
                'train': train_classes.count(cls),
                'test': test_classes.count(cls),
                'total': int((pixel_classes == cls).sum())
            }

        logger.info(
            f"Полигон {poly_id} ({biome_name}): blocked pixel-split "
            f"train={len(train_indices)}, test={len(test_indices)}"
        )

        return TrainTestSplit(
            poly_id=poly_id,
            biome_name=biome_name,
            train_indices=train_indices,
            test_indices=test_indices,
            train_classes=train_classes,
            test_classes=test_classes,
            class_distribution=class_distribution
        )

    def split_all_polygons(self) -> Dict[str, Any]:
        """Разделение всех макрополигонов (blocked pixel-split)"""

        gdf = self.load_classification_polygons()

        results = {
            'metadata': {
                'split_type': 'blocked_pixel',
                'block_size': self.block_size,
                'test_size': self.config['cross_validation']['test_size'],
                'random_state': self.random_state
            },
            'train_test_splits': {},
            'statistics': {
                'skipped_polygons': []
            }
        }

        for poly_id in sorted(gdf['poly_id'].unique()):
            poly_df = gdf[gdf['poly_id'] == poly_id]
            biome_name = poly_df['biome_name'].iloc[0]

            # ВАЖНО:
            # Реальные pixel_classes и raster_shape
            # должны быть переданы из maxlike.py
            # Здесь используется только структура

            logger.info(
                f"Полигон {poly_id} ({biome_name}): ожидает pixel-данные для split"
            )

        logger.info(
            "Blocked pixel split инициализирован. "
            "Фактическое разделение выполняется на этапе классификации."
        )

        return results

    def save_splits(self, splits: Dict[str, Any]):
        """Сохранение разбиений в файл"""
        output_file = self.output_dir / self.config['cross_validation']['output_file']

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(splits, f, indent=2, ensure_ascii=False)

        logger.info(f"Разбиения сохранены в {output_file}")


def main():
    """Основная функция выполнения"""
    logger.info("Запуск blocked pixel-based train/test split")

    splitter = TTSplitter()
    splits = splitter.split_all_polygons()
    splitter.save_splits(splits)

    logger.info("Blocked pixel-split завершён")


if __name__ == "__main__":
    main()