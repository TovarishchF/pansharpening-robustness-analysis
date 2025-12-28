import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

import yaml
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from collections import Counter

from sklearn.model_selection import StratifiedShuffleSplit

from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@dataclass
class TrainTestSplit:
    train_indices: List[int]
    test_indices: List[int]
    train_classes: List[int]
    test_classes: List[int]
    class_distribution: Dict[int, Dict[str, int]]


class TTSplitter:
    """
    Blocked pixel-based train/test split.
    Используется ТОЛЬКО как внутренний компонент классификатора.
    """

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = root_dir / 'config.yaml'

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.random_state = self.config['project']['random_state']
        self.test_size = self.config['cross_validation']['test_size']
        self.block_size = self.config['cross_validation'].get('block_size', 16)

        logger.info(
            f"Инициализирован TTSplitter "
            f"(blocked pixel split, block_size={self.block_size})"
        )

    def _build_blocks(
        self,
        pixel_classes: np.ndarray,
        width: int,
        height: int
    ) -> Tuple[List[List[int]], List[int]]:

        blocks = []
        block_labels = []

        cls_2d = pixel_classes.reshape(height, width)

        for y0 in range(0, height, self.block_size):
            for x0 in range(0, width, self.block_size):
                y1 = min(y0 + self.block_size, height)
                x1 = min(x0 + self.block_size, width)

                pixels = []
                labels = []

                for y in range(y0, y1):
                    for x in range(x0, x1):
                        idx = y * width + x
                        cls = int(cls_2d[y, x])
                        if cls <= 0:
                            continue
                        pixels.append(idx)
                        labels.append(cls)

                if not pixels:
                    continue

                dominant_class = Counter(labels).most_common(1)[0][0]
                blocks.append(pixels)
                block_labels.append(dominant_class)

        return blocks, block_labels

    def split(
        self,
        pixel_classes: np.ndarray,
        raster_shape: Tuple[int, int]
    ) -> TrainTestSplit:

        height, width = raster_shape
        pixel_classes = pixel_classes.astype(int)

        valid = pixel_classes > 0
        if valid.sum() == 0:
            raise ValueError("Нет валидных пикселей для split")

        blocks, block_labels = self._build_blocks(
            pixel_classes, width, height
        )

        if len(blocks) < 2:
            raise ValueError("Недостаточно блоков для split")

        if min(Counter(block_labels).values()) < 2:
            raise ValueError(
                "Невозможна стратификация: "
                "некоторые классы представлены < 2 блоками"
            )

        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.test_size,
            random_state=self.random_state
        )

        blk_idx = np.arange(len(blocks))
        train_blk, test_blk = next(
            sss.split(blk_idx.reshape(-1, 1), block_labels)
        )

        train_idx, test_idx = [], []
        train_cls, test_cls = [], []

        for i in train_blk:
            for px in blocks[i]:
                train_idx.append(px)
                train_cls.append(int(pixel_classes[px]))

        for i in test_blk:
            for px in blocks[i]:
                test_idx.append(px)
                test_cls.append(int(pixel_classes[px]))

        all_classes = set(pixel_classes[valid])
        if set(train_cls) != all_classes:
            raise ValueError("После split не все классы попали в train")

        class_distribution = {}
        for cls in sorted(all_classes):
            cls = int(cls)
            class_distribution[cls] = {
                'train': train_cls.count(cls),
                'test': test_cls.count(cls),
                'total': int((pixel_classes == cls).sum())
            }

        logger.info(
            f"Blocked pixel split выполнен: "
            f"train={len(train_idx)}, test={len(test_idx)}"
        )

        return TrainTestSplit(
            train_indices=train_idx,
            test_indices=test_idx,
            train_classes=train_cls,
            test_classes=test_cls,
            class_distribution=class_distribution
        )