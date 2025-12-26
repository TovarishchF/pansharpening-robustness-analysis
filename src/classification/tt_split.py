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
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import pandas as pd
from collections import Counter

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


@dataclass
class CrossValidationSplit:
    """Результат кросс-валидации для одного полигона"""
    poly_id: int
    biome_name: str
    folds: List[Dict[str, List[int]]]
    class_distribution: Dict[int, Dict[str, int]]


class TTSplitter:
    """
    Разделение данных на train/test и кросс-валидацию
    """

    def __init__(self, config_path: str = None):
        self.root_dir = Path(__file__).parent.parent.parent
        if config_path is None:
            config_path = self.root_dir / 'config.yaml'
        self.config = self._load_config(config_path)
        self._setup_paths()
        self.random_state = self.config['project']['random_state']
        logger.info("Инициализирован TTSplitter")

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

        # Загружаем с указанием слоя и создаем fid из индекса
        try:
            # Пробуем загрузить слой 'class_learning_polygons'
            gdf = gpd.read_file(self.polygons_path, layer='class_learning_polygons')
        except:
            # Если слоя нет, загружаем дефолтный
            gdf = gpd.read_file(self.polygons_path)

        # Создаем поле fid из индекса
        gdf = gdf.reset_index(drop=False)
        gdf = gdf.rename(columns={'index': 'fid'})

        # Проверка необходимых полей
        required_fields = ['class', 'poly_id', 'biome_name']
        for field in required_fields:
            if field not in gdf.columns:
                raise ValueError(f"Отсутствует поле {field} в данных")

        # Преобразование типов
        gdf['fid'] = gdf['fid'].astype(int)
        gdf['class'] = gdf['class'].astype(int)
        gdf['poly_id'] = gdf['poly_id'].astype(int)
        gdf['biome_name'] = gdf['biome_name'].astype(str)

        logger.info(f"Загружено {len(gdf)} полигонов")
        logger.info(f"Уникальные poly_id: {sorted(gdf['poly_id'].unique())}")
        logger.info(f"Уникальные биомы: {sorted(gdf['biome_name'].unique())}")
        logger.info(f"Уникальные классы: {sorted(gdf['class'].unique())}")

        # Вывод детальной информации о распределении
        for biome in sorted(gdf['biome_name'].unique()):
            biome_df = gdf[gdf['biome_name'] == biome]
            logger.info(f"Биом '{biome}': {len(biome_df)} полигонов, "
                        f"{len(biome_df['poly_id'].unique())} макрополигонов")

        for class_id in sorted(gdf['class'].unique()):
            class_df = gdf[gdf['class'] == class_id]
            logger.info(f"Класс {class_id}: {len(class_df)} полигонов, "
                        f"{len(class_df['poly_id'].unique())} макрополигонов")

        return gdf

    def check_class_distribution(self, classes: List[int]) -> Tuple[bool, str]:
        """Проверка, достаточно ли объектов в каждом классе для стратификации"""
        class_counts = Counter(classes)

        # Проверяем минимальное количество объектов в классе
        min_count = min(class_counts.values())

        if min_count < 2:
            problematic_classes = [cls for cls, count in class_counts.items() if count < 2]
            return False, f"Классы {problematic_classes} имеют менее 2 объектов (минимальное: {min_count})"

        return True, f"Все классы имеют как минимум 2 объекта"

    def check_cv_distribution(self, classes: List[int], n_folds: int = 3) -> Tuple[bool, str]:
        """Проверка, достаточно ли объектов в каждом классе для кросс-валидации"""
        class_counts = Counter(classes)

        # Проверяем минимальное количество объектов в классе
        min_count = min(class_counts.values())

        if min_count < n_folds:
            problematic_classes = [cls for cls, count in class_counts.items() if count < n_folds]
            return False, f"Классы {problematic_classes} имеют менее {n_folds} объектов (минимальное: {min_count})"

        return True, f"Все классы имеют как минимум {n_folds} объекта(ов)"

    def get_class_distribution(self, indices: List[int], classes: List[int]) -> Dict[int, Dict[str, int]]:
        """Получение распределения классов"""
        distribution = {}
        unique_classes = np.unique(classes)

        for class_id in unique_classes:
            class_id_int = int(class_id)  # Преобразуем в int
            class_indices = [idx for idx, cls in zip(indices, classes) if cls == class_id]
            distribution[class_id_int] = {
                'count': len(class_indices),
                'indices': class_indices
            }

        return distribution

    def train_test_split_single_polygon(self, poly_id: int,
                                        indices: List[int],
                                        classes: List[int],
                                        biome_name: str,
                                        test_size: float = 0.3) -> TrainTestSplit:
        """Разделение на train/test для одного полигона с стратификацией по классам"""

        if len(indices) < 2:
            logger.warning(f"Полигон {poly_id} имеет недостаточно данных для разделения")
            return None

        # Проверяем распределение классов
        can_split, message = self.check_class_distribution(classes)
        if not can_split:
            logger.warning(f"Полигон {poly_id} ({biome_name}): {message}. Стратифицированный split невозможен.")

            # Используем простое случайное разделение без стратификации
            from sklearn.model_selection import train_test_split

            X = np.array(indices).reshape(-1, 1)
            y = np.array(classes)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )

            train_indices = X_train.flatten().tolist()
            test_indices = X_test.flatten().tolist()
            train_classes = y_train.tolist()
            test_classes = y_test.tolist()

            logger.info(f"Полигон {poly_id}: Использовано простое случайное разделение (без стратификации)")
        else:
            # Используем StratifiedShuffleSplit для стратификации по классам
            X = np.array(indices).reshape(-1, 1)
            y = np.array(classes)

            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=self.random_state
            )

            for train_idx, test_idx in sss.split(X, y):
                train_indices = X[train_idx].flatten().tolist()
                test_indices = X[test_idx].flatten().tolist()
                train_classes = y[train_idx].tolist()
                test_classes = y[test_idx].tolist()

        # Получаем распределение классов
        train_dist = self.get_class_distribution(train_indices, train_classes)
        test_dist = self.get_class_distribution(test_indices, test_classes)

        # Объединяем распределения
        all_classes = set(train_dist.keys()) | set(test_dist.keys())
        class_distribution = {}

        for class_id in all_classes:
            class_distribution[class_id] = {
                'train': train_dist.get(class_id, {'count': 0})['count'],
                'test': test_dist.get(class_id, {'count': 0})['count'],
                'total': train_dist.get(class_id, {'count': 0})['count'] +
                         test_dist.get(class_id, {'count': 0})['count']
            }

        split = TrainTestSplit(
            poly_id=poly_id,
            biome_name=biome_name,
            train_indices=train_indices,
            test_indices=test_indices,
            train_classes=train_classes,
            test_classes=test_classes,
            class_distribution=class_distribution
        )

        logger.info(f"Полигон {poly_id} ({biome_name}): "
                    f"train={len(train_indices)}, test={len(test_indices)}")

        for class_id, dist in class_distribution.items():
            logger.debug(f"  Класс {class_id}: train={dist['train']}, "
                         f"test={dist['test']}, total={dist['total']}")

        return split

    def cross_validation_split_single_polygon(self, poly_id: int,
                                              indices: List[int],
                                              classes: List[int],
                                              biome_name: str,
                                              n_folds: int = 3) -> CrossValidationSplit:
        """Кросс-валидация для одного полигона с стратификацией по классам"""

        if len(indices) < n_folds:
            logger.warning(f"Полигон {poly_id} имеет недостаточно данных для {n_folds} фолдов")
            return None

        # Проверяем распределение классов для CV
        can_cv, message = self.check_cv_distribution(classes, n_folds)

        if not can_cv:
            logger.warning(f"Полигон {poly_id} ({biome_name}): {message}. Стратифицированная CV невозможна.")

            # Используем простую K-Fold без стратификации
            from sklearn.model_selection import KFold

            X = np.array(indices).reshape(-1, 1)
            y = np.array(classes)

            kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)

            folds = []
            class_distribution = {}

            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
                train_indices = X[train_idx].flatten().tolist()
                test_indices = X[test_idx].flatten().tolist()
                train_classes = y[train_idx].tolist()
                test_classes = y[test_idx].tolist()

                # Получаем распределение классов для фолда
                train_dist = self.get_class_distribution(train_indices, train_classes)
                test_dist = self.get_class_distribution(test_indices, test_classes)

                # Обновляем общее распределение
                for class_id in set(train_dist.keys()) | set(test_dist.keys()):
                    if class_id not in class_distribution:
                        class_distribution[class_id] = {
                            'total': len([c for c in classes if c == class_id]),
                            'folds': {i: {'train': 0, 'test': 0} for i in range(n_folds)}
                        }

                    class_distribution[class_id]['folds'][fold_idx] = {
                        'train': train_dist.get(class_id, {'count': 0})['count'],
                        'test': test_dist.get(class_id, {'count': 0})['count']
                    }

                folds.append({
                    'fold': fold_idx,
                    'train_indices': train_indices,
                    'test_indices': test_indices,
                    'train_classes': train_classes,
                    'test_classes': test_classes
                })

            logger.info(f"Полигон {poly_id}: Использована простая K-Fold CV (без стратификации)")
        else:
            # Используем StratifiedKFold для стратификации
            X = np.array(indices).reshape(-1, 1)
            y = np.array(classes)

            skf = StratifiedKFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=self.random_state
            )

            folds = []
            class_distribution = {}

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                train_indices = X[train_idx].flatten().tolist()
                test_indices = X[test_idx].flatten().tolist()
                train_classes = y[train_idx].tolist()
                test_classes = y[test_idx].tolist()

                # Получаем распределение классов для фолда
                train_dist = self.get_class_distribution(train_indices, train_classes)
                test_dist = self.get_class_distribution(test_indices, test_classes)

                # Обновляем общее распределение
                for class_id in set(train_dist.keys()) | set(test_dist.keys()):
                    if class_id not in class_distribution:
                        class_distribution[class_id] = {
                            'total': len([c for c in classes if c == class_id]),
                            'folds': {i: {'train': 0, 'test': 0} for i in range(n_folds)}
                        }

                    class_distribution[class_id]['folds'][fold_idx] = {
                        'train': train_dist.get(class_id, {'count': 0})['count'],
                        'test': test_dist.get(class_id, {'count': 0})['count']
                    }

                folds.append({
                    'fold': fold_idx,
                    'train_indices': train_indices,
                    'test_indices': test_indices,
                    'train_classes': train_classes,
                    'test_classes': test_classes
                })

        cv_split = CrossValidationSplit(
            poly_id=poly_id,
            biome_name=biome_name,
            folds=folds,
            class_distribution=class_distribution
        )

        logger.info(f"Полигон {poly_id} ({biome_name}): "
                    f"{n_folds}-fold кросс-валидация создана")

        return cv_split

    def split_all_polygons(self) -> Dict[str, Any]:
        """Разделение всех полигонов на train/test и кросс-валидацию"""
        gdf = self.load_classification_polygons()

        # Группировка по poly_id (макрополигонам)
        poly_groups = gdf.groupby('poly_id')

        results = {
            'metadata': {
                'total_polygons': len(gdf),
                'unique_poly_ids': len(poly_groups),
                'biomes': sorted([str(b) for b in gdf['biome_name'].unique().tolist()]),
                'classes': sorted([int(c) for c in gdf['class'].unique().tolist()]),
                'random_state': self.random_state,
                'test_size': self.config['cross_validation']['test_size'],
                'n_folds': self.config['cross_validation']['n_folds'],
                'fid_field': 'index'
            },
            'train_test_splits': {},
            'cross_validation_splits': {},
            'statistics': {
                'by_biome': {},
                'by_class': {},
                'by_polygon': {},
                'skipped_polygons': {
                    'train_test': [],
                    'cross_validation': []
                }
            }
        }

        # Статистика по биомам
        biome_stats = gdf.groupby('biome_name').agg({
            'fid': 'count',
            'poly_id': 'nunique'
        }).rename(columns={'fid': 'total_fids', 'poly_id': 'unique_polygons'})

        for biome, stats in biome_stats.iterrows():
            results['statistics']['by_biome'][str(biome)] = {
                'total_fids': int(stats['total_fids']),
                'unique_polygons': int(stats['unique_polygons'])
            }

        # Статистика по классам
        class_stats = gdf.groupby('class').agg({
            'fid': 'count',
            'poly_id': 'nunique'
        }).rename(columns={'fid': 'total_fids', 'poly_id': 'unique_polygons'})

        for class_id, stats in class_stats.iterrows():
            results['statistics']['by_class'][int(class_id)] = {
                'total_fids': int(stats['total_fids']),
                'unique_polygons': int(stats['unique_polygons'])
            }

        # Обработка каждого макрополигона
        for poly_id, group in poly_groups:
            poly_id_int = int(poly_id)  # Преобразуем в int
            biome_name = str(group['biome_name'].iloc[0])
            indices = group['fid'].tolist()
            classes = group['class'].tolist()

            # Сохраняем информацию о полигоне
            results['statistics']['by_polygon'][poly_id_int] = {
                'biome_name': biome_name,
                'total_fids': len(indices),
                'unique_classes': sorted([int(c) for c in group['class'].unique().tolist()]),
                'class_counts': {int(k): int(v) for k, v in group['class'].value_counts().to_dict().items()}
            }

            # Проверяем распределение классов
            class_counts = Counter(classes)
            min_count = min(class_counts.values())

            # Train/test split
            try:
                tt_split = self.train_test_split_single_polygon(
                    poly_id=poly_id_int,
                    indices=indices,
                    classes=classes,
                    biome_name=biome_name,
                    test_size=self.config['cross_validation']['test_size']
                )

                if tt_split:
                    # Преобразуем class_distribution ключи в int
                    class_dist_converted = {}
                    for class_id, dist in tt_split.class_distribution.items():
                        class_dist_converted[int(class_id)] = dist
                    
                    results['train_test_splits'][poly_id_int] = {
                        'biome_name': tt_split.biome_name,
                        'train_indices': tt_split.train_indices,
                        'test_indices': tt_split.test_indices,
                        'train_classes': tt_split.train_classes,
                        'test_classes': tt_split.test_classes,
                        'class_distribution': class_dist_converted,
                        'min_class_count': int(min_count)
                    }
                else:
                    results['statistics']['skipped_polygons']['train_test'].append(poly_id_int)

            except Exception as e:
                logger.error(f"Ошибка при создании train/test split для полигона {poly_id_int}: {e}")
                results['statistics']['skipped_polygons']['train_test'].append(poly_id_int)

            # Cross-validation split
            try:
                cv_split = self.cross_validation_split_single_polygon(
                    poly_id=poly_id_int,
                    indices=indices,
                    classes=classes,
                    biome_name=biome_name,
                    n_folds=self.config['cross_validation']['n_folds']
                )

                if cv_split:
                    # Преобразуем class_distribution ключи в int
                    class_dist_converted = {}
                    for class_id, dist in cv_split.class_distribution.items():
                        class_dist_converted[int(class_id)] = dist
                    
                    results['cross_validation_splits'][poly_id_int] = {
                        'biome_name': cv_split.biome_name,
                        'folds': cv_split.folds,
                        'class_distribution': class_dist_converted,
                        'min_class_count': int(min_count)
                    }
                else:
                    results['statistics']['skipped_polygons']['cross_validation'].append(poly_id_int)

            except Exception as e:
                logger.error(f"Ошибка при создании CV split для полигона {poly_id_int}: {e}")
                results['statistics']['skipped_polygons']['cross_validation'].append(poly_id_int)

        logger.info(f"Обработано {len(results['train_test_splits'])} макрополигонов (train/test)")
        logger.info(f"Обработано {len(results['cross_validation_splits'])} макрополигонов (CV)")

        return results

    def save_splits(self, splits: Dict[str, Any]):
        """Сохранение разбиений в файл"""
        output_file = self.output_dir / self.config['cross_validation']['output_file']

        # Конвертируем numpy типы в стандартные Python типы
        def convert_types(obj):
            if isinstance(obj, dict):
                # Преобразуем ключи словаря к строке, если они не str
                new_dict = {}
                for key, value in obj.items():
                    # Преобразуем ключ к строке (если это int, float или numpy тип)
                    if isinstance(key, (np.integer, int)):
                        new_key = int(key)
                    elif isinstance(key, (np.floating, float)):
                        new_key = float(key)
                    elif isinstance(key, (np.bool_, bool)):
                        new_key = bool(key)
                    elif isinstance(key, str):
                        new_key = key
                    else:
                        # Для остальных типов преобразуем в строку
                        new_key = str(key)
                    
                    # Рекурсивно обрабатываем значение
                    new_dict[new_key] = convert_types(value)
                return new_dict
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj

        splits_converted = convert_types(splits)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(splits_converted, f, indent=2, ensure_ascii=False)

        logger.info(f"Разбиения сохранены в {output_file}")

    def load_splits(self) -> Dict[str, Any]:
        """Загрузка сохраненных разбиений"""
        output_file = self.output_dir / self.config['cross_validation']['output_file']

        if not output_file.exists():
            raise FileNotFoundError(f"Файл разбиений не найден: {output_file}")

        with open(output_file, 'r', encoding='utf-8') as f:
            splits = json.load(f)

        logger.info(f"Разбиения загружены из {output_file}")
        return splits

    def get_split_for_polygon(self, poly_id: int, split_type: str = 'train_test',
                              fold_idx: int = None) -> Dict[str, List[int]]:
        """
        Получение разбиения для конкретного полигона

        Args:
            poly_id: ID макрополигона
            split_type: 'train_test' или 'cross_validation'
            fold_idx: Индекс фолда (только для cross_validation)

        Returns:
            Словарь с train и test indices
        """
        splits = self.load_splits()
        
        # Преобразуем poly_id в строку для поиска в JSON
        poly_id_str = str(poly_id)

        if split_type == 'train_test':
            if poly_id_str not in splits['train_test_splits']:
                raise ValueError(f"Train/test split не найден для poly_id={poly_id}")

            poly_split = splits['train_test_splits'][poly_id_str]
            return {
                'train': poly_split['train_indices'],
                'test': poly_split['test_indices'],
                'train_classes': poly_split['train_classes'],
                'test_classes': poly_split['test_classes'],
                'biome_name': poly_split['biome_name']
            }

        elif split_type == 'cross_validation':
            if poly_id_str not in splits['cross_validation_splits']:
                raise ValueError(f"Cross-validation split не найден для poly_id={poly_id}")

            poly_split = splits['cross_validation_splits'][poly_id_str]

            if fold_idx is None:
                return {
                    'folds': poly_split['folds'],
                    'biome_name': poly_split['biome_name']
                }
            else:
                if fold_idx >= len(poly_split['folds']):
                    raise ValueError(f"Фолд {fold_idx} не существует. "
                                     f"Всего фолдов: {len(poly_split['folds'])}")

                fold = poly_split['folds'][fold_idx]
                return {
                    'train': fold['train_indices'],
                    'test': fold['test_indices'],
                    'train_classes': fold['train_classes'],
                    'test_classes': fold['test_classes'],
                    'fold': fold_idx,
                    'biome_name': poly_split['biome_name']
                }

        else:
            raise ValueError(f"Неизвестный тип split_type: {split_type}")

    def get_geodataframe(self) -> gpd.GeoDataFrame:
        """Получение загруженного геодатафрейма"""
        return self.load_classification_polygons()


def main():
    """Основная функция выполнения"""
    logger.info("Запуск создания train/test и cross-validation разбиений")

    try:
        splitter = TTSplitter()
        splits = splitter.split_all_polygons()
        splitter.save_splits(splits)

        # Вывод статистики
        logger.info("\n=== СТАТИСТИКА ===")
        logger.info(f"Всего макрополигонов: {splits['metadata']['unique_poly_ids']}")
        logger.info(f"Всего полигонов для обучения: {splits['metadata']['total_polygons']}")

        for biome, stats in splits['statistics']['by_biome'].items():
            logger.info(f"Биом '{biome}': {stats['total_fids']} полигонов, "
                        f"{stats['unique_polygons']} макрополигонов")

        for class_id, stats in splits['statistics']['by_class'].items():
            logger.info(f"Класс {class_id}: {stats['total_fids']} полигонов, "
                        f"{stats['unique_polygons']} макрополигонов")

        # Вывод информации по каждому макрополигону
        logger.info("\n=== ИНФОРМАЦИЯ ПО МАКРОПОЛИГОНАМ ===")
        for poly_id, stats in splits['statistics']['by_polygon'].items():
            if poly_id in splits['train_test_splits']:
                split_info = splits['train_test_splits'][poly_id]
                logger.info(f"Полигон {poly_id} ({stats['biome_name']}): "
                            f"всего {stats['total_fids']} полигонов, "
                            f"классы {list(stats['class_counts'].keys())}, "
                            f"train={len(split_info['train_indices'])}, "
                            f"test={len(split_info['test_indices'])}, "
                            f"мин. объектов в классе: {split_info['min_class_count']}")

        # Вывод информации о пропущенных полигонах
        if splits['statistics']['skipped_polygons']['train_test']:
            logger.info(
                f"\nПропущено train/test split: {len(splits['statistics']['skipped_polygons']['train_test'])} полигонов")
            logger.info(f"Полигоны: {splits['statistics']['skipped_polygons']['train_test']}")

        if splits['statistics']['skipped_polygons']['cross_validation']:
            logger.info(
                f"Пропущено cross-validation: {len(splits['statistics']['skipped_polygons']['cross_validation'])} полигонов")
            logger.info(f"Полигоны: {splits['statistics']['skipped_polygons']['cross_validation']}")

        logger.info("Разбиения успешно созданы и сохранены")

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        raise


if __name__ == "__main__":
    main()