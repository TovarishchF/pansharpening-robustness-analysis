import sys
from pathlib import Path

# Добавляем корневую директорию проекта в sys.path
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_dir))

import yaml
import rasterio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from sklearn.decomposition import PCA
from skimage.exposure import match_histograms

# Импорт кастомного логгера
from src.utils.logger import get_logger, setup_logging

# Инициализация логгера
setup_logging()
logger = get_logger(__name__)


@dataclass
class PansharpeningResult:
    """Результаты паншарпенинга"""
    biome_name: str
    fid: int
    pansharpened_data: np.ndarray  # 6-канальный паншарпенный MS в высоком разрешении
    transform: any
    crs: str
    profile: dict


class PCAPansharpening:
    """
    Паншарпинг методом Principal Component Analysis (PCA)
    """

    def __init__(self, config_path: str = None):
        self.root_dir = Path(__file__).parent.parent.parent.parent  # Корень проекта
        if config_path is None:
            config_path = self.root_dir / 'config.yaml'
        self.config = self._load_config(config_path)
        self.method_name = "pca"
        self._setup_paths()
        logger.info(f"Инициализирован метод {self.method_name} паншарпенинга")

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_paths(self):
        """Настройка путей для ввода и вывода"""
        # Входные данные (обрезанные полигоны)
        self.input_path = self.root_dir / self.config['data']['intermediate'] / "clipped_polygons"

        # Выходные данные (паншарпенные)
        self.output_path = self.root_dir / self.config['data']['processed'] / "pansharpening"
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Создание папок для каждого биома
        for biome in self.config['biomes'].keys():
            biome_path = self.output_path / biome / self.method_name
            biome_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Входной путь: {self.input_path}")
        logger.info(f"Выходной путь: {self.output_path}")

    def _find_polygon_files(self) -> Dict[str, List[Dict]]:
        """
        Поиск всех файлов полигонов по биомам
        Returns: Словарь {biome: [{'fid': int, 'ms_path': Path, 'pan_path': Path}]}
        """
        polygon_files = {}

        for biome in self.config['biomes'].keys():
            biome_path = self.input_path / biome
            if not biome_path.exists():
                logger.warning(f"Путь биома не существует: {biome_path}")
                continue

            polygon_files[biome] = []

            # Ищем все MS файлы в папке биома
            ms_files = list(biome_path.glob("*_MS.tif"))

            for ms_path in ms_files:
                # Извлекаем fid из имени файла
                fid_str = ms_path.stem.replace('_MS', '')
                try:
                    fid = int(fid_str)
                except ValueError:
                    logger.warning(f"Неверный формат fid в файле: {ms_path.name}")
                    continue

                # Ищем соответствующий PAN файл
                pan_path = biome_path / f"{fid:02d}_PAN.tif"
                if not pan_path.exists():
                    logger.warning(f"PAN файл не найден для fid {fid}: {pan_path}")
                    continue

                polygon_files[biome].append({
                    'fid': fid,
                    'ms_path': ms_path,
                    'pan_path': pan_path
                })

            logger.info(f"Найдено {len(polygon_files[biome])} полигонов для биома {biome}")

        return polygon_files

    def _upsample_ms_to_pan(self, ms_data: np.ndarray, ms_profile: dict, pan_profile: dict) -> np.ndarray:
        """
        Апсемплинг MS данных (30м) до разрешения PAN (15м) с использованием билинейной интерполяции
        """
        from rasterio.warp import reproject, Resampling

        # Получаем параметры PAN данных
        pan_width = pan_profile['width']
        pan_height = pan_profile['height']
        pan_transform = pan_profile['transform']
        pan_crs = pan_profile['crs']

        # Создаем целевой массив для репроецирования (6 каналов, высокое разрешение)
        ms_upsampled = np.zeros((ms_data.shape[0], pan_height, pan_width), dtype=np.float32)

        # Репроецируем каждый канал MS к разрешению PAN
        for i in range(ms_data.shape[0]):
            reproject(
                source=ms_data[i],
                destination=ms_upsampled[i],
                src_transform=ms_profile['transform'],
                src_crs=ms_profile['crs'],
                dst_transform=pan_transform,
                dst_crs=pan_crs,
                resampling=Resampling.bilinear
            )

        return ms_upsampled

    def _apply_pca_pansharpening(self, ms_upsampled: np.ndarray, pan_data: np.ndarray) -> np.ndarray:
        """
        Применение PCA паншарпенинга
        """
        # Получаем размеры данных
        n_bands, height, width = ms_upsampled.shape

        # Преобразуем MS данные в 2D матрицу (пиксели x каналы)
        ms_2d = ms_upsampled.reshape(n_bands, -1).T  # Форма: (n_pixels, n_bands)

        # Применяем PCA
        pca = PCA()
        pca_components = pca.fit_transform(ms_2d)  # Форма: (n_pixels, n_components)

        # Восстанавливаем форму компонент PCA
        pca_components_3d = pca_components.T.reshape(n_bands, height, width)

        # Гистограммное согласование PAN с первым главным компонентом
        first_component = pca_components_3d[0]

        # Нормализуем данные для гистограммного согласования
        pan_normalized = (pan_data - np.min(pan_data)) / (np.max(pan_data) - np.min(pan_data))
        first_component_normalized = (first_component - np.min(first_component)) / (
                    np.max(first_component) - np.min(first_component))

        # Применяем гистограммное согласование
        pan_matched = match_histograms(pan_normalized, first_component_normalized)

        # Возвращаем к исходному диапазону значений первого компонента
        pan_matched = pan_matched * (np.max(first_component) - np.min(first_component)) + np.min(first_component)

        # Заменяем первый главный компонент согласованным PAN каналом
        pca_components_3d[0] = pan_matched

        # Преобразуем обратно в 2D для обратного PCA
        pca_components_2d = pca_components_3d.reshape(n_bands, -1).T  # Форма: (n_pixels, n_bands)

        # Обратное PCA преобразование
        pansharpened_2d = pca.inverse_transform(pca_components_2d)  # Форма: (n_pixels, n_bands)

        # Восстанавливаем исходную 3D форму
        pansharpened_3d = pansharpened_2d.T.reshape(n_bands, height, width)

        # Объясненная дисперсия для логирования
        explained_variance = pca.explained_variance_ratio_
        logger.debug(f"Объясненная дисперсия PCA компонент: {[f'{v:.3f}' for v in explained_variance]}")
        logger.debug(f"Первая компонента объясняет {explained_variance[0] * 100:.2f}% дисперсии")

        return pansharpened_3d

    def process_single_polygon(self, ms_path: Path, pan_path: Path, biome_name: str, fid: int) -> Optional[
        PansharpeningResult]:
        """
        Обработка одного полигона методом PCA паншарпенинга
        """
        # logger.info(f"Обработка полигона {fid} биома {biome_name} методом PCA")

        try:
            # Загрузка MS данных (6 каналов, 30м)
            with rasterio.open(ms_path) as ms_src:
                ms_data = ms_src.read()
                ms_profile = ms_src.profile
                ms_crs = ms_src.crs

            # Загрузка PAN данных (1 канал, 15м)
            with rasterio.open(pan_path) as pan_src:
                pan_data = pan_src.read(1)
                pan_profile = pan_src.profile
                pan_transform = pan_src.transform
                pan_crs = pan_src.crs

            # Проверяем совместимость CRS
            if ms_crs != pan_crs:
                logger.warning(f"Разные CRS: MS {ms_crs}, PAN {pan_crs}")

            # Апсемплинг MS данных до разрешения PAN (30м -> 15м)
            # logger.info(f"Апсемплинг MS {ms_data.shape} -> PAN разрешение {pan_data.shape}")
            ms_upsampled = self._upsample_ms_to_pan(ms_data, ms_profile, pan_profile)

            # Применяем метод PCA паншарпенинга
            pansharpened_data = self._apply_pca_pansharpening(ms_upsampled, pan_data)

            # Создаем профиль для выходных данных (высокое разрешение)
            output_profile = pan_profile.copy()
            output_profile.update({
                'count': pansharpened_data.shape[0],  # 6 каналов
                'dtype': 'float32'
            })

            result = PansharpeningResult(
                biome_name=biome_name,
                fid=fid,
                pansharpened_data=pansharpened_data,
                transform=pan_transform,  # Используем трансформ PAN (высокое разрешение)
                crs=pan_crs,
                profile=output_profile
            )

            # logger.info(f"Успешно обработан полигон {fid} методом PCA")
            return result

        except Exception as e:
            logger.error(f"Ошибка обработки полигона {fid}: {e}")
            return None

    def _export_pansharpened(self, result: PansharpeningResult):
        """
        Экспорт паншарпенных данных
        """
        output_profile = result.profile.copy()
        output_profile.update({
            'dtype': 'float32',
            'compress': 'DEFLATE'
        })

        # Путь для экспорта
        method_path = self.output_path / result.biome_name / self.method_name
        output_filename = f"{result.fid:02d}_{self.method_name}.tif"
        output_path = method_path / output_filename

        with rasterio.open(output_path, 'w', **output_profile) as dst:
            dst.write(result.pansharpened_data)
            # Установка описаний каналов
            band_descriptions = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
            for i, desc in enumerate(band_descriptions, 1):
                dst.set_band_description(i, desc)

        # logger.info(f"Экспортирован паншарпенный файл: {output_path}")

    def process_all_biomes(self):
        """
        Обработка всех биомов и полигонов
        """
        logger.info("Начало обработки всех биомов методом PCA паншарпенинга")

        # Находим все файлы полигонов
        polygon_files = self._find_polygon_files()

        total_processed = 0
        total_errors = 0

        # Обрабатываем каждый биом
        for biome_name, polygons in polygon_files.items():
            logger.info(f"Обработка биома {biome_name} ({len(polygons)} полигонов)")

            for poly_info in polygons:
                result = self.process_single_polygon(
                    ms_path=poly_info['ms_path'],
                    pan_path=poly_info['pan_path'],
                    biome_name=biome_name,
                    fid=poly_info['fid']
                )

                if result:
                    self._export_pansharpened(result)
                    total_processed += 1
                    # logger.info(f"Успешно паншарплен полигон {poly_info['fid']} биома {biome_name} методом {self.method_name}")
                else:
                    total_errors += 1
                    logger.error(f"Ошибка паншарпенинга полигона {poly_info['fid']} биома {biome_name} методом {self.method_name}")

        logger.info(f"Обработка {self.method_name} завершена. Успешно: {total_processed}, Ошибок: {total_errors}")


def main():
    """Основная функция выполнения"""
    logger.info("Запуск паншарпенинга методом PCA")

    try:
        pca_processor = PCAPansharpening()
        pca_processor.process_all_biomes()
        logger.info("Паншарпинг PCA успешно завершен")

    except Exception as e:
        logger.error(f"Критическая ошибка в выполнении: {e}")


if __name__ == "__main__":
    main()