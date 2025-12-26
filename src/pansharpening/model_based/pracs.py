import sys
from pathlib import Path

# Добавляем корневую директорию проекта в sys.path
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_dir))

import yaml
import rasterio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from skimage.exposure import match_histograms
import warnings

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


class PRACSPansharpening:
    """
    Паншарпинг методом PRACS (Partial Replacement Adaptive Component Substitution)
    Усовершенствованный компонентный метод с частичной заменой и адаптивной инъекцией деталей
    """

    def __init__(self, config_path: str = None):
        self.root_dir = Path(__file__).parent.parent.parent.parent  # Корень проекта
        if config_path is None:
            config_path = self.root_dir / 'config.yaml'
        self.config = self._load_config(config_path)
        self.method_name = "pracs"
        self._setup_paths()
        logger.info(f"Инициализирован метод PRACS паншарпенинга")

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

    def _downsample_pan_to_ms(self, pan_data: np.ndarray, pan_profile: dict, ms_profile: dict) -> np.ndarray:
        """
        Понижение разрешения PAN (15м) до разрешения MS (30м) для вычисления регрессионных коэффициентов
        """
        from rasterio.warp import reproject, Resampling

        # Получаем параметры MS данных
        ms_width = ms_profile['width']
        ms_height = ms_profile['height']
        ms_transform = ms_profile['transform']
        ms_crs = ms_profile['crs']

        # Создаем целевой массив для репроецирования
        pan_downsampled = np.zeros((ms_height, ms_width), dtype=np.float32)

        # Репроецируем PAN к разрешению MS
        reproject(
            source=pan_data,
            destination=pan_downsampled,
            src_transform=pan_profile['transform'],
            src_crs=pan_profile['crs'],
            dst_transform=ms_transform,
            dst_crs=ms_crs,
            resampling=Resampling.average
        )

        return pan_downsampled

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

    def _calculate_correlation_weights(self, ms_data: np.ndarray, pan_downsampled: np.ndarray) -> np.ndarray:
        """
        Вычисление весовых коэффициентов через корреляцию между MS каналами и PAN
        Formula: alpha_k = corr(MS_k, PAN) / sum(corr(MS_i, PAN))
        """
        n_bands = ms_data.shape[0]
        correlations = np.zeros(n_bands, dtype=np.float32)

        # Вычисляем корреляцию для каждого канала
        for k in range(n_bands):
            # Вычисляем корреляцию Пирсона между k-м каналом MS и PAN
            ms_flat = ms_data[k].flatten()
            pan_flat = pan_downsampled.flatten()

            # Исключаем нулевые и NaN значения
            mask = (ms_flat != 0) & (pan_flat != 0) & ~np.isnan(ms_flat) & ~np.isnan(pan_flat)
            if np.sum(mask) > 10:  # Минимальное количество точек для корреляции
                corr_matrix = np.corrcoef(ms_flat[mask], pan_flat[mask])
                correlations[k] = abs(corr_matrix[0, 1])  # Берем абсолютное значение
            else:
                correlations[k] = 1.0 / n_bands  # Равномерное распределение при недостатке данных

        # Нормализуем корреляции для получения весов
        if np.sum(correlations) > 0:
            weights = correlations / np.sum(correlations)
        else:
            weights = np.ones(n_bands) / n_bands

        logger.debug(f"Корреляционные веса: {weights}")
        return weights

    def _calculate_regression_coefficients(self, ms_data: np.ndarray, pan_downsampled: np.ndarray) -> np.ndarray:
        """
        Вычисление регрессионных коэффициентов для инъекции деталей
        Formula: g_k = cov(MS_k, PAN) / var(PAN)
        """
        n_bands = ms_data.shape[0]
        coefficients = np.zeros(n_bands, dtype=np.float32)

        # Вычисляем дисперсию PAN
        pan_variance = np.var(pan_downsampled)

        # Избегаем деления на ноль
        if pan_variance < 1e-10:
            logger.warning("Низкая дисперсия PAN, использование единичных коэффициентов")
            return np.ones(n_bands, dtype=np.float32)

        # Вычисляем коэффициенты для каждого канала
        for k in range(n_bands):
            # Вычисляем ковариацию между k-м каналом MS и PAN
            covariance = np.cov(ms_data[k].flatten(), pan_downsampled.flatten())[0, 1]
            coefficients[k] = covariance / pan_variance

            # Ограничиваем коэффициенты разумными пределами для стабильности
            coefficients[k] = np.clip(coefficients[k], 0.1, 2.0)

        logger.debug(f"Регрессионные коэффициенты инъекции: {coefficients}")
        return coefficients

    def _calculate_low_freq_pan(self, ms_upsampled: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Вычисление низкочастотной версии PAN как взвешенной суммы MS каналов
        Formula: PAN_low = sum(beta_k * MS_upsampled_k)
        В PRACS используются те же веса, что и для интенсивности
        """
        pan_low = np.zeros_like(ms_upsampled[0], dtype=np.float32)

        for i in range(ms_upsampled.shape[0]):
            pan_low += weights[i] * ms_upsampled[i]

        return pan_low

    def _calculate_intensity_component(self, ms_upsampled: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Вычисление яркостной компоненты как взвешенной суммы MS каналов
        Formula: I_MS = sum(alpha_k * MS_upsampled_k)
        """
        intensity = np.zeros_like(ms_upsampled[0], dtype=np.float32)

        for i in range(ms_upsampled.shape[0]):
            intensity += weights[i] * ms_upsampled[i]

        return intensity

    def _apply_histogram_matching(self, pan_data: np.ndarray, intensity: np.ndarray) -> np.ndarray:
        """
        Применение гистограммного согласования PAN канала с интенсивностью MS
        """
        # Нормализуем данные к диапазону [0, 1] для гистограммного согласования
        pan_normalized = (pan_data - np.min(pan_data)) / (np.max(pan_data) - np.min(pan_data))
        intensity_normalized = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))

        # Применяем гистограммное согласование
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pan_matched = match_histograms(pan_normalized, intensity_normalized)

        # Возвращаем к исходному диапазону значений PAN
        pan_matched = pan_matched * (np.max(pan_data) - np.min(pan_data)) + np.min(pan_data)

        return pan_matched

    def _apply_pracs_transform(self, ms_upsampled: np.ndarray, pan_matched: np.ndarray,
                               intensity: np.ndarray, pan_low: np.ndarray,
                               coefficients: np.ndarray) -> np.ndarray:
        """
        Применение метода PRACS для паншарпенинга
        Formula: PS_i = MS_upsampled_i + g_i * (PAN_matched - PAN_low)
        Частичная замена с адаптивной инъекцией деталей
        """
        # Проверяем совместимость размеров
        if pan_matched.shape != ms_upsampled.shape[1:]:
            raise ValueError(
                f"Несовместимые размеры: MS {ms_upsampled.shape[1:]}, PAN {pan_matched.shape}")

        # Вычисляем высокочастотные детали PAN
        pan_detail = pan_matched - pan_low

        # Применяем адаптивную инъекцию деталей с индивидуальными коэффициентами для каждого канала
        pansharpened = np.zeros_like(ms_upsampled, dtype=np.float32)
        for i in range(ms_upsampled.shape[0]):
            pansharpened[i] = ms_upsampled[i] + coefficients[i] * pan_detail

        return pansharpened

    def process_single_polygon(self, ms_path: Path, pan_path: Path, biome_name: str, fid: int) -> Optional[
        PansharpeningResult]:
        """
        Обработка одного полигона методом PRACS паншарпенинга
        """
        # logger.info(f"Обработка полигона {fid} биома {biome_name} методом {self.method_name}")

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

            # Понижение разрешения PAN для вычисления коэффициентов
            pan_downsampled = self._downsample_pan_to_ms(pan_data, pan_profile, ms_profile)
            logger.debug(f"PAN понижен до разрешения MS: {pan_downsampled.shape}")

            # Вычисление корреляционных весов для интенсивности и PAN_low
            weights = self._calculate_correlation_weights(ms_data, pan_downsampled)
            logger.debug(f"Вычислены корреляционные веса: {weights}")

            # Вычисление регрессионных коэффициентов для инъекции деталей
            coefficients = self._calculate_regression_coefficients(ms_data, pan_downsampled)
            logger.debug(f"Вычислены регрессионные коэффициенты: {coefficients}")

            # Апсемплинг MS данных до разрешения PAN (30м -> 15м)
            ms_upsampled = self._upsample_ms_to_pan(ms_data, ms_profile, pan_profile)

            # Вычисление яркостной компоненты
            intensity = self._calculate_intensity_component(ms_upsampled, weights)
            logger.debug(f"Вычислена интенсивность MS: min={np.min(intensity):.4f}, max={np.max(intensity):.4f}")

            # Вычисление низкочастотной версии PAN
            pan_low = self._calculate_low_freq_pan(ms_upsampled, weights)
            logger.debug(f"Вычислен PAN_low: min={np.min(pan_low):.4f}, max={np.max(pan_low):.4f}")

            # Гистограммное согласование PAN с интенсивностью MS
            pan_matched = self._apply_histogram_matching(pan_data, intensity)
            logger.debug(
                f"Гистограммное согласование PAN: min={np.min(pan_matched):.4f}, max={np.max(pan_matched):.4f}")

            # Применяем метод PRACS паншарпенинга
            pansharpened_data = self._apply_pracs_transform(
                ms_upsampled, pan_matched, intensity, pan_low, coefficients
            )

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

            # logger.info(f"Успешно обработан полигон {fid} методом {self.method_name}")
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
        logger.info(f"Начало обработки всех биомов методом {self.method_name}")

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
                    logger.error(
                        f"Ошибка паншарпенинга полигона {poly_info['fid']} биома {biome_name} методом {self.method_name}")

        logger.info(f"Обработка {self.method_name} завершена. Успешно: {total_processed}, Ошибок: {total_errors}")


def main():
    """Основная функция выполнения"""
    logger.info("Запуск паншарпенинга методом PRACS")

    try:
        pracs_processor = PRACSPansharpening()
        pracs_processor.process_all_biomes()
        logger.info("Паншарпинг PRACS успешно завершен")

    except Exception as e:
        logger.error(f"Критическая ошибка в выполнении: {e}")


if __name__ == "__main__":
    main()