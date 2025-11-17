import yaml
import rasterio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter

from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@dataclass
class PansharpeningResult:
    """Результаты паншарпенинга"""
    biome_name: str
    fid: int
    pansharpened_data: np.ndarray
    transform: any
    crs: str
    profile: dict


class HPFPansharpening:
    """
    Паншарпинг методом High-Pass Filtering (HPF)
    """

    def __init__(self, config_path: str = None):
        self.root_dir = Path(__file__).parent.parent.parent.parent
        if config_path is None:
            config_path = self.root_dir / 'config.yaml'
        self.config = self._load_config(config_path)
        self.method_name = "hpf"
        self._setup_paths()
        logger.info(f"Инициализирован метод {self.method_name} паншарпенинга")

        # Параметры гауссовского фильтра для выделения высоких частот
        self.sigma = 1.5  # Стандартное отклонение для гауссовского фильтра

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_paths(self):
        """Настройка путей для ввода и вывода"""
        self.input_path = self.root_dir / self.config['data']['intermediate'] / "clipped_polygons"
        self.output_path = self.root_dir / self.config['data']['processed'] / "pansharpening"
        self.output_path.mkdir(parents=True, exist_ok=True)

        for biome in self.config['biomes'].keys():
            biome_path = self.output_path / biome / self.method_name
            biome_path.mkdir(parents=True, exist_ok=True)

    def _find_polygon_files(self) -> Dict[str, List[Dict]]:
        """Поиск всех файлов полигонов по биомам"""
        polygon_files = {}

        for biome in self.config['biomes'].keys():
            biome_path = self.input_path / biome
            if not biome_path.exists():
                logger.warning(f"Путь биома не существует: {biome_path}")
                continue

            polygon_files[biome] = []
            ms_files = list(biome_path.glob("*_MS.tif"))

            for ms_path in ms_files:
                fid_str = ms_path.stem.replace('_MS', '')
                try:
                    fid = int(fid_str)
                except ValueError:
                    logger.warning(f"Неверный формат fid в файле: {ms_path.name}")
                    continue

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
        """Апсемплинг MS данных до разрешения PAN"""
        from rasterio.warp import reproject, Resampling

        pan_width = pan_profile['width']
        pan_height = pan_profile['height']
        pan_transform = pan_profile['transform']
        pan_crs = pan_profile['crs']

        ms_upsampled = np.zeros((ms_data.shape[0], pan_height, pan_width), dtype=np.float32)

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

    def _extract_high_frequency_details(self, pan_data: np.ndarray) -> np.ndarray:
        """
        Извлечение высокочастотных деталей из PAN изображения
        с использованием гауссовского фильтра
        """
        # Применяем гауссовский фильтр низких частот
        pan_low_freq = gaussian_filter(pan_data, sigma=self.sigma)

        # Вычисляем высокочастотные детали как разность
        pan_high_freq = pan_data - pan_low_freq

        logger.debug(f"Высокочастотные детали: min={np.min(pan_high_freq):.4f}, max={np.max(pan_high_freq):.4f}")

        return pan_high_freq

    def _calculate_adaptive_gains(self, ms_upsampled: np.ndarray, pan_data: np.ndarray) -> np.ndarray:
        """
        Расчет адаптивных коэффициентов инъекции на основе корреляции
        между MS каналами и высокочастотными деталями PAN
        """
        # Сначала извлекаем высокочастотные детали PAN
        pan_high_freq = self._extract_high_frequency_details(pan_data)

        gains = np.zeros(ms_upsampled.shape[0], dtype=np.float32)

        for i in range(ms_upsampled.shape[0]):
            # Вычисляем корреляцию между MS каналом и высокочастотными деталями PAN
            ms_flat = ms_upsampled[i].flatten()
            pan_hf_flat = pan_high_freq.flatten()

            # Используем только валидные пиксели (не NaN и не бесконечные)
            valid_mask = ~(np.isnan(ms_flat) | np.isnan(pan_hf_flat) |
                           np.isinf(ms_flat) | np.isinf(pan_hf_flat))

            if np.sum(valid_mask) > 100:  # Минимальное количество валидных пикселей
                correlation = np.corrcoef(ms_flat[valid_mask], pan_hf_flat[valid_mask])[0, 1]
                if not np.isnan(correlation):
                    gains[i] = np.clip(correlation, 0.1, 1.0)
                else:
                    gains[i] = 0.5  # Значение по умолчанию при отсутствии корреляции
            else:
                gains[i] = 0.5  # Значение по умолчанию

        logger.debug(f"Коэффициенты инъекции HPF: {gains}")
        return gains

    def _apply_hpf_pansharpening(self, ms_upsampled: np.ndarray, pan_data: np.ndarray) -> np.ndarray:
        """
        Применение HPF паншарпенинга
        """
        # Извлекаем высокочастотные детали из PAN
        pan_high_freq = self._extract_high_frequency_details(pan_data)

        # Рассчитываем адаптивные коэффициенты инъекции
        gains = self._calculate_adaptive_gains(ms_upsampled, pan_data)

        # Применяем инъекцию высокочастотных деталей к каждому MS каналу
        pansharpened = np.zeros_like(ms_upsampled, dtype=np.float32)
        for i in range(ms_upsampled.shape[0]):
            pansharpened[i] = ms_upsampled[i] + gains[i] * pan_high_freq

        return pansharpened

    def process_single_polygon(self, ms_path: Path, pan_path: Path, biome_name: str, fid: int) -> Optional[
        PansharpeningResult]:
        """Обработка одного полигона методом HPF"""
        # logger.info(f"Обработка полигона {fid} биома {biome_name} методом HPF")

        try:
            # Загрузка MS данных
            with rasterio.open(ms_path) as ms_src:
                ms_data = ms_src.read()
                ms_profile = ms_src.profile
                ms_crs = ms_src.crs

            # Загрузка PAN данных
            with rasterio.open(pan_path) as pan_src:
                pan_data = pan_src.read(1)
                pan_profile = pan_src.profile
                pan_transform = pan_src.transform
                pan_crs = pan_src.crs

            if ms_crs != pan_crs:
                logger.warning(f"Разные CRS: MS {ms_crs}, PAN {pan_crs}")

            # Апсемплинг MS данных
            # logger.info(f"Апсемплинг MS {ms_data.shape} -> PAN разрешение {pan_data.shape}")
            ms_upsampled = self._upsample_ms_to_pan(ms_data, ms_profile, pan_profile)

            # Применяем метод HPF
            pansharpened_data = self._apply_hpf_pansharpening(ms_upsampled, pan_data)

            # Создаем профиль для выходных данных
            output_profile = pan_profile.copy()
            output_profile.update({
                'count': pansharpened_data.shape[0],
                'dtype': 'float32'
            })

            result = PansharpeningResult(
                biome_name=biome_name,
                fid=fid,
                pansharpened_data=pansharpened_data,
                transform=pan_transform,
                crs=pan_crs,
                profile=output_profile
            )

            # logger.info(f"Успешно обработан полигон {fid} методом HPF")
            return result

        except Exception as e:
            logger.error(f"Ошибка обработки полигона {fid}: {e}")
            return None

    def _export_pansharpened(self, result: PansharpeningResult):
        """Экспорт паншарпенных данных"""
        output_profile = result.profile.copy()
        output_profile.update({
            'dtype': 'float32',
            'compress': 'DEFLATE'
        })

        method_path = self.output_path / result.biome_name / self.method_name
        output_filename = f"{result.fid:02d}_{self.method_name}.tif"
        output_path = method_path / output_filename

        with rasterio.open(output_path, 'w', **output_profile) as dst:
            dst.write(result.pansharpened_data)
            band_descriptions = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
            for i, desc in enumerate(band_descriptions, 1):
                dst.set_band_description(i, desc)

        # logger.info(f"Экспортирован паншарпенный файл: {output_path}")

    def process_all_biomes(self):
        """Обработка всех биомов и полигонов"""
        logger.info("Начало обработки всех биомов методом HPF")

        polygon_files = self._find_polygon_files()
        total_processed = 0
        total_errors = 0

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
                    # logger.info(f"Успешно паншарплен полигон {poly_info['fid']} биома {biome_name} методом HPF")
                else:
                    total_errors += 1
                    logger.error(f"Ошибка паншарпенинга полигона {poly_info['fid']} биома {biome_name} методом HPF")

        logger.info(f"Обработка HPF завершена. Успешно: {total_processed}, Ошибок: {total_errors}")


def main():
    """Основная функция выполнения"""
    logger.info("Запуск паншарпенинга методом HPF")

    try:
        hpf_processor = HPFPansharpening()
        hpf_processor.process_all_biomes()
        # logger.info("Паншарпинг HPF успешно завершен")

    except Exception as e:
        logger.error(f"Критическая ошибка в выполнении: {e}")


if __name__ == "__main__":
    main()