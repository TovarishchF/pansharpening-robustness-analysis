import yaml
import rasterio
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CorrectedData:
    """Результаты атмосферной коррекции"""
    ms_corrected: np.ndarray  # 6-канальный MS (B2-B7)
    pan_corrected: np.ndarray  # 1-канальный PAN (B8)
    ms_metadata: dict
    pan_metadata: dict
    scene_name: str


class AtmosphericCorrection:
    """
    Атмосферная коррекция с экспортом многоканальных файлов
    """

    def __init__(self, config_path: str = None):
        self.root_dir = Path(__file__).parent.parent.parent  # Корень проекта
        if config_path is None:
            config_path = self.root_dir / 'config.yaml'
        self.config = self._load_config(config_path)
        logger.info(f"Корневая директория проекта: {self.root_dir}")
        self._setup_paths()

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_paths(self):
        """Настройка путей для экспорта"""
        if self.config['export']['save_intermediate']:
            self.export_path = self.root_dir / self.config['export']['intermediate_path'] / "corrected_data"
            self.export_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Путь для экспорта: {self.export_path}")

    def find_raw_scene(self) -> Optional[Path]:
        """Поиск единственной сцены Landsat в папке raw"""
        raw_path = self.root_dir / self.config['data']['raw']
        logger.info(f"Ищем сырые данные в: {raw_path}")

        if not raw_path.exists():
            logger.error(f"Путь не существует: {raw_path}")
            return None

        # Проверяем, есть ли MTL файл
        mtl_files = list(raw_path.glob("*_MTL.txt"))
        if not mtl_files:
            logger.error(f"MTL файл не найден в {raw_path}")
            return None

        # Берем первый MTL файл для определения сцены
        mtl_file = mtl_files[0]
        scene_name = mtl_file.name.replace('_MTL.txt', '')
        logger.info(f"Найдена сцена: {scene_name}")

        # Возвращаем путь к папке raw как место расположения сцены
        return raw_path

    def _parse_mtl(self, scene_path: Path) -> Dict:
        """Парсинг MTL файла"""
        mtl_files = list(scene_path.glob('*_MTL.txt'))
        if not mtl_files:
            raise FileNotFoundError(f"MTL файл не найден в {scene_path}")

        # Берем первый найденный MTL файл
        mtl_file = mtl_files[0]
        logger.info(f"Используем MTL файл: {mtl_file.name}")

        metadata = {}
        with open(mtl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    metadata[key.strip()] = value.strip().strip('"')
        return metadata

    def _open_band_files(self, scene_path: Path, metadata: Dict) -> Tuple[Dict, Dict]:
        """Открытие файлов каналов"""
        band_files = {}
        band_metadata = {}

        # MS каналы (B2-B7)
        for band in self.config['landsat']['ms_bands']:
            band_key = f"FILE_NAME_BAND_{band}"
            if band_key in metadata:
                band_filename = metadata[band_key]
                band_path = scene_path / band_filename
                if band_path.exists():
                    try:
                        band_files[f"B{band}"] = rasterio.open(band_path)
                        band_metadata[f"B{band}"] = band_files[f"B{band}"].meta
                        logger.info(f"Загружен канал B{band}: {band_filename}")
                    except Exception as e:
                        logger.error(f"Ошибка загрузки канала B{band}: {e}")
                else:
                    logger.warning(f"Файл канала не найден: {band_path}")
            else:
                logger.warning(f"Метаданные для канала {band} не найдены")

        # PAN канал (B8)
        pan_key = f"FILE_NAME_BAND_{self.config['landsat']['pan_band']}"
        if pan_key in metadata:
            pan_filename = metadata[pan_key]
            pan_path = scene_path / pan_filename
            if pan_path.exists():
                try:
                    band_files["PAN"] = rasterio.open(pan_path)
                    band_metadata["PAN"] = band_files["PAN"].meta
                    logger.info(f"Загружен PAN канал: {pan_filename}")
                except Exception as e:
                    logger.error(f"Ошибка загрузки PAN канала: {e}")
            else:
                logger.warning(f"PAN файл не найден: {pan_path}")
        else:
            logger.warning("Метаданные для PAN канала не найдены")

        return band_files, band_metadata

    def _apply_atmospheric_correction_ms(self, dn_array: np.ndarray,
                                         sun_elevation: float) -> np.ndarray:
        """Применение атмосферной коррекции DOS1 для MS каналов"""
        # Параметры коррекции (в реальности из MTL)
        reflectance_mult = 2.0e-05
        reflectance_add = -0.1

        # TOA отражательность
        toa_reflectance = reflectance_mult * dn_array + reflectance_add
        toa_reflectance = toa_reflectance / np.sin(np.radians(sun_elevation))

        # DOS1 коррекция
        valid_mask = (toa_reflectance > 0) & (toa_reflectance < 1.0)
        if np.sum(valid_mask) > 0:
            dark_object = np.percentile(toa_reflectance[valid_mask], 1)
            corrected = toa_reflectance - dark_object
            corrected = np.clip(corrected, 0, None)
        else:
            corrected = toa_reflectance

        return corrected.astype(np.float32)

    def _apply_radiometric_correction_pan(self, dn_array: np.ndarray,
                                          sun_elevation: float) -> np.ndarray:
        """
        Радиометрическая коррекция для PAN канала
        Только TOA отражательность без атмосферной коррекции
        """
        # Параметры коррекции
        reflectance_mult = 2.0e-05
        reflectance_add = -0.1

        # TOA отражательность
        toa_reflectance = reflectance_mult * dn_array + reflectance_add
        toa_reflectance = toa_reflectance / np.sin(np.radians(sun_elevation))

        return np.clip(toa_reflectance, 0, 1).astype(np.float32)

    def process_scene(self, scene_path: Path) -> Optional[CorrectedData]:
        """Обработка одной сцены с коррекцией"""
        logger.info(f"Атмосферная коррекция сцены в: {scene_path}")

        try:
            # Загрузка метаданных
            metadata = self._parse_mtl(scene_path)
            sun_elevation = float(metadata.get('SUN_ELEVATION', 45.0))
            logger.info(f"Высота солнца: {sun_elevation}")

            # Открытие и обработка каналов
            band_files, band_metadata = self._open_band_files(scene_path, metadata)

            # Проверка что все каналы загружены
            expected_bands = 7  # 6 MS + 1 PAN
            if len(band_files) < expected_bands:
                logger.error(f"Не все каналы найдены. Найдено: {len(band_files)}, ожидалось: {expected_bands}")
                for band_file in band_files.values():
                    band_file.close()
                return None

            # Коррекция MS каналов (полная атмосферная коррекция)
            ms_corrected = []
            ms_bands_order = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']

            for band_name in ms_bands_order:
                if band_name in band_files:
                    dn_data = band_files[band_name].read(1)
                    corrected_band = self._apply_atmospheric_correction_ms(dn_data, sun_elevation)
                    ms_corrected.append(corrected_band)
                    logger.info(f"Обработан MS канал {band_name}")

            # Коррекция PAN канала (только радиометрическая)
            pan_corrected = None
            if "PAN" in band_files:
                pan_dn = band_files["PAN"].read(1)
                pan_corrected = self._apply_radiometric_correction_pan(pan_dn, sun_elevation)
                logger.info("Обработан PAN канал")

            # Закрытие файлов
            for band_file in band_files.values():
                band_file.close()

            if not ms_corrected or pan_corrected is None:
                logger.error(f"Не удалось обработать каналы")
                return None

            # Сборка многоканального MS
            ms_stack = np.stack(ms_corrected, axis=0)

            # Базовые метаданные
            base_ms_meta = band_metadata['B2'].copy()
            base_ms_meta.update({'count': len(ms_corrected)})

            # Получаем имя сцены из MTL файла
            scene_name = metadata.get('LANDSAT_SCENE_ID', 'unknown_scene')

            corrected_data = CorrectedData(
                ms_corrected=ms_stack,
                pan_corrected=pan_corrected,
                ms_metadata=base_ms_meta,
                pan_metadata=band_metadata['PAN'].copy(),
                scene_name=scene_name
            )

            # Экспорт если включено
            if self.config['export']['save_intermediate']:
                self._export_corrected_data(corrected_data)

            logger.info(f"Успешно обработана сцена: {scene_name}")
            return corrected_data

        except Exception as e:
            logger.error(f"Ошибка коррекции: {e}")
            return None

    def _export_corrected_data(self, data: CorrectedData):
        """Экспорт скорректированных данных"""
        # Многоканальный MS файл
        ms_profile = data.ms_metadata.copy()
        ms_profile.update({
            'dtype': 'float32',
            'compress': 'DEFLATE'
        })

        ms_path = self.export_path / f"{data.scene_name}_MS.tif"
        with rasterio.open(ms_path, 'w', **ms_profile) as dst:
            dst.write(data.ms_corrected)
            # Установка описаний каналов
            for i, band_name in enumerate(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'], 1):
                dst.set_band_description(i, band_name)

        # PAN файл
        pan_profile = data.pan_metadata.copy()
        pan_profile.update({
            'dtype': 'float32',
            'compress': 'DEFLATE'
        })

        pan_path = self.export_path / f"{data.scene_name}_PAN.tif"
        with rasterio.open(pan_path, 'w', **pan_profile) as dst:
            dst.write(data.pan_corrected.reshape(1, *data.pan_corrected.shape))
            dst.set_band_description(1, 'PAN')

        logger.info(f"Экспортированы скорректированные данные: {data.scene_name}")


def main():
    logger.info("Запуск атмосферной коррекции")
    corrector = AtmosphericCorrection()

    # Найти единственную сцену
    scene_path = corrector.find_raw_scene()
    if not scene_path:
        logger.error("Сцена не найдена")
        return

    # Обработать сцену
    corrected_data = corrector.process_scene(scene_path)
    if corrected_data:
        logger.info(f"Успешно обработана сцена: {corrected_data.scene_name}")
    else:
        logger.error("Не удалось обработать сцену")


if __name__ == "__main__":
    main()