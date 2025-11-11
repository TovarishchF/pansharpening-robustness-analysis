import yaml
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
from pathlib import Path
import logging
from typing import Iterator, Optional, Tuple
from dataclasses import dataclass

# Импорт кастомного логгера
from src.utils.logger import get_logger, setup_logging

# Инициализация логгера
setup_logging()
logger = get_logger(__name__)


@dataclass
class ClippedPolygon:
    """Обрезанные данные полигона"""
    biome_name: str
    fid: int  # Используем fid вместо poly_id
    ms_data: np.ndarray  # 6-канальный MS
    pan_data: np.ndarray  # 1-канальный PAN
    transform: any
    crs: str


class ClippingTool:
    """
    Инструмент обрезки с экспортом многоканальных файлов
    {fid:02d}_MS.tif и {fid:02d}_PAN.tif в папках биомов
    """

    def __init__(self, config_path: str = None):
        self.root_dir = Path(__file__).parent.parent.parent
        if config_path is None:
            config_path = self.root_dir / 'config.yaml'
        self.config = self._load_config(config_path)
        self._setup_paths()

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_paths(self):
        """Настройка путей для экспорта"""
        self.export_path = self.root_dir / self.config['data']['intermediate'] / "clipped_polygons"
        self.export_path.mkdir(parents=True, exist_ok=True)

        # Создание папок для каждого биома
        for biome in self.config['biomes'].keys():
            biome_path = self.export_path / biome
            biome_path.mkdir(exist_ok=True)

    def load_polygons(self) -> gpd.GeoDataFrame:
        """Загрузка полигонов с информацией о биомами"""
        polygons_path = self.root_dir / "data/polygons/biome_polygons.gpkg"
        if not polygons_path.exists():
            raise FileNotFoundError(f"Файл полигонов не найден: {polygons_path}")

        gdf = gpd.read_file(polygons_path)

        # Проверяем необходимые колонки
        if 'biome_name' not in gdf.columns:
            raise ValueError("Отсутствует колонка 'biome_name' в полигонах")

        # Если нет колонки 'fid', используем индекс как fid
        if 'fid' not in gdf.columns:
            logger.info("Колонка 'fid' не найдена, используем индекс как fid")
            gdf = gdf.reset_index().rename(columns={'index': 'fid'})

        # Валидация биомов
        invalid_biomes = set(gdf['biome_name']) - set(self.config['biomes'].keys())
        if invalid_biomes:
            raise ValueError(f"Неизвестные биомы: {invalid_biomes}")

        logger.info(f"Загружено {len(gdf)} полигонов")
        logger.info(f"Столбцы в данных: {list(gdf.columns)}")
        return gdf

    def find_corrected_data(self) -> Tuple[Path, Path]:
        """Поиск скорректированных данных"""
        corrected_path = self.root_dir / self.config['data']['intermediate'] / "corrected_data"

        if not corrected_path.exists():
            raise FileNotFoundError(f"Путь не существует: {corrected_path}")

        ms_files = list(corrected_path.glob("*_MS.tif"))
        pan_files = list(corrected_path.glob("*_PAN.tif"))

        if not ms_files:
            raise FileNotFoundError(f"MS данные не найдены в {corrected_path}")
        if not pan_files:
            raise FileNotFoundError(f"PAN данные не найдены в {corrected_path}")

        return ms_files[0], pan_files[0]

    def _clip_single_polygon(self, polygons_gdf: gpd.GeoDataFrame, idx: int,
                             ms_src: rasterio.DatasetReader,
                             pan_src: rasterio.DatasetReader) -> Optional[ClippedPolygon]:
        """Обрезка одного полигона"""
        poly_row = polygons_gdf.iloc[idx]
        poly_geom = poly_row.geometry
        biome_name = poly_row['biome_name']
        fid = poly_row['fid']

        # Получаем CRS из GeoDataFrame, а не из геометрии
        crs = polygons_gdf.crs

        try:
            # Обрезка многоканального MS
            with rasterio.vrt.WarpedVRT(ms_src, crs=crs) as vrt:
                ms_data, transform = mask(vrt, [poly_geom], crop=True, filled=False)
                if ms_data.size == 0:
                    logger.warning(f"Пустой полигон {fid}")
                    return None

            # Обрезка PAN
            with rasterio.vrt.WarpedVRT(pan_src, crs=crs) as vrt:
                pan_data, pan_transform = mask(vrt, [poly_geom], crop=True, filled=False)
                if pan_data.size == 0:
                    return None

            return ClippedPolygon(
                biome_name=biome_name,
                fid=fid,  # Используем fid вместо poly_id
                ms_data=ms_data,  # 6-канальный MS
                pan_data=pan_data[0],  # 1-канальный PAN
                transform=transform,
                crs=crs
            )

        except Exception as e:
            logger.error(f"Ошибка обрезки полигона {fid}: {e}")
            return None

    def _export_polygon_files(self, polygon_data: ClippedPolygon):
        """
        Экспорт полигона в многоканальные файлы
        {fid:02d}_MS.tif и {fid:02d}_PAN.tif
        """
        biome_path = self.export_path / polygon_data.biome_name

        # Базовый профиль
        base_profile = {
            'driver': 'GTiff',
            'crs': polygon_data.crs,
            'transform': polygon_data.transform,
            'dtype': 'float32',
            'compress': 'DEFLATE'
        }

        # Экспорт многоканального MS
        ms_profile = base_profile.copy()
        ms_profile.update({
            'height': polygon_data.ms_data.shape[1],
            'width': polygon_data.ms_data.shape[2],
            'count': polygon_data.ms_data.shape[0]  # 6 каналов
        })

        ms_filename = f"{polygon_data.fid:02d}_MS.tif"  # Используем fid
        ms_path = biome_path / ms_filename

        with rasterio.open(ms_path, 'w', **ms_profile) as dst:
            dst.write(polygon_data.ms_data)
            # Установка описаний каналов
            band_descriptions = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
            for i, desc in enumerate(band_descriptions, 1):
                dst.set_band_description(i, desc)

        # Экспорт PAN
        pan_profile = base_profile.copy()
        pan_profile.update({
            'height': polygon_data.pan_data.shape[0],
            'width': polygon_data.pan_data.shape[1],
            'count': 1
        })

        pan_filename = f"{polygon_data.fid:02d}_PAN.tif"  # Используем fid
        pan_path = biome_path / pan_filename

        with rasterio.open(pan_path, 'w', **pan_profile) as dst:
            dst.write(polygon_data.pan_data.reshape(1, *polygon_data.pan_data.shape))
            dst.set_band_description(1, 'PAN')

    def clip_all_polygons(self, polygons_gdf: gpd.GeoDataFrame) -> Iterator[ClippedPolygon]:
        """
        Обрезка всех полигонов
        Returns: Генератор обрезанных данных
        """
        logger.info("Начало обрезки полигонов")

        ms_path, pan_path = self.find_corrected_data()

        with rasterio.open(ms_path) as ms_src, rasterio.open(pan_path) as pan_src:
            for idx in range(len(polygons_gdf)):
                polygon_data = self._clip_single_polygon(polygons_gdf, idx, ms_src, pan_src)
                if polygon_data:
                    # Экспорт
                    self._export_polygon_files(polygon_data)

                    yield polygon_data

    def load_polygon_from_files(self, biome_name: str, fid: int) -> Optional[ClippedPolygon]:
        """
        Загрузка полигона из экспортированных файлов
        """
        biome_path = self.export_path / biome_name

        try:
            # Загрузка многоканального MS
            ms_path = biome_path / f"{fid:02d}_MS.tif"  # Используем fid
            with rasterio.open(ms_path) as src:
                ms_data = src.read()
                transform = src.transform
                crs = src.crs

            # Загрузка PAN
            pan_path = biome_path / f"{fid:02d}_PAN.tif"  # Используем fid
            with rasterio.open(pan_path) as src:
                pan_data = src.read(1)

            return ClippedPolygon(
                biome_name=biome_name,
                fid=fid,  # Используем fid
                ms_data=ms_data,
                pan_data=pan_data,
                transform=transform,
                crs=crs
            )

        except Exception as e:
            logger.error(f"Ошибка загрузки полигона {fid}: {e}")
            return None


def main():
    """Пример использования"""
    clipper = ClippingTool()

    try:
        # Загрузка полигонов
        polygons_gdf = clipper.load_polygons()

        # Обработка с экспортом
        for polygon_data in clipper.clip_all_polygons(polygons_gdf):
            logger.info(f"Обработан полигон {polygon_data.fid} ({polygon_data.biome_name})")

    except Exception as e:
        logger.error(f'Критическая ошибка в исполнении: {e}')


if __name__ == "__main__":
    main()