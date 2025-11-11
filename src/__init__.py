"""
Пакет исследовательского пайплайна паншарпенинга

Основные модули:
- utils: Система логирования и утилиты
- preprocessing: Предобработка данных (атмосферная коррекция, обрезка)
"""

from .utils import setup_logging, get_logger, log_metric
from .preprocessing import AtmosphericCorrection, CorrectedData, ClippingTool, ClippedPolygon

__all__ = [
    # Утилиты
    'setup_logging',
    'get_logger',
    'log_metric',

    # Предобработка
    'AtmosphericCorrection',
    'CorrectedData',
    'ClippingTool',
    'ClippedPolygon'
]