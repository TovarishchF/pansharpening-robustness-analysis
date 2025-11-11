"""
Модуль предобработки данных для исследовательского пайплайна паншарпенинга

Включает:
- Атмосферную коррекцию (AtmosphericCorrection)
- Обрезку по полигонам (ClippingTool)
- Структуры данных для передачи между этапами (CorrectedData, ClippedPolygon)
"""

from .atm_corr_tool import AtmosphericCorrection, CorrectedData
from .clipping_tool import ClippingTool, ClippedPolygon

__all__ = [
    'AtmosphericCorrection',
    'CorrectedData',
    'ClippingTool',
    'ClippedPolygon'
]