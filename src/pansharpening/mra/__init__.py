"""
Модуль методов паншарпенинга на основе Multi-Resolution Analysis (MRA)

Включает:
- Метод High-Pass Filtering (HPFPansharpening)
- Метод À Trous Wavelet Transform (ATWTPansharpening)
- Метод Smoothing Filter-based Intensity Modulation (SFIMPansharpening)
- Структуры данных для результатов паншарпенинга (PansharpeningResult)
"""

from .hpf import HPFPansharpening, PansharpeningResult
from .atwt import ATWTPansharpening
from .sfim import SFIMPansharpening

__all__ = [
    'HPFPansharpening',
    'ATWTPansharpening',
    'SFIMPansharpening',
    'PansharpeningResult'
]