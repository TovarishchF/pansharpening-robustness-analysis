"""
Модуль методов паншарпенинга на основе Component Substitution (CS)

Включает:
- Классический метод Brovey (BroveyPansharpening)
- Brovey с гистограммным согласованием (BroveyHistogramPansharpening)
- Метод главных компонент (PCAPansharpening)
- Структуры данных для результатов паншарпенинга (PansharpeningResult)
"""

from .brovey import BroveyPansharpening, PansharpeningResult
from .bt_h import BroveyHistogramPansharpening
from .pca import PCAPansharpening

__all__ = [
    'BroveyPansharpening',
    'BroveyHistogramPansharpening',
    'PCAPansharpening',
    'PansharpeningResult'
]
