"""
Модуль методов паншарпенинга на основе Model-Based подходов

Включает:
- Классический метод Gram-Schmidt (GramSchmidtPansharpening)
- Адаптивный метод Gram-Schmidt (GramSchmidtAdaptivePansharpening)
- Метод PRACS (PRACSPansharpening)
- Структуры данных для результатов паншарпенинга (PansharpeningResult)
"""

from .gs import GramSchmidtPansharpening, PansharpeningResult
from .gs_a import GramSchmidtAdaptivePansharpening
from .pracs import PRACSPansharpening

__all__ =[
    'GramSchmidtPansharpening',
    'GramSchmidtAdaptivePansharpening',
    'PRACSPansharpening',
    'PansharpeningResult'
]