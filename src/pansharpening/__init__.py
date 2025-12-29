from dataclasses import dataclass
import numpy as np
from typing import Dict, Any, Optional


@dataclass
class PansharpeningResult:
    """Результат паншарпенинга"""
    sharpened_image: np.ndarray
    method_name: str
    parameters: Dict[str, Any]
    execution_time: float
    metrics: Optional[Dict[str, float]] = None


# Импорт всех методов паншарпенинга
from src.pansharpening.cs import BroveyPansharpening, BroveyHistogramPansharpening, PCAPansharpening
from src.pansharpening.mra import HPFPansharpening, ATWTPansharpening, SFIMPansharpening
from src.pansharpening.model_based import GramSchmidtPansharpening, GramSchmidtAdaptivePansharpening, PRACSPansharpening

# импорт калькулятора метрик
from src.pansharpening.metrics import PansharpeningMetricsCalculator

class PansharpeningMethodFactory:
    """
    Фабрика для создания методов паншарпенинга
    """

    # Реестр всех доступных методов
    _methods = {
        # CS методы
        'brovey': BroveyPansharpening,
        'brovey_histogram': BroveyHistogramPansharpening,
        'pca': PCAPansharpening,

        # MRA методы
        'hpf': HPFPansharpening,
        'atwt': ATWTPansharpening,
        'sfim': SFIMPansharpening,

        # Model-Based методы
        'gram_schmidt': GramSchmidtPansharpening,
        'gram_schmidt_adaptive': GramSchmidtAdaptivePansharpening,
        'pracs': PRACSPansharpening,
    }

    @classmethod
    def create_method(cls, method_name: str, **kwargs):
        """
        Создает экземпляр метода паншарпенинга по имени
        """
        method_name = method_name.lower()

        if method_name not in cls._methods:
            available_methods = ', '.join(cls._methods.keys())
            raise ValueError(
                f"Метод паншарпенинга '{method_name}' не найден. "
                f"Доступные методы: {available_methods}"
            )

        method_class = cls._methods[method_name]

        return method_class(**kwargs)

    @classmethod
    def get_available_methods(cls):
        """
        Возвращает список доступных методов паншарпенинга
        """
        return list(cls._methods.keys())

    @classmethod
    def get_method_categories(cls):
        """
        Возвращает методы, сгруппированные по категориям
        """
        return {
            'cs': ['brovey', 'brovey_histogram', 'pca'],
            'mra': ['hpf', 'atwt', 'sfim'],
            'model_based': ['gram_schmidt', 'gram_schmidt_adaptive', 'pracs']
        }


# Создаем глобальный экземпляр фабрики для удобства использования
pansharpening_factory = PansharpeningMethodFactory()

__all__ = [
    # Фабрика
    'PansharpeningMethodFactory',
    'pansharpening_factory',

    # Структуры данных
    'PansharpeningResult',

    # CS методы
    'BroveyPansharpening',
    'BroveyHistogramPansharpening',
    'PCAPansharpening',

    # MRA методы
    'HPFPansharpening',
    'ATWTPansharpening',
    'SFIMPansharpening',

    # Model-Based методы
    'GramSchmidtPansharpening',
    'GramSchmidtAdaptivePansharpening',
    'PRACSPansharpening',

    # Метрики
    'PansharpeningMetricsCalculator',
]