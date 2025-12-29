from dataclasses import dataclass
from typing import Dict, Any, Optional, Type
import numpy as np

@dataclass
class ClassificationResult:
    """
    Результат классификации одного полигона
    """
    biome_name: str
    poly_id: int
    pansharp_method: str
    classification_method: str
    classified_data: np.ndarray
    transform: Any
    crs: Any
    profile: Dict[str, Any]

# Train/Test split 
from src.classification.tt_split import (
    TTSplitter,
    TrainTestSplit,
)

# Метрики классификации
from src.classification.metrics import MetricsCalculator

# Импорт всех методов классификации
from src.classification.maxlike import MaximumLikelihoodClassifier
from src.classification.rf import RandomForestImageClassifier
from src.classification.xgb import XGBoostImageClassifier


class ClassificationMethodFactory:
    """
    Фабрика для создания классификаторов
    """

    _methods: Dict[str, Type] = {
        "maximum_likelihood": MaximumLikelihoodClassifier,
        "random_forest": RandomForestImageClassifier,
        "xgboost": XGBoostImageClassifier,
    }

    @classmethod
    def create(
        cls,
        method_name: str,
        *,
        config_path: Optional[str] = None,
        **kwargs
    ):
        """
        Создает классификатор по имени метода
        """
        method_name = method_name.lower()

        if method_name not in cls._methods:
            available = ", ".join(cls._methods.keys())
            raise ValueError(
                f"Классификатор '{method_name}' не найден. "
                f"Доступные методы: {available}"
            )

        classifier_cls = cls._methods[method_name]

        # Все классификаторы уже принимают config_path
        if config_path is not None:
            return classifier_cls(config_path=config_path, **kwargs)

        return classifier_cls(**kwargs)

    @classmethod
    def get_available_methods(cls):
        """
        Список доступных классификаторов
        """
        return list(cls._methods.keys())


# Создаем глобальный экземпляр фабрики для удобства использования
classification_factory = ClassificationMethodFactory()

__all__ = [
    # Фабрика
    "ClassificationMethodFactory",
    "classification_factory",

    # Разделитель наборот TT
    "TTSplitter",

    # Структура данных
    "ClassificationResult",
    "TrainTestSplit",

    # Методы классификации
    "MaximumLikelihoodClassifier",
    "RandomForestImageClassifier",
    "XGBoostImageClassifier",

    #Метрики
    "MetricsCalculator",
]