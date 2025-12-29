"""
Модуль статистического анализа результатов экспериментов.

Содержит этапы:
1. Дескриптивный анализ метрик классификации
2. Консенсусное ранжирование методов паншарпенинга
3. Анализ согласованности метрик (Kendall W)
4. Bootstrap-оценку устойчивости рангов
"""

from .decsript import (
    DescriptiveStatisticsAnalyzer
)

from .ranking import (
    RankingAnalyzer
)

from .agreement import (
    AgreementAnalyzer
)

from .boots import (
    BootstrapAnalyzer
)

__all__ = [
    # descriptive statistics
    "DescriptiveStatisticsAnalyzer",

    # consensus ranking
    "RankingAnalyzer",

    # agreement analysis
    "AgreementAnalyzer",

    # bootstrap stability
    "BootstrapAnalyzer"
]
