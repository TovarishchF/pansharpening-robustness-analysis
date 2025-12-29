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
    PansharpeningRankingAnalyzer
)

from .agreement import (
    KendallAgreementAnalyzer
)

from .boots import (
    BootstrapRankingAnalyzer
)

__all__ = [
    # descriptive statistics
    "DescriptiveStatisticsAnalyzer",

    # consensus ranking
    "PansharpeningRankingAnalyzer",

    # agreement analysis
    "KendallAgreementAnalyzer",

    # bootstrap stability
    "BootstrapRankingAnalyzer"
]
