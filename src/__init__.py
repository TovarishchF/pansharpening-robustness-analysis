"""
Пакет исследовательского пайплайна паншарпенинга.

Этапы пайплайна:
- preprocessing     : предобработка данных
- pansharpening     : методы паншарпенинга
- classification    : пиксельная классификация
- stat_analysis     : статистический анализ
- visualisation     : визуализация результатов
"""

# Утилиты
from .utils import (
    setup_logging,
    get_logger,
)

# Предобработка
from .preprocessing import (
    AtmosphericCorrection,
    ClippingTool,
)

# Паншарпеннинг&Классификация <- Фабрики, Метрики
from .pansharpening import pansharpening_factory,  PansharpeningMetricsCalculator
from .classification import classification_factory, MetricsCalculator

# Статистический анализ
from .statistical_analysis import (
    DescriptiveStatisticsAnalyzer,
    PansharpeningRankingAnalyzer,
    KendallAgreementAnalyzer,
    BootstrapRankingAnalyzer,
)

# Визуализация
from .visualisation import (
    plot_boxplots,
    plot_rank_heatmap,
    plot_borda_scores,
    plot_kendall_w,
    plot_bootstrap_ci,
    plot_topk_probability_by_biome,
)

# Публичный API 
__all__ = [
    # утилиты
    "setup_logging",
    "get_logger",

    # предобработка
    "AtmosphericCorrection",
    "ClippingTool",

    # фабрики
    "pansharpening_factory",
    "classification_factory",

    # метрики классификации и пш
    "PansharpeningMetricsCalculator",
    "MetricsCalculator",

    # стат анализ
    "DescriptiveStatisticsAnalyzer",
    "PansharpeningRankingAnalyzer",
    "KendallAgreementAnalyzer",
    "BootstrapRankingAnalyzer",

    # визуализация
    "plot_boxplots",
    "plot_rank_heatmap",
    "plot_borda_scores",
    "plot_kendall_w",
    "plot_bootstrap_ci",
    "plot_topk_probability_by_biome",
]
