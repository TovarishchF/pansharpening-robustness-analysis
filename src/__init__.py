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

# Паншарпеннинг&Классификация <- Фабрики
from .pansharpening import pansharpening_factory
from .classification import classification_factory

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
