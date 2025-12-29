"""
Модуль визуализации результатов статистического анализа.

Содержит скрипты для построения графиков:
- дескриптивной статистики,
- консенсусного ранжирования,
- согласованности метрик (Kendall W),
- bootstrap-оценок устойчивости.
"""

from .vis_descriptive import (
    plot_boxplots,
)

from .vis_ranking import (
    plot_rank_heatmap,
    plot_borda_scores
)

from .vis_agreement import (
    plot_kendall_w
)

from .vis_bootstrap import (
    plot_bootstrap_ci,
    plot_topk_probability_by_biome
)

__all__ = [
    # descriptive
    "plot_boxplots",
    "plot_violin",

    # ranking
    "plot_rank_heatmap",
    "plot_borda_scores",

    # agreement
    "plot_kendall_w",

    # bootstrap
    "plot_bootstrap_ci",
    "plot_topk_probability_by_biome"
]
