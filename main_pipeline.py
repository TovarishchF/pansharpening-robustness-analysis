"""
Главный сценарий исследовательского пайплайна паншарпенинга
"""

import sys
import os
from pathlib import Path

script_path = os.path.abspath(__file__)
root_dir = Path(script_path).parent.parent.parent
sys.path.insert(0, str(root_dir))

import yaml
import time

from src import (
    setup_logging,
    get_logger,

    AtmosphericCorrection,
    ClippingTool,

    pansharpening_factory,
    classification_factory,

    PansharpeningMetricsCalculator,
    MetricsCalculator,

    DescriptiveStatisticsAnalyzer,
    PansharpeningRankingAnalyzer,
    KendallAgreementAnalyzer,
    BootstrapRankingAnalyzer,

    plot_boxplots,
    plot_rank_heatmap,
    plot_borda_scores,
    plot_kendall_w,
    plot_bootstrap_ci,
    plot_topk_probability_by_biome,
)


# ======================================================================
# Utils
# ======================================================================

def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Пайллайн
def main():

    start_time = time.time()

    # --------------------------------------------------
    # Инициализация
    # --------------------------------------------------

    BASE_DIR = Path(__file__).resolve().parent
    CONFIG_PATH = BASE_DIR / "config.yaml"

    setup_logging()
    logger = get_logger("pipeline")

    logger.info("=== Запуск пайплайна паншарпенинга ===")

    # --------------------------------------------------
    # Загрузка конфига
    # --------------------------------------------------

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # --------------------------------------------------
    # Пути
    # --------------------------------------------------

    data_cfg = config["data"]

    raw_data_dir = BASE_DIR / data_cfg["raw"]
    processed_data_dir = BASE_DIR / data_cfg["processed"]
    intermediate_data_dir = BASE_DIR / data_cfg["intermediate"]

    processed_data_dir.mkdir(exist_ok=True, parents=True)
    intermediate_data_dir.mkdir(exist_ok=True, parents=True)

    polygons_path = BASE_DIR / data_cfg["clip_polygons"]

    # --------------------------------------------------
    # 1. Предобработка данных
    # --------------------------------------------------

    logger.info("Предобработка данных (Landsat)")

    ac = AtmosphericCorrection(
        config_path=CONFIG_PATH
    )

    scene_path = ac.find_raw_scene()
    if scene_path is None:
        logger.error("Сырая сцена Landsat не найдена")
        return

    corrected = ac.process_scene(scene_path)
    if corrected is None:
        logger.error("Ошибка атмосферной коррекции")
        return

    clipper = ClippingTool()

    try:
        polygons_gdf = clipper.load_polygons()
    except Exception as e:
        logger.error(f"Ошибка загрузки полигонов: {e}")
        return

    clipped_polygons = []

    try:
        for polygon_data in clipper.clip_all_polygons(polygons_gdf):
            clipped_polygons.append(polygon_data)
    except Exception as e:
        logger.error(f"Ошибка обрезки полигонов: {e}")
        return

    if not clipped_polygons:
        logger.error("Нет данных после обрезки по полигонам")
        return

    # --------------------------------------------------
    # 2. Паншарпенинг
    # --------------------------------------------------

    logger.info("Паншарпенинг")

    for method_name in pansharpening_factory.get_available_methods():
        # logger.info(f"Метод паншарпенинга: {method_name}")

        method = pansharpening_factory.create_method(
            method_name,
            config_path=CONFIG_PATH
        )

        method.process_all_biomes()
    
    pan_calculator = PansharpeningMetricsCalculator()
    pan_calculator.run_calculation()

    # --------------------------------------------------
    # 3. Классификация
    # --------------------------------------------------

    logger.info("Классификация")

    for clf_name in classification_factory.get_available_methods():
        # logger.info(f"Классификатор: {clf_name}")

        classifier = classification_factory.create(
            clf_name,
            config_path=CONFIG_PATH
        )

        classifier.process_all_polygons()

    class_calculator = MetricsCalculator()
    class_calculator.run()

    # --------------------------------------------------
    # 4. Статистический анализ
    # --------------------------------------------------

    logger.info("Статистический анализ")

    desc = DescriptiveStatisticsAnalyzer()
    desc.run()

    pansh = PansharpeningRankingAnalyzer()
    pansh.run()

    kendal = KendallAgreementAnalyzer()
    kendal.run()

    boots = BootstrapRankingAnalyzer()
    boots.run()

    # --------------------------------------------------
    # 5. Визуализация
    # --------------------------------------------------

    logger.info("Визуализация")

    plot_boxplots()
    plot_rank_heatmap()
    plot_borda_scores()
    plot_kendall_w()
    plot_bootstrap_ci()
    plot_topk_probability_by_biome()

    end_time = time.time()
    execution_time = end_time - start_time

    logger.info(f"=== Пайплайн успешно завершён за {execution_time:.2f} секунд ===")



# ======================================================================
# Входная точка
# ======================================================================

if __name__ == "__main__":
    main()
