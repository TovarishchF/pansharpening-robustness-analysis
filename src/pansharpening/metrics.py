# pansharpening-robustness-analysis/src/pansharpening/metrics.py
import yaml
import json
import numpy as np
import rasterio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim
import cv2
from scipy.ndimage import sobel
import warnings
from sklearn.preprocessing import MinMaxScaler

from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)
warnings.filterwarnings('ignore')


class PansharpeningMetricsCalculator:
    """
    Калькулятор метрик качества для методов паншарпенинга
    с корректной нормализацией и интегральным рейтингом
    """

    def __init__(self, config_path: str = None):
        self.root_dir = Path(__file__).parent.parent.parent
        if config_path is None:
            config_path = self.root_dir / 'config.yaml'
        self.config = self._load_config(config_path)
        self._setup_paths()

        # Определение типов метрик
        self.maximize_metrics = ['QNR', 'SC', 'SSIM', 'Entropy', 'GS', 'UIQI']
        self.minimize_metrics = ['SAM', 'ERGAS']

    def _load_config(self, config_path: str) -> dict:
        """Загрузка конфигурации"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_paths(self):
        """Настройка путей для ввода и вывода"""
        self.pansharpening_base = self.root_dir / self.config['data']['processed'] / "pansharpening"
        self.clipped_polygons_base = self.root_dir / self.config['data']['intermediate'] / "clipped_polygons"
        self.output_dir = self.root_dir / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _resize_to_match(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Приведение изображений к одинаковому размеру"""
        if img1.shape[-2:] != img2.shape[-2:]:
            target_height = min(img1.shape[-2], img2.shape[-2])
            target_width = min(img1.shape[-1], img2.shape[-1])

            if len(img1.shape) == 3:
                resized1 = np.zeros((img1.shape[0], target_height, target_width), dtype=img1.dtype)
                for i in range(img1.shape[0]):
                    resized1[i] = cv2.resize(img1[i], (target_width, target_height), interpolation=cv2.INTER_AREA)
            else:
                resized1 = cv2.resize(img1, (target_width, target_height), interpolation=cv2.INTER_AREA)

            if len(img2.shape) == 3:
                resized2 = np.zeros((img2.shape[0], target_height, target_width), dtype=img2.dtype)
                for i in range(img2.shape[0]):
                    resized2[i] = cv2.resize(img2[i], (target_width, target_height), interpolation=cv2.INTER_AREA)
            else:
                resized2 = cv2.resize(img2, (target_width, target_height), interpolation=cv2.INTER_AREA)

            return resized1, resized2
        return img1, img2

    def calculate_sam(self, ms_image: np.ndarray, sharpened_image: np.ndarray) -> float:
        """
        Вычисление Spectral Angle Mapper (SAM)
        """
        try:
            ms_image, sharpened_image = self._resize_to_match(ms_image, sharpened_image)

            # Приведение к форме (высота, ширина, каналы)
            if len(ms_image.shape) == 3 and ms_image.shape[0] < ms_image.shape[-1]:
                ms_image = np.moveaxis(ms_image, 0, -1)
            if len(sharpened_image.shape) == 3 and sharpened_image.shape[0] < sharpened_image.shape[-1]:
                sharpened_image = np.moveaxis(sharpened_image, 0, -1)

            ms_flat = ms_image.reshape(-1, ms_image.shape[-1])
            sharpened_flat = sharpened_image.reshape(-1, sharpened_image.shape[-1])

            # Удаление нулевых векторов
            mask = np.all(ms_flat > 0, axis=1) & np.all(sharpened_flat > 0, axis=1)
            ms_flat = ms_flat[mask]
            sharpened_flat = sharpened_flat[mask]

            if len(ms_flat) == 0:
                return 0.0

            # Вычисление косинуса угла между векторами
            dot_product = np.sum(ms_flat * sharpened_flat, axis=1)
            norm_ms = np.linalg.norm(ms_flat, axis=1)
            norm_sharpened = np.linalg.norm(sharpened_flat, axis=1)

            # Избегаем деления на ноль
            valid_mask = (norm_ms > 0) & (norm_sharpened > 0)
            if not np.any(valid_mask):
                return 0.0

            cos_theta = dot_product[valid_mask] / (norm_ms[valid_mask] * norm_sharpened[valid_mask])
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            angles = np.arccos(cos_theta)
            sam_value = np.mean(angles)

            return float(sam_value)

        except Exception as e:
            logger.debug(f"Ошибка в SAM: {e}")
            return np.pi / 2  # Максимальное значение угла

    def calculate_ergas(self, ms_image: np.ndarray, sharpened_image: np.ndarray) -> float:
        """
        Вычисление ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse)
        """
        try:
            ms_image, sharpened_image = self._resize_to_match(ms_image, sharpened_image)

            n_bands = ms_image.shape[0]
            ergas_sum = 0.0
            valid_bands = 0

            # Параметры из конфигурации
            h = self.config['pansharpening_metrics'].get('pan_resolution', 15)
            l = self.config['pansharpening_metrics'].get('ms_resolution', 30)

            for i in range(n_bands):
                ms_band = ms_image[i].astype(np.float64)
                sharpened_band = sharpened_image[i].astype(np.float64)

                # Пропускаем каналы с нулевой дисперсией
                if np.std(ms_band) == 0 or np.std(sharpened_band) == 0:
                    continue

                rmse = np.sqrt(np.mean((ms_band - sharpened_band) ** 2))
                mean_ms = np.mean(ms_band)

                if mean_ms > 0:
                    ergas_sum += (rmse / mean_ms) ** 2
                    valid_bands += 1

            if valid_bands == 0:
                return float('inf')

            ergas_value = 100 * (h / l) * np.sqrt(ergas_sum / valid_bands)
            return float(ergas_value)

        except Exception as e:
            logger.debug(f"Ошибка в ERGAS: {e}")
            return float('inf')

    def calculate_uiqi(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Вычисление Universal Image Quality Index (UIQI)
        """
        try:
            img1, img2 = self._resize_to_match(img1, img2)
            img1 = img1.astype(np.float64)
            img2 = img2.astype(np.float64)

            if img1.size == 0 or img2.size == 0:
                return 0.0

            mu1 = np.mean(img1)
            mu2 = np.mean(img2)
            sigma1 = np.std(img1)
            sigma2 = np.std(img2)
            sigma12 = np.cov(img1.flatten(), img2.flatten())[0, 1]

            if sigma1 == 0 or sigma2 == 0:
                return 0.0

            # Компоненты UIQI
            correlation = sigma12 / (sigma1 * sigma2)
            luminance = (2 * mu1 * mu2) / (mu1 ** 2 + mu2 ** 2)
            contrast = (2 * sigma1 * sigma2) / (sigma1 ** 2 + sigma2 ** 2)

            uiqi_value = correlation * luminance * contrast
            return float(uiqi_value)

        except Exception as e:
            logger.debug(f"Ошибка в UIQI: {e}")
            return 0.0

    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Вычисление Structural Similarity Index (SSIM)
        """
        try:
            img1, img2 = self._resize_to_match(img1, img2)

            if len(img1.shape) == 3:
                ssim_values = []
                for i in range(img1.shape[0]):
                    channel_ssim = ssim(img1[i], img2[i], data_range=img1[i].max() - img1[i].min())
                    ssim_values.append(channel_ssim)
                return float(np.mean(ssim_values))
            else:
                return float(ssim(img1, img2, data_range=img1.max() - img1.min()))

        except Exception as e:
            logger.debug(f"Ошибка в SSIM: {e}")
            return 0.0

    def calculate_gradient_similarity(self, pan_image: np.ndarray, sharpened_image: np.ndarray) -> float:
        """
        Вычисление Gradient Similarity (GS)
        """
        try:
            if len(sharpened_image.shape) == 3:
                sharpened_gray = np.mean(sharpened_image, axis=0)
            else:
                sharpened_gray = sharpened_image

            pan_image, sharpened_gray = self._resize_to_match(pan_image, sharpened_gray)

            # Нормализация
            pan_norm = cv2.normalize(pan_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            sharpened_norm = cv2.normalize(sharpened_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Вычисление градиентов
            grad_x_pan = sobel(pan_norm, axis=1)
            grad_y_pan = sobel(pan_norm, axis=0)
            grad_pan = np.sqrt(grad_x_pan ** 2 + grad_y_pan ** 2)

            grad_x_sharp = sobel(sharpened_norm, axis=1)
            grad_y_sharp = sobel(sharpened_norm, axis=0)
            grad_sharp = np.sqrt(grad_x_sharp ** 2 + grad_y_sharp ** 2)

            C = 0.001
            numerator = 2 * grad_pan * grad_sharp + C
            denominator = grad_pan ** 2 + grad_sharp ** 2 + C

            gs_map = numerator / denominator
            return float(np.nanmean(gs_map))

        except Exception as e:
            logger.debug(f"Ошибка в GS: {e}")
            return 0.0

    def calculate_spatial_correlation(self, pan_image: np.ndarray, sharpened_image: np.ndarray) -> float:
        """
        Вычисление Spatial Correlation (SC)
        """
        try:
            if len(sharpened_image.shape) == 3:
                sharpened_gray = np.mean(sharpened_image, axis=0)
            else:
                sharpened_gray = sharpened_image

            pan_image, sharpened_gray = self._resize_to_match(pan_image, sharpened_gray)

            correlation_matrix = np.corrcoef(pan_image.flatten(), sharpened_gray.flatten())

            if correlation_matrix.size >= 4:
                correlation = correlation_matrix[0, 1]
            else:
                correlation = 0.0

            if np.isnan(correlation):
                return 0.0

            return float(correlation)

        except Exception as e:
            logger.debug(f"Ошибка в SC: {e}")
            return 0.0

    def calculate_entropy(self, image: np.ndarray) -> float:
        """
        Вычисление энтропии изображения
        """
        try:
            if len(image.shape) == 3:
                entropies = []
                for channel in range(image.shape[0]):
                    channel_data = image[channel]
                    channel_norm = cv2.normalize(channel_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    hist, _ = np.histogram(channel_norm, bins=256, range=(0, 255))
                    prob = hist / hist.sum()
                    prob = prob[prob > 0]
                    if len(prob) > 0:
                        entropy = -np.sum(prob * np.log2(prob))
                        entropies.append(entropy)
                return float(np.mean(entropies)) if entropies else 0.0
            else:
                image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                hist, _ = np.histogram(image_norm, bins=256, range=(0, 255))
                prob = hist / hist.sum()
                prob = prob[prob > 0]
                if len(prob) > 0:
                    return float(-np.sum(prob * np.log2(prob)))
                return 0.0

        except Exception as e:
            logger.debug(f"Ошибка в энтропии: {e}")
            return 0.0

    def calculate_qnr(self, ms_image: np.ndarray, pan_image: np.ndarray, sharpened_image: np.ndarray) -> float:
        """
        Вычисление QNR (Quality with No Reference)
        """
        try:
            n_bands = ms_image.shape[0]

            # Расчет D_lambda (спектральное искажение)
            d_lambda_sum = 0.0
            count_pairs = 0

            for i in range(n_bands):
                for j in range(i + 1, n_bands):
                    uiqi_ms = self.calculate_uiqi(ms_image[i], ms_image[j])
                    uiqi_ps = self.calculate_uiqi(sharpened_image[i], sharpened_image[j])

                    d_lambda_sum += abs(uiqi_ms - uiqi_ps)
                    count_pairs += 1

            if count_pairs == 0:
                return 0.0

            qnr_params = self.config['pansharpening_metrics']['qnr_params']
            p = qnr_params.get('p', 1)
            d_lambda = (d_lambda_sum / count_pairs) ** (1 / p)

            # Расчет D_S (пространственное искажение)
            d_s_sum = 0.0
            valid_bands = 0

            for i in range(n_bands):
                ms_band_resized = cv2.resize(ms_image[i], (pan_image.shape[1], pan_image.shape[0]),
                                             interpolation=cv2.INTER_CUBIC)

                uiqi_ms_pan = self.calculate_uiqi(ms_band_resized, pan_image)
                uiqi_ps_pan = self.calculate_uiqi(sharpened_image[i], pan_image)

                if not np.isnan(uiqi_ms_pan) and not np.isnan(uiqi_ps_pan):
                    d_s_sum += abs(uiqi_ms_pan - uiqi_ps_pan)
                    valid_bands += 1

            if valid_bands == 0:
                return 0.0

            q = qnr_params.get('q', 1)
            d_s = (d_s_sum / valid_bands) ** (1 / q)

            alpha = qnr_params.get('alpha', 1)
            beta = qnr_params.get('beta', 1)

            d_lambda = max(0.0, min(1.0, d_lambda))
            d_s = max(0.0, min(1.0, d_s))

            qnr = (1 - d_lambda) ** alpha * (1 - d_s) ** beta
            return float(max(0.0, min(1.0, qnr)))

        except Exception as e:
            logger.debug(f"Ошибка в QNR: {e}")
            return 0.0

    def load_reference_images(self, poly_id: int, biome_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Загрузка исходных MS и PAN изображений для полигона
        """
        try:
            clipped_dir = self.clipped_polygons_base / biome_name

            # Загрузка MS изображения
            ms_path = clipped_dir / f"{poly_id:02d}_MS.tif"
            if not ms_path.exists():
                logger.warning(f"MS файл не найден: {ms_path}")
                return None, None

            with rasterio.open(ms_path) as src:
                ms_image = src.read()

            # Загрузка PAN изображения
            pan_path = clipped_dir / f"{poly_id:02d}_PAN.tif"
            if not pan_path.exists():
                logger.warning(f"PAN файл не найден: {pan_path}")
                return None, None

            with rasterio.open(pan_path) as src:
                pan_image = src.read(1)

            logger.debug(f"Загружены изображения: MS {ms_image.shape}, PAN {pan_image.shape}")
            return ms_image, pan_image

        except Exception as e:
            logger.error(f"Ошибка загрузки для poly_id={poly_id}, biome={biome_name}: {e}")
            return None, None

    def calculate_all_metrics(self, sharpened_path: str, biome_name: str) -> Dict[str, float]:
        """
        Вычисление всех метрик для одного паншарпенного изображения
        """
        try:
            filename = Path(sharpened_path).stem
            poly_id = int(filename.split('_')[0])

            # Загрузка паншарпенного изображения
            with rasterio.open(sharpened_path) as src:
                sharpened_image = src.read()

            # Загрузка эталонных изображений
            ms_image, pan_image = self.load_reference_images(poly_id, biome_name)
            if ms_image is None or pan_image is None:
                return {}

            metrics = {'poly_id': poly_id, 'image_path': sharpened_path}

            # Вычисление метрик в зависимости от конфигурации
            metric_config = self.config['pansharpening_metrics']['metrics_to_calculate']

            if "GS" in metric_config:
                metrics['GS'] = self.calculate_gradient_similarity(pan_image, sharpened_image)

            if "SC" in metric_config:
                metrics['SC'] = self.calculate_spatial_correlation(pan_image, sharpened_image)

            if "Entropy" in metric_config:
                metrics['Entropy'] = self.calculate_entropy(sharpened_image)

            if "QNR" in metric_config:
                metrics['QNR'] = self.calculate_qnr(ms_image, pan_image, sharpened_image)

            if "SAM" in metric_config:
                metrics['SAM'] = self.calculate_sam(ms_image, sharpened_image)

            if "ERGAS" in metric_config:
                metrics['ERGAS'] = self.calculate_ergas(ms_image, sharpened_image)

            if "SSIM" in metric_config:
                metrics['SSIM'] = self.calculate_ssim(ms_image, sharpened_image)

            if "UIQI" in metric_config:
                metrics['UIQI'] = self.calculate_uiqi(ms_image, sharpened_image)

            logger.debug(
                f"Метрики для {filename}: { {k: v for k, v in metrics.items() if k not in ['poly_id', 'image_path']} }")
            return metrics

        except Exception as e:
            logger.error(f"Ошибка вычисления метрик для {sharpened_path}: {e}")
            return {}

    def _is_numeric_metric(self, value: Any) -> bool:
        """Проверка, является ли значение числовой метрикой"""
        return isinstance(value, (int, float, np.number)) and not np.isnan(value)

    def normalize_metrics_per_polygon(self, metrics_by_polygon: Dict[int, List[Dict]]) -> Dict[int, List[Dict]]:
        """
        Нормализация метрик в рамках каждого тестового полигона
        """
        normalized_metrics = {}

        for poly_id, metrics_list in metrics_by_polygon.items():
            if not metrics_list:
                continue

            # Собираем числовые метрики
            numeric_metrics = {}
            for metric_name in metrics_list[0].keys():
                if metric_name not in ['poly_id', 'image_path', 'method_name']:
                    values = [m[metric_name] for m in metrics_list
                              if metric_name in m and self._is_numeric_metric(m[metric_name])]
                    if values:
                        numeric_metrics[metric_name] = np.array(values)

            # Нормализация согласно методологии
            normalized_list = []
            for metrics in metrics_list:
                normalized = {
                    'poly_id': poly_id,
                    'image_path': metrics['image_path'],
                    'method_name': metrics.get('method_name', 'unknown')
                }

                for metric_name, value in metrics.items():
                    if (metric_name not in ['poly_id', 'image_path', 'method_name'] and
                            self._is_numeric_metric(value) and metric_name in numeric_metrics):

                        values = numeric_metrics[metric_name]
                        min_val, max_val = np.min(values), np.max(values)

                        if max_val > min_val:
                            if metric_name in self.maximize_metrics:
                                # Нормализация для метрик максимизации
                                normalized_val = (value - min_val) / (max_val - min_val)
                            elif metric_name in self.minimize_metrics:
                                # Нормализация для метрик минимизации
                                normalized_val = (max_val - value) / (max_val - min_val)
                            else:
                                # По умолчанию - максимизация
                                normalized_val = (value - min_val) / (max_val - min_val)
                        else:
                            normalized_val = 1.0  # если все значения одинаковы

                        normalized[metric_name] = float(normalized_val)

                normalized_list.append(normalized)

            normalized_metrics[poly_id] = normalized_list

        return normalized_metrics

    def calculate_integral_rating(self, normalized_metrics: Dict[int, List[Dict]], biome_name: str) -> Dict[
        int, Dict[str, float]]:
        """
        Расчет интегрального рейтинга методов (ИРМ) для каждого полигона
        """
        weights = self.config['pansharpening_metrics']['weights'].get(biome_name, {})
        integral_ratings = {}

        for poly_id, metrics_list in normalized_metrics.items():
            polygon_ratings = {}

            for metrics in metrics_list:
                method_name = metrics['method_name']
                integral_rating = 0.0
                total_weight = 0.0

                for metric_name, value in metrics.items():
                    if (metric_name not in ['poly_id', 'image_path', 'method_name'] and
                            self._is_numeric_metric(value)):
                        weight = weights.get(metric_name, 1.0)
                        integral_rating += value * weight
                        total_weight += weight

                if total_weight > 0:
                    # Нормализация интегрального рейтинга в диапазон [0, 1]
                    polygon_ratings[method_name] = integral_rating / total_weight
                else:
                    polygon_ratings[method_name] = 0.0

            integral_ratings[poly_id] = polygon_ratings

        return integral_ratings

    def validate_normalization(self, normalized_metrics: Dict[int, List[Dict]]) -> bool:
        """
        Валидация нормализации - проверяет, что все значения в диапазоне [0, 1]
        """
        for poly_id, metrics_list in normalized_metrics.items():
            for metrics in metrics_list:
                for metric_name, value in metrics.items():
                    if (metric_name not in ['poly_id', 'image_path', 'method_name'] and
                            self._is_numeric_metric(value)):
                        if value < 0 or value > 1:
                            logger.warning(f"Ненормализованное значение: {metric_name} = {value} "
                                           f"в полигоне {poly_id}, метод {metrics['method_name']}")
                            return False
        return True

    def process_pansharpening_methods(self) -> Dict[str, Any]:
        """
        Обработка всех методов паншарпенинга
        """
        logger.info("Начало вычисления метрик паншарпенинга")

        results = {
            'raw_metrics': {},
            'normalized_metrics': {},
            'integral_ratings': {},
            'validation_passed': True
        }

        # Поиск всех биомов
        biome_paths = [f for f in self.pansharpening_base.iterdir() if f.is_dir()]

        for biome_path in biome_paths:
            biome_name = biome_path.name
            logger.info(f"Обработка биома: {biome_name}")

            results['raw_metrics'][biome_name] = {}
            results['normalized_metrics'][biome_name] = {}
            results['integral_ratings'][biome_name] = {}

            metrics_by_polygon = {}
            method_metrics = {}

            # Обработка методов паншарпенинга
            method_paths = [f for f in biome_path.iterdir() if f.is_dir()]

            for method_path in method_paths:
                method_name = method_path.name
                logger.info(f"  Метод: {method_name}")

                method_metrics[method_name] = []
                image_paths = list(method_path.glob("*.tif"))

                for image_path in image_paths:
                    metrics = self.calculate_all_metrics(str(image_path), biome_name)
                    if metrics:
                        metrics['method_name'] = method_name
                        method_metrics[method_name].append(metrics)

                        # Группировка по полигонам
                        poly_id = metrics['poly_id']
                        if poly_id not in metrics_by_polygon:
                            metrics_by_polygon[poly_id] = []
                        metrics_by_polygon[poly_id].append(metrics)

            # Сохранение сырых метрик
            results['raw_metrics'][biome_name] = method_metrics

            # Нормализация и валидация
            if metrics_by_polygon:
                normalized_metrics = self.normalize_metrics_per_polygon(metrics_by_polygon)
                results['normalized_metrics'][biome_name] = normalized_metrics

                # Валидация нормализации
                if not self.validate_normalization(normalized_metrics):
                    results['validation_passed'] = False
                    logger.warning(f"Валидация нормализации не пройдена для биома {biome_name}")

                # Интегральный рейтинг
                integral_ratings = self.calculate_integral_rating(normalized_metrics, biome_name)
                results['integral_ratings'][biome_name] = integral_ratings

        return results

    def save_results(self, results: Dict):
        """Сохранение результатов в JSON"""
        output_path = self.output_dir / "pansharpening_metrics.json"

        # Конвертация numpy типов для сериализации
        def convert_types(obj):
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj

        serializable_results = convert_types(results)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Результаты сохранены в: {output_path}")

    def run_calculation(self):
        """Основной метод запуска вычисления метрик"""
        try:
            logger.info("Запуск вычисления метрик паншарпенинга")
            results = self.process_pansharpening_methods()
            self.save_results(results)

            if results['validation_passed']:
                logger.info("Вычисление метрик завершено успешно, нормализация валидирована")
            else:
                logger.warning("Вычисление метрик завершено с предупреждениями по нормализации")

        except Exception as e:
            logger.error(f"Критическая ошибка при вычислении метрик: {e}")
            raise


def main():
    """Основная функция выполнения"""
    try:
        calculator = PansharpeningMetricsCalculator()
        calculator.run_calculation()
    except Exception as e:
        logger.error(f"Ошибка выполнения: {e}")
        raise


if __name__ == "__main__":
    main()