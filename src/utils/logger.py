import logging
import logging.handlers
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import yaml


class CustomFormatter(logging.Formatter):
    """Кастомный форматтер с временными метками"""

    def format(self, record):
        # Добавляем timestamp ко всем записям
        record.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        return super().format(record)


class MetricsFilter(logging.Filter):
    """Фильтр для отбора записей METRIC"""

    def filter(self, record: logging.LogRecord) -> bool:
        message = getattr(record, 'msg', '')
        if isinstance(message, str) and message.startswith('METRIC '):
            return True
        return False


class ResearchLogger:
    """
    Логгер для полного пайплайна с ротацией файлов
    """

    def __init__(self, config_path: Optional[str] = None, log_subdir: Optional[str] = None):
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.config = self._load_config(config_path)

        # CHANGED: создаем подпапку с датой или используем переданную
        if log_subdir:
            self.log_dir = self.root_dir / "logs" / log_subdir
        else:
            date_str = datetime.now().strftime('%Y-%m-%d')
            self.log_dir = self.root_dir / "logs" / date_str

        self._setup_logging()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Загружает конфигурацию или использует значения по умолчанию"""
        default_config = {
            'level': 'INFO',
            'console_level': 'INFO',
            'file_level': 'DEBUG',
            'rotation': {
                'max_size_mb': 10,
                'backup_count': 5,
                'when': 'D',
                'interval': 1,
            },
            'formats': {
                'detailed': '%(timestamp)s - %(name)-35s - %(levelname)-8s - %(message)s',
                'simple': '%(levelname)-8s %(message)s',
            }
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                    if 'logging' in user_config:
                        return self._deep_update(default_config, user_config['logging'])
            except Exception as e:
                print(f"Ошибка загрузки конфигурации: {e}. Использую настройки по умолчанию")

        return default_config

    def _deep_update(self, original: Dict, update: Dict) -> Dict:
        """Рекурсивно обновляет словарь конфигурации"""
        for key, value in update.items():
            if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                original[key] = self._deep_update(original[key], value)
            else:
                original[key] = value
        return original

    def _setup_logging(self):
        """Настраивает всю систему логирования"""
        # CHANGED: создаем директорию с датой
        self.log_dir.mkdir(parents=True, exist_ok=True)

        main_logger = logging.getLogger('research')
        main_logger.setLevel(logging.DEBUG)
        main_logger.handlers.clear()

        self._setup_console_handler(main_logger)
        self._setup_file_handlers(main_logger)

        main_logger.info("=" * 60)
        main_logger.info("СИСТЕМА ЛОГИРОВАНИЯ ИНИЦИАЛИЗИРОВАНА")
        main_logger.info(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        main_logger.info(f"Директория логов: {self.log_dir}")
        main_logger.info("=" * 60)

    def _setup_console_handler(self, logger: logging.Logger):
        """Настраивает вывод в консоль"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.config['console_level']))

        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)-8s [%(name)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    def _setup_file_handlers(self, logger: logging.Logger):
        """Настраивает файловые обработчики с ротацией"""
        rotation_config = self.config['rotation']

        # 1. Основной лог-файл (ВСЕ сообщения)
        self._setup_main_file_handler(logger, rotation_config)

        # 2. Лог-файл для метрик (только METRIC JSON)
        self._setup_metrics_file_handler(logger, rotation_config)

        # 3. Лог-файл для ошибок (WARNING и выше)
        self._setup_error_file_handler(logger, rotation_config)

    def _setup_main_file_handler(self, logger: logging.Logger, rotation_config: Dict):
        """Основной файл с детальной информацией"""
        # CHANGED: файл создается в подпапке с датой
        main_log_path = self.log_dir / "pipeline.log"

        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=main_log_path,
            when=rotation_config['when'],
            interval=rotation_config['interval'],
            backupCount=rotation_config['backup_count'],
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, self.config['file_level']))

        file_formatter = CustomFormatter(
            self.config['formats']['detailed']
        )
        file_handler.setFormatter(file_formatter)

        logger.addHandler(file_handler)

    def _setup_metrics_file_handler(self, logger: logging.Logger, rotation_config: Dict):
        """Файл для метрик в JSON формате"""
        # CHANGED: файл создается в подпапке с датой
        metrics_log_path = self.log_dir / "metrics.log"

        metrics_handler = logging.handlers.TimedRotatingFileHandler(
            filename=metrics_log_path,
            when=rotation_config['when'],
            interval=rotation_config['interval'],
            backupCount=rotation_config['backup_count'],
            encoding='utf-8'
        )
        metrics_handler.setLevel(logging.INFO)
        metrics_handler.addFilter(MetricsFilter())

        metrics_formatter = logging.Formatter('%(message)s')
        metrics_handler.setFormatter(metrics_formatter)

        logger.addHandler(metrics_handler)

    def _setup_error_file_handler(self, logger: logging.Logger, rotation_config: Dict):
        """Файл только для ошибок и предупреждений"""
        # CHANGED: файл создается в подпапке с датой
        error_log_path = self.log_dir / "errors.log"

        error_handler = logging.handlers.TimedRotatingFileHandler(
            filename=error_log_path,
            when=rotation_config['when'],
            interval=rotation_config['interval'],
            backupCount=rotation_config['backup_count'],
            encoding='utf-8'
        )
        error_handler.setLevel(logging.WARNING)

        error_formatter = CustomFormatter(
            '%(timestamp)s - %(name)s - %(levelname)s - %(message)s'
        )
        error_handler.setFormatter(error_formatter)

        logger.addHandler(error_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """Создает и возвращает именованный логгер для модуля"""
        return logging.getLogger(f'research.{name}')

    def log_metric(self, logger: logging.Logger, metric_name: str, value: float,
                   context: Optional[Dict] = None):
        """Логирует метрику в структурированном JSON формате"""
        metric_data = {
            'metric': metric_name,
            'value': float(value),
            'timestamp': datetime.now().isoformat(),
            'context': context or {}
        }

        logger.info(f"METRIC {json.dumps(metric_data, ensure_ascii=False)}")

    def log_module_start(self, logger: logging.Logger, module_name: str,
                         params: Optional[Dict] = None):
        """Логирует начало работы модуля"""
        logger.info("┌─ НАЧАЛО МОДУЛЯ: %s", module_name)
        if params:
            logger.debug("│ Параметры: %s", params)

    def log_module_end(self, logger: logging.Logger, module_name: str,
                       execution_time: Optional[float] = None):
        """Логирует завершение работы модуля"""
        if execution_time is not None:
            logger.info("└─ ЗАВЕРШЕНИЕ МОДУЛЯ: %s (время: %.2fс)", module_name, execution_time)
        else:
            logger.info("└─ ЗАВЕРШЕНИЕ МОДУЛЯ: %s", module_name)


# Глобальный экземпляр логгера
_research_logger = None


def setup_logging(config_path: Optional[str] = None, log_subdir: Optional[str] = None) -> ResearchLogger:
    """Инициализирует систему логирования с возможностью указать подпапку"""
    global _research_logger
    if _research_logger is None:
        _research_logger = ResearchLogger(config_path, log_subdir)
    return _research_logger


def get_logger(name: str) -> logging.Logger:
    """Возвращает именованный логгер"""
    global _research_logger
    if _research_logger is None:
        _research_logger = ResearchLogger()
    return _research_logger.get_logger(name)


def log_metric(logger: logging.Logger, metric_name: str, value: float,
               context: Optional[Dict] = None):
    """Логирует метрику"""
    global _research_logger
    if _research_logger is None:
        _research_logger = ResearchLogger()
    _research_logger.log_metric(logger, metric_name, value, context)


# CHANGED: добавлена функция для создания логгера с конкретной датой
def setup_logging_for_date(target_date: str, config_path: Optional[str] = None) -> ResearchLogger:
    """
    Создает логгер для указанной даты

    Args:
        target_date: Дата в формате 'YYYY-MM-DD'
        config_path: Путь к конфигурационному файлу
    """
    return setup_logging(config_path, target_date)