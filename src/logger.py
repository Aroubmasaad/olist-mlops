import logging
from src.config import PROJECT_ROOT

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
formatter = logging.Formatter(LOG_FORMAT)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(
    LOG_DIR / "app.log",
    encoding="utf-8",
)
file_handler.setFormatter(formatter)
logger = logging.getLogger("olist_api")
logger.setLevel(logging.INFO)

logger.addHandler(console_handler)
logger.addHandler(file_handler)
