import os
import warnings

from loguru import logger
from rich.logging import RichHandler
from dotenv import find_dotenv, load_dotenv

# Pydantic's langchain integration emits harmless serializer warnings on every
# structured-output call ("PydanticSerializationUnexpectedValue"). They don't
# affect results and only clutter stdout; silence them globally on import.
warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic(\..*)?$")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

_repo_env = os.path.join(CURRENT_DIR, "..", ".env")
if os.path.exists(_repo_env):
    load_dotenv(dotenv_path=_repo_env, override=False)
load_dotenv(find_dotenv(usecwd=True), override=False)

DEFAULT_LOGS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "logs"))
SLIDERS_LOGS_DIR = os.environ.get("SLIDERS_LOGS_DIR") or DEFAULT_LOGS_DIR

# Remove the default loguru handler
logger.remove()

# Add a new handler using RichHandler for console output
_console_handler_id = logger.add(
    RichHandler(markup=True, show_time=False),
    level="INFO",
    format="{message}",
    backtrace=True,
    diagnose=True,
)


def suppress_console_logging():
    """Remove the console handler, keeping only file logging."""
    global _console_handler_id
    if _console_handler_id is not None:
        logger.remove(_console_handler_id)
        _console_handler_id = None


def _ensure_log_dir(base_dir: str) -> str | None:
    if not base_dir:
        return None
    experiments_dir = os.path.join(base_dir, "experiments")
    try:
        os.makedirs(experiments_dir, exist_ok=True)
        return base_dir
    except OSError:
        return None


logs_dir = _ensure_log_dir(SLIDERS_LOGS_DIR) or _ensure_log_dir(os.path.join("/tmp", "sliders_logs"))

if logs_dir:
    try:
        logger.add(
            os.path.join(
                logs_dir, "experiments/debug_logs_{time:YYYYMMDD_HHmm}.jsonl"
            ),  # Use .jsonl extension for JSON Lines format
            level="DEBUG",
            format=(
                '{{"time": "{time:YYYY-MM-DD HH:mm:ss}", "level": "{level.name}", '
                '"name": "{name}", "message": "{message}"}}'
            ),
            rotation="5 MB",
            retention=2,
            backtrace=True,
            diagnose=True,
            enqueue=True,
        )
    except OSError:
        # If file logging cannot be configured, fall back to console-only logging.
        pass
