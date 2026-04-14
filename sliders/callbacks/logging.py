import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from sliders.log_utils import logger

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

if TYPE_CHECKING:
    pass


def _ensure_log_dir(base_dir: str) -> str:
    try:
        os.makedirs(base_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating log directory: {e}")
        pass
    return base_dir


def _load_experiment_id() -> str | None:
    """Safely obtain the current experiment id without triggering circular imports."""
    try:
        from sliders.globals import SlidersGlobal  # Local import to avoid circular dependency
    except Exception:
        return None
    return getattr(SlidersGlobal, "experiment_id", None)


class LoggingHandler(BaseCallbackHandler):
    """
    LangChain callback handler that logs every LLM call as a structured JSON object.

    Destination: logs/seq_agent_logs/<YYYYMMDD_HHMMSS_mmmmmmZ>.jsonl (per run)

    Required keys logged per user request:
      - prompt_file
      - system_message
      - user_message
      - llm_output
      - metadata (includes question and other metadata)

    Recommended extras:
      - timestamp, level, event, model, provider, temperature, max_tokens, top_p
      - response_ms, token_usage, request_id, session_id, run_id
    """

    def __init__(
        self,
        prompt_file: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.prompt_file = prompt_file or "unknown"
        self.metadata = metadata or {}
        self._start_times: Dict[str, float] = {}
        self._cached_experiment_id: str | None = _load_experiment_id()
        _ensure_log_dir(os.path.join(os.environ.get("SLIDERS_LOGS_DIR"), "prompt_logs"))
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%fZ")

        experiment_id = self._get_experiment_id()
        if experiment_id:
            # Include experiment id in filename per user preference; id already contains 'exp-' prefix
            filename = f"{experiment_id}.jsonl"
        else:
            logger.warning("Experiment id is not set, using timestamp instead")
            filename = f"{ts}.jsonl"
        self._run_log_path = os.path.join(os.environ.get("SLIDERS_LOGS_DIR"), "prompt_logs", filename)

    # ===== Utility =====
    def _log_path(self) -> str:
        return self._run_log_path

    def _append_jsonl(self, payload: Dict[str, Any]) -> None:
        try:
            with open(self._log_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            # Best-effort only; do not raise
            logger.error(f"Error appending JSONL: {e}")
            pass

    def _get_experiment_id(self) -> str | None:
        """Return the cached experiment id, refreshing if necessary."""
        if self._cached_experiment_id:
            return self._cached_experiment_id
        experiment_id = _load_experiment_id()
        if experiment_id:
            self._cached_experiment_id = experiment_id
        return experiment_id

    def _extract_messages(self, messages: List[BaseMessage]) -> Dict[str, str]:
        system_parts: List[str] = []
        user_parts: List[str] = []

        for m in messages or []:
            if isinstance(m, SystemMessage):
                system_parts.append(m.content or "")
            elif isinstance(m, HumanMessage):
                user_parts.append(m.content or "")
        return {
            "system_message": "\n\n".join(system_parts) if system_parts else "",
            "user_message": "\n\n".join(user_parts) if user_parts else "",
        }

    # ===== LLM callbacks =====
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._start_times[run_id] = time.time()
        provider = (serialized or {}).get("id")
        model = (serialized or {}).get("kwargs", {}).get("model")
        params = (serialized or {}).get("kwargs", {})

        # Attempt to reconstruct messages if available via kwargs
        messages: List[BaseMessage] = params.get("messages") or []
        msg_info = self._extract_messages(messages)

        system_message = msg_info.get("system_message", "")
        user_message = msg_info.get("user_message", "")

        if system_message == "" and user_message == "":
            user_message = "\n\n".join(prompts) if prompts else ""

        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": "INFO",
            "event": "llm_call_start",
            "prompt_file": self.prompt_file,
            "experiment_id": self._get_experiment_id(),
            "system_message": system_message,
            "user_message": user_message,
            "llm_output": None,
            "metadata": self.metadata,
            # Recommended
            "provider": provider,
            "model": model,
            "temperature": params.get("temperature"),
            "max_tokens": params.get("max_tokens") or params.get("max_completion_tokens"),
            "top_p": params.get("top_p"),
            "request_id": run_id,
            "parent_run_id": parent_run_id,
        }
        self._append_jsonl(payload)

    def on_llm_end(self, response, *, run_id: str, **kwargs: Any) -> None:
        end = time.time()
        start = self._start_times.pop(run_id, None)
        duration_ms = int((end - start) * 1000) if start is not None else None

        # Try to extract text and token usage
        try:
            generations = getattr(response, "generations", None)
            if generations and len(generations) > 0 and len(generations[0]) > 0:
                text = getattr(generations[0][0], "text", None)
            else:
                text = getattr(response, "content", None) or getattr(response, "text", None)
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            text = None

        token_usage = None
        try:
            usage = getattr(response, "llm_output", {}) or {}
            token_usage = usage.get("token_usage") or usage.get("usage")
        except Exception as e:
            logger.error(f"Error extracting token usage: {e}")
            pass

        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": "INFO",
            "event": "llm_call_end",
            "prompt_file": self.prompt_file,
            "experiment_id": self._get_experiment_id(),
            "system_message": None,
            "user_message": None,
            "llm_output": text,
            "metadata": self.metadata,
            # Recommended
            "response_ms": duration_ms,
            "token_usage": token_usage,
            "request_id": run_id,
        }
        self._append_jsonl(payload)

    def on_llm_error(self, error: BaseException, *, run_id: str, **kwargs: Any) -> None:
        end = time.time()
        start = self._start_times.pop(run_id, None)
        duration_ms = int((end - start) * 1000) if start is not None else None

        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": "ERROR",
            "event": "llm_call_error",
            "prompt_file": self.prompt_file,
            "experiment_id": self._get_experiment_id(),
            "system_message": None,
            "user_message": None,
            "llm_output": None,
            "metadata": self.metadata,
            "response_ms": duration_ms,
            "error": str(error),
            "request_id": run_id,
        }
        self._append_jsonl(payload)
