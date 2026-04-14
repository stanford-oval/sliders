import hashlib
import inspect
import json
import logging
import os
from urllib.parse import urlparse
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any, Optional

import redis.asyncio as redis
from openai import RateLimitError
from overrides import override
from tenacity import before_sleep_log

from langchain_core.callbacks.base import Callbacks
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables.retry import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from sliders.callbacks.logging import LoggingHandler
from sliders.log_utils import logger

from dotenv import find_dotenv, load_dotenv

# Load credentials from a .env file. We first try the repo-relative path (works
# for editable/dev installs), then fall back to walking upward from the current
# working directory (works for pip-installed users running from any project).
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_repo_env = os.path.join(CURRENT_DIR, "..", "..", ".env")
if os.path.exists(_repo_env):
    load_dotenv(dotenv_path=_repo_env, override=False)
load_dotenv(find_dotenv(usecwd=True), override=False)

# Per-task credential overrides (set via set_llm_credentials, read by get_llm_client)
_azure_api_key_var: ContextVar[str | None] = ContextVar("_azure_api_key", default=None)
_azure_endpoint_var: ContextVar[str | None] = ContextVar("_azure_endpoint", default=None)
_openai_api_key_var: ContextVar[str | None] = ContextVar("_openai_api_key", default=None)
_openai_base_url_var: ContextVar[str | None] = ContextVar("_openai_base_url", default=None)
_provider_var: ContextVar[str | None] = ContextVar("_provider", default=None)


def set_llm_credentials(
    api_key: str | None = None,
    endpoint: str | None = None,
    *,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    provider: str | None = None,
):
    """Set LLM credentials for the current async task context.

    Azure OpenAI is the default provider. Pass ``openai_api_key`` (and
    optionally ``openai_base_url``) to route the current task to the
    public OpenAI API (or any OpenAI-compatible endpoint) without
    touching environment variables.
    """
    if api_key is not None:
        _azure_api_key_var.set(api_key)
    if endpoint is not None:
        _azure_endpoint_var.set(endpoint)
    if openai_api_key is not None:
        _openai_api_key_var.set(openai_api_key)
        if provider is None:
            provider = "openai"
    if openai_base_url is not None:
        _openai_base_url_var.set(openai_base_url)
    if provider is not None:
        _provider_var.set(provider.lower())


# Redis client for caching
_redis_client: Optional[redis.Redis] = None


async def get_redis_client() -> redis.Redis | None:
    """Get or create the Redis client used for LLM response caching.

    Host and port are read from ``REDIS_HOST`` / ``REDIS_PORT`` environment
    variables (also loaded from a ``.env`` file), defaulting to
    ``localhost:6379``. If the connection fails, caching is silently disabled.
    """
    global _redis_client
    if _redis_client is None:
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        try:
            _redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
            await _redis_client.ping()
            logger.info(f"Connected to Redis for LLM caching at {redis_host}:{redis_port}")
        except Exception as e:
            logger.debug(f"Redis unavailable at {redis_host}:{redis_port} ({e}); LLM cache disabled.")
            _redis_client = None
    return _redis_client


def generate_cache_key(prompt_content: str, model_name: str, response_format: str = "") -> str:
    """Generate a cache key for the LLM request, including the output class name."""
    # Create a deterministic hash of the inputs
    content = f"{prompt_content}:{model_name}:{response_format}"
    return f"llm_cache:{hashlib.md5(content.encode()).hexdigest()}"


def _build_rate_limiter(slow_rate_limiter: bool) -> InMemoryRateLimiter:
    if slow_rate_limiter:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=1,
            max_bucket_size=1,
        )
    else:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=40,
            max_bucket_size=40,
        )

    return rate_limiter


def _is_openai_compatible_azure_endpoint(endpoint: str | None) -> bool:
    """Detect Azure AI Foundry OpenAI-compatible endpoints (e.g. .../openai/v1/)."""
    if not endpoint:
        return False
    try:
        parsed = urlparse(endpoint)
    except Exception:
        return False
    normalized_path = (parsed.path or "").rstrip("/")
    return normalized_path.endswith("/openai/v1")


def get_llm_client(*, model: str, slow_rate_limiter: bool = False, provider: str | None = None, **kwargs):
    """Get an LLM client with optional caching support across providers.

    Defaults to Azure OpenAI, but can be routed to OpenAI-compatible endpoints (e.g. vLLM)
    by passing provider information via kwargs or environment variables.
    """

    provider_value = (
        kwargs.pop("provider", provider)
        or _provider_var.get()
        or os.getenv("SLIDERS_LLM_PROVIDER", "azure")
    )
    provider_value = provider_value.lower()

    if "gpt-5" in model:
        kwargs.pop("temperature", None)

    shared_kwargs = {"model": model}
    max_retries = kwargs.pop("max_retries", 3)
    timeout = kwargs.pop("timeout", 200)

    if provider_value == "azure":
        rate_limiter = kwargs.pop("rate_limiter", _build_rate_limiter(slow_rate_limiter))
        api_key = kwargs.pop("api_key", None) or _azure_api_key_var.get() or os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = (
            kwargs.pop("azure_endpoint", None) or _azure_endpoint_var.get() or os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        api_version = kwargs.pop("api_version", os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"))

        # Azure AI Foundry's OpenAI-compatible endpoint should be called via ChatOpenAI(base_url=...),
        # not AzureChatOpenAI deployment routing.
        if _is_openai_compatible_azure_endpoint(azure_endpoint):
            chat_kwargs = {
                "api_key": api_key,
                "base_url": azure_endpoint,
                "max_retries": max_retries,
            }
            if timeout is not None:
                chat_kwargs["timeout"] = timeout

            return CachedChatOpenAI(
                rate_limiter=rate_limiter,
                **chat_kwargs,
                **shared_kwargs,
                **kwargs,
            )

        return CachedAzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            rate_limiter=rate_limiter,
            max_retries=max_retries,
            timeout=timeout,
            **shared_kwargs,
            **kwargs,
        )

    if provider_value in {"vllm", "openai"}:
        if provider_value == "vllm":
            default_api_key = os.getenv("VLLM_API_KEY")
            default_base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        else:
            default_api_key = _openai_api_key_var.get() or os.getenv("OPENAI_API_KEY")
            default_base_url = (
                _openai_base_url_var.get()
                or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            )
        api_key = kwargs.pop("api_key", default_api_key or "EMPTY")
        base_url = kwargs.pop("base_url", default_base_url)

        chat_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "max_retries": max_retries,
        }
        if timeout is not None:
            chat_kwargs["timeout"] = timeout

        return CachedChatOpenAI(
            **chat_kwargs,
            **shared_kwargs,
            **kwargs,
        )

    raise ValueError(f"Unsupported LLM provider '{provider_value}'.")


class CachedLLMMixin:
    async def _cached_agenerate(
        self,
        call_super: Callable[..., Awaitable[LLMResult]],
        messages: list[list[BaseMessage]],
        stop: Optional[list[str]] = None,
        callbacks: Callbacks = None,
        *,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        run_name: Optional[str] = None,
        run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> LLMResult:
        serialize_messages = json.dumps(
            [[msg.model_dump(mode="json") for msg in message] for message in messages], sort_keys=True
        )
        response_format = kwargs.get("response_format", None)
        if response_format:
            if not isinstance(response_format, dict):
                response_format = {
                    "type": "pydantic",
                    "keys": list(inspect.signature(response_format).parameters.keys()),
                }

            response_format = json.dumps(response_format)
        cache_key = generate_cache_key(serialize_messages, self.model_name, response_format)

        redis_client = None
        if kwargs.get("use_cache", True) and self.temperature == 0.0:
            redis_client = await get_redis_client()

        if redis_client:
            try:
                cached_result = await redis_client.get(cache_key)
                if cached_result:
                    logger.debug("Cache hit")
                    cached_data = json.loads(cached_result)

                    # Emit synthetic callbacks to log cached response
                    try:
                        # Build serialized stub similar to LangChain provider info
                        serialized = {
                            "id": "cache",
                            "kwargs": {
                                "model": self.model_name,
                                "temperature": getattr(self, "temperature", None),
                                "top_p": getattr(self, "top_p", None),
                                "max_tokens": getattr(self, "max_tokens", None),
                                # Pass original messages so LoggingHandler can extract system/user text
                                "messages": [m for m in messages[0]] if messages else [],
                            },
                        }

                        # Derive prompts list best-effort (not strictly needed since messages are provided)
                        prompts = []

                        # Merge metadata with cached flag
                        cache_metadata: dict[str, Any] = {}
                        if metadata:
                            cache_metadata.update(metadata)
                        cache_metadata.update(
                            {
                                "cached": True,
                                "cache_key": cache_key,
                            }
                        )

                        if callbacks:
                            if len(callbacks.handlers) > 0:
                                for handler in callbacks.handlers:
                                    if isinstance(handler, LoggingHandler):
                                        handler.metadata.update(cache_metadata)
                                        # Ensure we have a run_id string
                                        run_id_str = str(run_id) if run_id is not None else str(uuid.uuid4())

                                        # Start event
                                        handler.on_llm_start(serialized, prompts, run_id=run_id_str)

                                        # Build a lightweight response-like object for on_llm_end
                                        class _ResponseLike:
                                            def __init__(self, text: str):
                                                self.generations = [[type("Gen", (), {"text": text})()]]
                                                self.llm_output = {
                                                    "token_usage": {
                                                        "prompt_tokens": 0,
                                                        "completion_tokens": 0,
                                                        "total_tokens": 0,
                                                    }
                                                }

                                        # Reconstruct text from cached LLMResult
                                        # Prefer the first generation text if available
                                        cached_text: str | None = None
                                        try:
                                            if (
                                                isinstance(cached_data, dict)
                                                and cached_data.get("generations")
                                                and len(cached_data["generations"]) > 0
                                                and len(cached_data["generations"][0]) > 0
                                                and isinstance(cached_data["generations"][0][0], dict)
                                            ):
                                                cached_text = cached_data["generations"][0][0].get("text")
                                        except Exception:
                                            cached_text = None

                                        if cached_text is None:
                                            # Fallback: attempt to serialize the first generation object to string
                                            try:
                                                cached_text = json.dumps(cached_data)
                                            except Exception:
                                                cached_text = ""

                                        handler.on_llm_end(_ResponseLike(cached_text), run_id=run_id_str)
                    except Exception as log_e:
                        logger.warning(f"Synthetic cache logging error: {log_e}")

                    return LLMResult.model_validate(cached_data)
            except Exception as e:
                logger.warning(f"Cache read error: {e}")

        logger.debug("Cache miss")

        async def _invoke_llm() -> LLMResult:
            return await call_super(
                messages,
                stop,
                callbacks,
                tags=tags,
                metadata=metadata,
                run_name=run_name,
                run_id=run_id,
                **kwargs,
            )

        max_attempts = max(1, getattr(self, "max_retries", 1))
        retrying = AsyncRetrying(
            retry=retry_if_exception_type(RateLimitError),
            wait=wait_exponential_jitter(initial=1, max=30, exp_base=2, jitter=1),
            stop=stop_after_attempt(max_attempts),
            reraise=True,
            before_sleep=before_sleep_log(logger, logging.WARNING, exc_info=True),
        )

        try:
            async for attempt in retrying:
                with attempt:
                    output = await _invoke_llm()
                    break
        except RateLimitError as exc:
            logger.error(f"Exceeded retry attempts due to rate limiting: {exc}")
            raise

        if redis_client:
            try:
                await redis_client.set(cache_key, json.dumps(output.model_dump(mode="json")))
            except Exception as e:
                logger.warning(f"Cache write error: {e}")

        return output


class CachedAzureChatOpenAI(CachedLLMMixin, AzureChatOpenAI):
    """Azure-specific client with shared caching mixin."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override
    async def agenerate(
        self,
        messages: list[list[BaseMessage]],
        stop: Optional[list[str]] = None,
        callbacks: Callbacks = None,
        *,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        run_name: Optional[str] = None,
        run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> LLMResult:
        return await self._cached_agenerate(
            super().agenerate,
            messages,
            stop,
            callbacks,
            tags=tags,
            metadata=metadata,
            run_name=run_name,
            run_id=run_id,
            **kwargs,
        )


class CachedChatOpenAI(CachedLLMMixin, ChatOpenAI):
    """OpenAI-compatible client (e.g., vLLM) with shared caching mixin."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override
    async def agenerate(
        self,
        messages: list[list[BaseMessage]],
        stop: Optional[list[str]] = None,
        callbacks: Callbacks = None,
        *,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        run_name: Optional[str] = None,
        run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> LLMResult:
        return await self._cached_agenerate(
            super().agenerate,
            messages,
            stop,
            callbacks,
            tags=tags,
            metadata=metadata,
            run_name=run_name,
            run_id=run_id,
            **kwargs,
        )
