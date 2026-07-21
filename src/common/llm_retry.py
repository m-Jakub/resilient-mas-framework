"""
Retry helpers for LLM calls to handle transient network failures.
"""

from __future__ import annotations

from typing import Any, Callable

from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter


_RETRYABLE_MESSAGE_SNIPPETS = (
    "timed out",
    "timeout",
    "connection",
    "temporarily unavailable",
    "connection reset",
    "connection aborted",
    "remote disconnected",
    "name resolution",
    "network is unreachable",
)


def _is_retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, (OSError, TimeoutError, ConnectionError)):
        return True
    message = str(exc).lower()
    return any(snippet in message for snippet in _RETRYABLE_MESSAGE_SNIPPETS)


def invoke_with_retry(
    invoke_fn: Callable[..., Any],
    *args: Any,
    operation: str = "llm.invoke",
    max_attempts: int = 5,
    min_wait: float = 2.0,
    max_wait: float = 20.0,
    **kwargs: Any,
) -> Any:
    """
    Invoke an LLM call with retry logic for transient connectivity errors.

    Args:
        invoke_fn: The callable to invoke (e.g., llm.invoke, chain.invoke)
        *args: Positional args forwarded to invoke_fn
        operation: Label used in retry logs
        max_attempts: Maximum attempts before failing
        min_wait: Minimum backoff in seconds
        max_wait: Maximum backoff in seconds
        **kwargs: Keyword args forwarded to invoke_fn
    """

    def _log_before_sleep(retry_state):
        exc = retry_state.outcome.exception()
        attempt = retry_state.attempt_number
        next_sleep = getattr(retry_state.next_action, "sleep", None)
        if next_sleep is not None:
            print(
                f"[Retry] {operation} failed on attempt {attempt}/{max_attempts}: {exc}. "
                f"Retrying in {next_sleep:.1f}s..."
            )
        else:
            print(
                f"[Retry] {operation} failed on attempt {attempt}/{max_attempts}: {exc}. "
                "Retrying..."
            )

    retrying = Retrying(
        retry=retry_if_exception(_is_retryable_exception),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=min_wait, max=max_wait),
        reraise=True,
        before_sleep=_log_before_sleep,
    )

    for attempt in retrying:
        with attempt:
            return invoke_fn(*args, **kwargs)
