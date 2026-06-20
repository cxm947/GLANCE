from __future__ import annotations
import json
import time
import logging
import os
from openai import OpenAI

logger = logging.getLogger(__name__)

class LLMClient:

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1",
                 model: str = "deepseek-chat", max_retries: int = 3,
                 retry_delay: float = 2.0,
                 response_format_enabled: bool = True,
                 extra_body: dict | None = None, seed: int | None = None):
        self.model = model
        self.seed = seed
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit_retry_delay = float(os.environ.get("EDL_RATE_LIMIT_RETRY_DELAY_SECONDS", "65"))
        self.response_format_enabled = response_format_enabled
        self.extra_body = extra_body or None

        _ua = os.environ.get("EDL_USER_AGENT")
        self.client = OpenAI(api_key=api_key, base_url=base_url,
                             default_headers=({"User-Agent": _ua} if _ua else None))
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0
        self.agent_call_log: list[dict] = []

    def chat(self, system_prompt: str, user_prompt: str,
             temperature: float = 0.3, max_tokens: int = 4096,
             extra_body: dict | None = None, agent: str | None = None) -> dict:

        if os.environ.get("EDL_MERGE_SYSTEM"):
            messages = [{"role": "user", "content": (system_prompt + "\n\n" + user_prompt) if system_prompt else user_prompt}]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        last_error = None
        eb = extra_body if extra_body is not None else self.extra_body

        for attempt in range(self.max_retries):
            try:
                request = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                if self.seed is not None:
                    request["seed"] = self.seed
                if self.response_format_enabled:
                    request["response_format"] = {"type": "json_object"}
                if eb:
                    request["extra_body"] = eb

                resp = self.client.chat.completions.create(**request)
                content = (resp.choices[0].message.content or "").strip()
                finish_reason = getattr(resp.choices[0], "finish_reason", None)

                if resp.usage:
                    self._total_input_tokens += resp.usage.prompt_tokens
                    self._total_output_tokens += resp.usage.completion_tokens
                    self.record_agent_call({
                        "agent": agent or "unknown",
                        "input_tokens": resp.usage.prompt_tokens,
                        "output_tokens": resp.usage.completion_tokens,
                        "total_tokens": getattr(resp.usage, "total_tokens", 0)
                        or (resp.usage.prompt_tokens + resp.usage.completion_tokens),
                        "finish_reason": finish_reason,
                        "max_tokens": max_tokens,
                    })
                self._call_count += 1

                if finish_reason == "length":
                    logger.warning(
                        "[%s] completion truncated (finish_reason=length, max_tokens=%d). "
                        "If thinking is ON, disable it for this agent or raise max_tokens.",
                        agent or "?", max_tokens,
                    )

                result = json.loads(content)
                logger.debug("LLM call #%d succeeded (attempt %d)", self._call_count, attempt + 1)
                return result

            except json.JSONDecodeError as e:
                last_error = e
                logger.warning("JSON parse error on attempt %d: %s\nRaw: %s",
                               attempt + 1, e, content[:500])

                result = self._try_extract_json(content)
                if result is not None:
                    return result

                result = self._salvage_json(content)
                if result is not None:
                    logger.warning("Salvaged truncated JSON (agent=%s)", agent)
                    return result
            except Exception as e:
                last_error = e
                if self.response_format_enabled and self._is_json_mode_unsupported_error(e):
                    logger.warning(
                        "Provider rejected API-level JSON mode; retrying with prompt-only JSON parsing."
                    )
                    self.response_format_enabled = False
                    continue
                logger.warning("API error on attempt %d: %s", attempt + 1, e)

            if attempt < self.max_retries - 1:
                delay = self._retry_delay_for_error(last_error, attempt)
                logger.info("Retrying in %.1fs...", delay)
                time.sleep(delay)

        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_error}")

    def chat_raw(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.3, max_tokens: int = 4096) -> str:

        if os.environ.get("EDL_MERGE_SYSTEM"):
            messages = [{"role": "user", "content": (system_prompt + "\n\n" + user_prompt) if system_prompt else user_prompt}]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        for attempt in range(self.max_retries):
            try:
                request = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if self.extra_body:
                    request["extra_body"] = self.extra_body
                resp = self.client.chat.completions.create(**request)
                if resp.usage:
                    self._total_input_tokens += resp.usage.prompt_tokens
                    self._total_output_tokens += resp.usage.completion_tokens
                self._call_count += 1
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self._retry_delay_for_error(e, attempt)
                    logger.info("Retrying raw chat in %.1fs...", delay)
                    time.sleep(delay)
                else:
                    raise

    def _retry_delay_for_error(self, error: Exception | None, attempt: int) -> float:
        message = str(error).lower() if error else ""
        if "429" in message or "rate limit" in message or "tpm limit" in message:
            return self.rate_limit_retry_delay * (attempt + 1)
        return self.retry_delay * (2 ** attempt)

    @staticmethod
    def _is_json_mode_unsupported_error(error: Exception) -> bool:
        message = str(error).lower()
        return (
            "json mode is not supported" in message
            or ("response_format" in message and "not support" in message)
            or ("response_format" in message and "unsupported" in message)
        )

    @staticmethod
    def _try_extract_json(text: str) -> dict | None:
        import re

        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        depth = 0
        start = None
        for i, c in enumerate(text):
            if c == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        start = None
        return None

    @staticmethod
    def _salvage_json(text: str) -> dict | None:
        if not text:
            return None
        start = text.find("{")
        if start < 0:
            return None
        s = text[start:]
        in_str = esc = False
        safe = None
        stack = 0
        for i, c in enumerate(s):
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c in "{[":
                stack += 1
            elif c in "}]":
                stack -= 1
                if stack == 0:
                    try:
                        return json.loads(s[: i + 1])
                    except json.JSONDecodeError:
                        pass
                safe = i + 1
            elif c == ",":
                safe = i
        if safe is None:
            return None
        candidate = s[:safe].rstrip().rstrip(",").rstrip()

        closers: list[str] = []
        in_str = esc = False
        for c in candidate:
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                closers.append("}")
            elif c == "[":
                closers.append("]")
            elif c in "}]" and closers:
                closers.pop()
        try:
            return json.loads(candidate + "".join(reversed(closers)))
        except json.JSONDecodeError:
            return None

    @property
    def usage_stats(self) -> dict:
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_calls": self._call_count,
        }

    def record_agent_call(self, record: dict) -> None:
        enriched = dict(record)
        enriched.setdefault("call_index", len(self.agent_call_log) + 1)
        self.agent_call_log.append(enriched)

    def clear_agent_call_log(self) -> None:
        self.agent_call_log = []
