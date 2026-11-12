import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

# Thinking-off switch for the chat template; Task 1 fills this in. Sent only when non-empty.
CHAT_TEMPLATE_KWARGS: dict = {}


class ServerError(RuntimeError):
    pass


@dataclass
class Completion:
    text: str
    n_prompt: int
    n_gen: int
    prompt_tps: float
    gen_tps: float
    seconds: float
    draft_n: int = 0
    draft_accepted: int = 0


def _post(url: str, path: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise ServerError(f"HTTP {e.code} from {path}: {e.read()[:500]!r}") from e
    except (OSError, ValueError) as e:  # URLError, timeouts, resets, bad JSON
        raise ServerError(f"{path} failed: {e!r}") from e


def complete(url: str, prompt: str, max_tokens: int, timeout: float = 900) -> Completion:
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "seed": 0}
    if CHAT_TEMPLATE_KWARGS:
        body["chat_template_kwargs"] = CHAT_TEMPLATE_KWARGS
    t0 = time.time()
    d = _post(url, "/v1/chat/completions", body, timeout)
    secs = time.time() - t0
    try:
        text = d["choices"][0]["message"]["content"] or ""
        u, t = d.get("usage", {}), d.get("timings", {})
        return Completion(text, u.get("prompt_tokens", 0), u.get("completion_tokens", 0),
                          t.get("prompt_per_second", 0.0), t.get("predicted_per_second", 0.0), secs,
                          t.get("draft_n", 0), t.get("draft_n_accepted", 0))
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        raise ServerError(f"malformed response: {d!r:.300}") from e


def embed_last(url: str, text: str) -> list[float]:
    d = _post(url, "/embedding", {"content": text}, 300)
    try:
        e = d[0]["embedding"] if isinstance(d, list) else d["embedding"]
        return e[0] if e and isinstance(e[0], list) else e  # pooled: [[...]] or [...]
    except (KeyError, IndexError, TypeError) as e:
        raise ServerError(f"malformed embedding response: {d!r:.300}") from e
