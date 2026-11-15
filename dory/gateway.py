"""OpenAI chat gateway backed by one llama-server with per-request loop counts."""
import argparse
import json
import signal
import sys
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dory.policy import decide_k, load_policy
from dory.server import Server, free_port, server_args


def last_user_text(messages: list[dict]) -> str:
    """Read the last user turn, ignoring tool replies and non-text parts."""
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(part["text"] for part in content
                               if isinstance(part, dict) and part.get("type") == "text"
                               and isinstance(part.get("text"), str))
            return ""
    return ""


class _Stats:
    """Observe JSON/SSE for logging without changing the forwarded bytes."""
    def __init__(self):
        self.usage = {}
        self.timings = {}
        self.pending = b""

    def observe(self, body):
        try:
            value = json.loads(body)
            if isinstance(value, dict):
                if isinstance(value.get("usage"), dict):
                    self.usage.update(value["usage"])
                if isinstance(value.get("timings"), dict):
                    self.timings.update(value["timings"])
        except (ValueError, UnicodeError):
            pass

    def stream(self, chunk):
        self.pending += chunk
        while b"\n" in self.pending:
            line, self.pending = self.pending.split(b"\n", 1)
            if line.startswith(b"data:"):
                self.observe(line[5:].strip())


class Gateway(ThreadingHTTPServer):
    def __init__(self, port=8083, policy_path=Path("results/policy_text.joblib"),
                 gguf=Path("models/dory-q5.gguf"), ctx=16384,
                 log_path=Path("results/logs/gateway.log")):
        policy = load_policy(policy_path)
        if policy["feature_set"] != "text":
            raise ValueError("gateway requires a text policy")
        self.log_path = Path(log_path)
        self.log_lock = threading.Lock()
        self.decision_lock = threading.Lock()

        # Cache only by text. The policy is fixed for this gateway's lifetime.
        @lru_cache(maxsize=256)
        def decision(text):
            return decide_k(policy, text)

        self.decision = decision
        self.upstream = None
        super().__init__(("127.0.0.1", port), _Handler)
        try:
            upstream_port = free_port()
            args = server_args(gguf, 4, upstream_port, slots=1, ctx_per_slot=ctx,
                               runtime_loops=True, spec_draft_loops=1)
            self.upstream = Server(args, upstream_port)
        except BaseException:
            self.server_close()
            raise

    def server_close(self):
        try:
            super().server_close()
        finally:
            if self.upstream is not None:
                self.upstream.stop()
                self.upstream = None

    def log_request(self, k, stats):
        # Logging must never fail a request, even with incomplete upstream metrics.
        try:
            usage, timings = stats.usage, stats.timings

            def metric(values, key, fallback="-"):
                value = values.get(key)
                return fallback if value is None else value

            draft, accepted = timings.get("draft_n"), timings.get("draft_n_accepted")
            acceptance = (accepted / draft if isinstance(draft, (int, float)) and draft > 0
                          and isinstance(accepted, (int, float)) else "-")
            line = (f"time={datetime.now(timezone.utc).isoformat()} k={k} "
                    f"prompt_tokens={metric(usage, 'prompt_tokens', metric(timings, 'prompt_n'))} "
                    f"generated_tokens={metric(usage, 'completion_tokens', metric(timings, 'predicted_n'))} "
                    f"tok/s={metric(timings, 'predicted_per_second')} "
                    f"draft_acceptance={acceptance}\n")
            with self.log_lock:
                try:
                    sys.stderr.write(line)
                    sys.stderr.flush()
                except OSError:
                    pass
                try:
                    self.log_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.log_path.open("a") as log:
                        log.write(line)
                except OSError:
                    pass
        except Exception:
            pass


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # one metrics line instead of the standard access log

    def do_GET(self):
        if self.path == "/v1/models":
            self._json(200, {"data": [{"id": "dory", "object": "model"}]})
        else:
            self._json(404, {"error": "not found"})

    def _json(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        stats, k = _Stats(), "-"
        try:
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                messages = body.get("messages", [])
                if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
                    raise ValueError("messages must be a list of objects")
                kwargs = body.setdefault("chat_template_kwargs", {})
                if not isinstance(kwargs, dict):
                    raise ValueError("chat_template_kwargs must be an object")
                kwargs.setdefault("enable_thinking", False)
                text = last_user_text(messages)
            except (ValueError, AttributeError, TypeError) as exc:
                self._json(400, {"error": str(exc)})
                return
            # MPS classifier is not thread-safe. Lock outside the cache so concurrent
            # tool steps cannot race into duplicate inference on a cache miss.
            with self.server.decision_lock:
                k = self.server.decision(text)
            body["n_loops"] = int(k)  # always override, including a client-supplied null
            req = urllib.request.Request(self.server.upstream.url + self.path,
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                upstream = urllib.request.urlopen(req)
            except urllib.error.HTTPError as exc:
                upstream = exc  # forward upstream validation errors verbatim too
            except (urllib.error.URLError, OSError):
                self._json(502, {"error": "upstream unavailable"})
                return
            with upstream:
                self.send_response(upstream.status)
                self.send_header("Content-Type", upstream.headers.get("Content-Type", "application/json"))
                self.end_headers()
                if body.get("stream"):
                    while chunk := upstream.read1(65536):
                        stats.stream(chunk)
                        self.wfile.write(chunk)
                        self.wfile.flush()
                else:
                    data = upstream.read()
                    stats.observe(data)
                    self.wfile.write(data)
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # leaving the upstream context closes the generation request
        finally:
            self.server.log_request(k, stats)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--policy", type=Path, default=Path("results/policy_text.joblib"))
    parser.add_argument("--gguf", type=Path, default=Path("models/dory-q5.gguf"))
    parser.add_argument("--ctx", type=int, default=16384)
    args = parser.parse_args(argv)

    def stop(signum, frame):
        raise KeyboardInterrupt

    old_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    gateway = None
    try:
        gateway = Gateway(port=args.port, policy_path=args.policy, gguf=args.gguf, ctx=args.ctx)
        gateway.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if gateway is not None:
            gateway.server_close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
