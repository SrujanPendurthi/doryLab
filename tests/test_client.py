import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from dory import client
from dory.client import ServerError, complete

OK = {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 2},
      "timings": {"prompt_per_second": 100.0, "predicted_per_second": 20.0, "draft_n": 4, "draft_n_accepted": 3}}


def fake(status, body):
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            self.send_response(status)
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}", seen


def test_complete_parses_timings(monkeypatch):
    monkeypatch.setattr(client, "CHAT_TEMPLATE_KWARGS", {})
    srv, url, seen = fake(200, OK)
    c = complete(url, "q", 8)
    srv.shutdown()
    assert (c.text, c.n_prompt, c.n_gen, c.gen_tps, c.prompt_tps, c.draft_n, c.draft_accepted) == ("hi", 5, 2, 20.0, 100.0, 4, 3)
    path, body = seen[0]
    assert path == "/v1/chat/completions" and body["max_tokens"] == 8 and body["temperature"] == 0 and body["seed"] == 0
    assert "chat_template_kwargs" not in body


def test_complete_sends_template_kwargs_when_set(monkeypatch):
    monkeypatch.setattr(client, "CHAT_TEMPLATE_KWARGS", {"enable_thinking": False})
    srv, url, seen = fake(200, OK)
    complete(url, "q", 8)
    srv.shutdown()
    assert seen[0][1]["chat_template_kwargs"] == {"enable_thinking": False}


def test_complete_http_error_raises():
    srv, url, _ = fake(400, {"error": "ctx"})
    with pytest.raises(ServerError):
        complete(url, "q", 8)
    srv.shutdown()


def test_complete_connection_error_raises():
    with pytest.raises(ServerError):
        complete("http://127.0.0.1:1", "q", 8, timeout=2)
