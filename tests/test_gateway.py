import http.client
import json
import signal
import socket
import struct
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from dory import gateway as G


REPLY = b'{ "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}, "timings": {"predicted_per_second": 20.0, "draft_n": 4, "draft_n_accepted": 3} }'
SSE = [b'data: {"choices":[{"delta":{"content":"h',
       b'i"}}]}\r\n\r\ndata: {"choices":[]}\n\n',
       b'data: {"usage":{"prompt_tokens":5,"completion_tokens":2},"timings":{"predicted_per_second":20.0,"draft_n":4,"draft_n_accepted":3}}\n\n',
       b'data: [DONE]\n\n']


@contextmanager
def serving(server):
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.fixture
def rig(monkeypatch, tmp_path):
    seen, instances, decisions = [], [], []
    disconnected = threading.Event()
    state = {"status": 200, "body": REPLY, "sse": SSE}

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body))
            self.send_response(state["status"])
            self.send_header("Content-Type", "text/event-stream" if body.get("stream") else "application/json")
            self.end_headers()
            try:
                if body.get("stream"):
                    for index, chunk in enumerate(state["sse"]):
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        if index == 0 and "stream_gate" in state:
                            state["stream_gate"].wait(5)
                    if state.get("endless"):
                        while True:
                            self.wfile.write(b": keepalive\n\n" * 8192)
                            self.wfile.flush()
                else:
                    self.wfile.write(state["body"])
            except (BrokenPipeError, ConnectionResetError):
                disconnected.set()

        def log_message(self, *args):
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)

    class FakeServer:
        def __init__(self, args, port):
            self.args, self.port, self.stopped = args, port, False
            self.url = f"http://127.0.0.1:{upstream.server_port}"
            instances.append(self)

        def stop(self):
            self.stopped = True

    def decide(policy, text):
        decisions.append(text)
        return 4 if "hard" in text else 1

    monkeypatch.setattr(G, "Server", FakeServer)
    monkeypatch.setattr(G, "load_policy", lambda path: {"feature_set": "text"})
    monkeypatch.setattr(G, "decide_k", decide)
    log_path = tmp_path / "logs" / "gateway.log"
    with serving(upstream):
        with serving(G.Gateway(port=0, log_path=log_path)) as gateway:
            gateway.logged = threading.Event()
            original_log = gateway.log_request

            def logged(*args):
                original_log(*args)
                gateway.logged.set()

            gateway.log_request = logged
            yield gateway, seen, state, decisions, disconnected, log_path, instances
    assert instances[0].stopped


def request(gateway, body=None, path="/v1/chat/completions"):
    conn = http.client.HTTPConnection("127.0.0.1", gateway.server_port, timeout=5)
    try:
        conn.request("GET" if body is None else "POST", path,
                     body=None if body is None else json.dumps(body),
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def test_last_user_text_shapes():
    assert G.last_user_text([{"role": "user", "content": "hello"}]) == "hello"
    assert G.last_user_text([{"role": "user", "content": [
        {"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {}},
        {"type": "text", "text": " world"}]}]) == "hello world"
    assert G.last_user_text([{"role": "user", "content": "old"},
                             {"role": "user", "content": "last"},
                             {"role": "tool", "content": "result"}]) == "last"
    assert G.last_user_text([{"role": "system", "content": "system"}]) == ""
    assert G.last_user_text([]) == ""
    assert G.last_user_text([{"role": "user", "content": None}]) == ""


def test_k_follows_last_user_message(rig):
    gateway, seen, _, decisions, *_ = rig
    messages = [{"role": "user", "content": "hard question"}]
    for suffix in [[], [{"role": "assistant", "tool_calls": [{"id": "a"}]},
                        {"role": "tool", "tool_call_id": "a", "content": "easy result"}]]:
        assert request(gateway, {"messages": messages + suffix, "n_loops": None}) == (200, REPLY)
        assert seen[-1][0] == "/v1/chat/completions"
        assert seen[-1][1]["n_loops"] == 4
        assert seen[-1][1]["messages"] == messages + suffix
    assert decisions == ["hard question"]  # tool steps use the text-only cache
    assert request(gateway, {"messages": [{"role": "user", "content": "easy"}]})[0] == 200
    assert seen[-1][1]["n_loops"] == 1


@pytest.mark.parametrize("kwargs, expected", [(None, {"enable_thinking": False}),
    ({"enable_thinking": True}, {"enable_thinking": True}),
    ({"other": 7}, {"other": 7, "enable_thinking": False})])
def test_enable_thinking_default(rig, kwargs, expected):
    gateway, seen, *_ = rig
    body = {"messages": [], "tools": [{"type": "function"}], "stream_options": {"include_usage": True}}
    if kwargs is not None:
        body["chat_template_kwargs"] = kwargs
    assert request(gateway, body)[0] == 200
    assert seen[-1][1] == {**body, "n_loops": 1, "chat_template_kwargs": expected}


def test_stream_passthrough(rig):
    gateway, *_ = rig
    assert request(gateway, {"messages": [], "stream": True}) == (200, b"".join(SSE))


def test_stream_flushes_before_upstream_finishes(rig):
    gateway, _, state, *_ = rig
    state["stream_gate"] = threading.Event()
    conn = http.client.HTTPConnection("127.0.0.1", gateway.server_port, timeout=2)
    try:
        conn.request("POST", "/v1/chat/completions", json.dumps({"messages": [], "stream": True}))
        response = conn.getresponse()
        assert response.read(len(SSE[0])) == SSE[0]
        state["stream_gate"].set()
        assert response.read() == b"".join(SSE[1:])
    finally:
        state["stream_gate"].set()
        conn.close()


def test_client_disconnect(rig):
    gateway, _, state, _, disconnected, *_ = rig
    state["endless"] = True
    conn = http.client.HTTPConnection("127.0.0.1", gateway.server_port, timeout=5)
    conn.connect()
    client_socket = conn.sock
    conn.request("POST", "/v1/chat/completions", json.dumps({"messages": [], "stream": True}))
    response = conn.getresponse()
    assert response.read(5) == b"data:"
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    response.close()
    conn.close()
    assert disconnected.wait(5), "upstream must be closed when the client disconnects"
    assert request(gateway, {"messages": []}) == (200, REPLY)


def test_models_endpoint(rig):
    status, body = request(rig[0], path="/v1/models")
    assert status == 200
    assert json.loads(body) == {"data": [{"id": "dory", "object": "model"}]}


def test_server_defaults(rig):
    args = rig[-1][0].args
    for flag, value in [("-m", "models/dory-q5.gguf"), ("-np", "1"), ("-c", "16384"),
                        ("--spec-type", "draft-simple"), ("--spec-draft-n-max", "4"),
                        ("--spec-draft-loops", "1")]:
        assert args[args.index(flag) + 1] == value
    assert "--jinja" in args and "--override-kv" not in args


@pytest.mark.parametrize("stream", [False, True])
def test_request_logging(rig, capsys, stream):
    gateway, _, _, _, _, path, _ = rig
    assert request(gateway, {"messages": [], "stream": stream})[0] == 200
    assert gateway.logged.wait(5)
    line = path.read_text()
    assert len(line.splitlines()) == 1
    assert all(field in line for field in ["time=", "k=1", "prompt_tokens=5", "generated_tokens=2",
                                           "tok/s=20.0", "draft_acceptance=0.75"])
    assert line in capsys.readouterr().err


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("body", [b'{"timings":null,"usage":null}',
    b'{"timings":{"predicted_per_second":null},"usage":{"prompt_tokens":null,"completion_tokens":null}}'])
def test_missing_or_bad_timings_do_not_fail(rig, stream, body):
    gateway, _, state, _, _, path, _ = rig
    state["body"] = body
    state["sse"] = [b"data: " + body + b"\n\n", b"data: [DONE]\n\n"]
    expected = b"".join(state["sse"]) if stream else body
    assert request(gateway, {"messages": [], "stream": stream}) == (200, expected)
    assert gateway.logged.wait(5)
    assert all(field in path.read_text() for field in ["prompt_tokens=-", "generated_tokens=-",
                                                     "tok/s=-", "draft_acceptance=-"])


def test_logging_failure_does_not_fail_request(rig, capsys):
    gateway = rig[0]
    gateway.log_path = gateway.log_path.parent  # a directory cannot be opened for append
    assert request(gateway, {"messages": []}) == (200, REPLY)
    assert gateway.logged.wait(5)
    assert "k=1" in capsys.readouterr().err


def test_upstream_error_passthrough(rig):
    gateway, _, state, *_ = rig
    state.update(status=400, body=b'{ "error": "invalid request" }')
    assert request(gateway, {"messages": []}) == (400, state["body"])


def test_invalid_json_then_valid_request(rig):
    conn = http.client.HTTPConnection("127.0.0.1", rig[0].server_port, timeout=5)
    conn.request("POST", "/v1/chat/completions", b"{")
    response = conn.getresponse()
    assert response.status == 400
    response.read()
    conn.close()
    assert request(rig[0], {"messages": []}) == (200, REPLY)


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_main_stops_on_signal(monkeypatch, signum):
    handlers, stopped = {}, []

    class FakeGateway:
        def __init__(self, **kwargs):
            assert kwargs == {"port": 8083, "policy_path": Path("results/policy_text.joblib"),
                              "gguf": Path("models/dory-q5.gguf"), "ctx": 16384}

        def serve_forever(self):
            handlers[signum](signum, None)

        def server_close(self):
            stopped.append(True)

    monkeypatch.setattr(G, "Gateway", FakeGateway)
    monkeypatch.setattr(G.signal, "signal", lambda sig, fn: handlers.setdefault(sig, fn))
    G.main([])
    assert stopped == [True]
