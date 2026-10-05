"""llmroute tests: local mock OpenAI servers. Run: python -m pytest tests/ -x -q"""
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

BIN = [sys.executable, "-m", "llmroute"]


class MockLLM(BaseHTTPRequestHandler):
    delay = 0
    fail = False
    name = "mock"

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "boom"}}')
            return
        n = body.get("max_tokens", 8)
        reply = {
            "choices": [{"message": {"content": f"reply-from-{self.name}"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": n, "total_tokens": 10 + n},
        }
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture()
def servers():
    """fast (cheap-ish), slow (cheap), dead (500s). Returns {name: url}."""
    specs = {"fast": {"delay": 0.05}, "slow": {"delay": 0.4}, "dead": {"fail": True}}
    procs, urls = [], {}
    for name, kw in specs.items():
        cls = type(f"Mock{name}", (MockLLM,), dict(kw, name=name))
        srv = HTTPServer(("127.0.0.1", 0), cls)
        urls[name] = f"http://127.0.0.1:{srv.server_port}"
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        procs.append(srv)
    yield urls
    for s in procs:
        s.shutdown()


@pytest.fixture()
def lr(tmp_path, monkeypatch, servers):
    cfg = tmp_path / "endpoints.toml"
    cfg.write_text(f"""
[[endpoint]]
name = "fast"
base_url = "{servers['fast']}"
model = "fast-model"
price_in = 2.0
price_out = 4.0
tags = ["parallel"]

[[endpoint]]
name = "slow"
base_url = "{servers['slow']}"
model = "slow-model"
price_in = 0.5
price_out = 1.0

[[endpoint]]
name = "dead"
base_url = "{servers['dead']}"
model = "dead-model"
price_in = 0.1
price_out = 0.2
""")
    env = dict(os.environ)
    def run(*args, **kwargs):
        return subprocess.run(BIN + ["--config", str(cfg)] + list(args),
                              capture_output=True, text=True, env=env,
                              timeout=kwargs.pop("timeout", 60), **kwargs)
    run.cfg = cfg
    run.results = tmp_path / "results.json"
    return run


def benched(lr):
    r = lr("bench", "--probes", "2")
    assert r.returncode == 0, r.stderr
    return json.loads(lr.results.read_text())["endpoints"]


def test_bench_measures_latency_and_success(lr):
    eps = benched(lr)
    assert eps["fast"]["p50_ms"] < eps["slow"]["p50_ms"]
    assert eps["fast"]["success_rate"] == 1.0
    assert eps["dead"]["success_rate"] == 0.0
    assert eps["dead"]["p50_ms"] is None


def test_run_fastest_policy(lr):
    benched(lr)
    r = lr("run", "--policy", "fastest", "--prompt", "hello")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "reply-from-fast"
    assert "served by fast" in r.stderr


def test_run_cheapest_policy(lr):
    benched(lr)
    r = lr("run", "--policy", "cheapest", "--prompt", "hello")
    assert r.returncode == 0, r.stderr
    # dead is cheapest but failing -> falls through to slow
    assert r.stdout.strip() == "reply-from-slow"
    assert "dead failed" in r.stderr


def test_run_balanced_policy(lr):
    benched(lr)
    r = lr("run", "--policy", "balanced", "--prompt", "hello")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() in ("reply-from-fast", "reply-from-slow")


def test_run_tag_filter(lr):
    benched(lr)
    r = lr("run", "--policy", "fastest", "--tag", "parallel", "--prompt", "hi")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "reply-from-fast"


def test_run_all_fail(lr, tmp_path, servers):
    cfg = tmp_path / "only-dead.toml"
    cfg.write_text(f"""
[[endpoint]]
name = "dead"
base_url = "{servers['dead']}"
model = "x"
""")
    r = subprocess.run(BIN + ["--config", str(cfg), "bench"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0
    r = subprocess.run(BIN + ["--config", str(cfg), "--results", str(tmp_path / "results.json"),
                             "run", "--prompt", "hi"], capture_output=True, text=True, timeout=60)
    assert r.returncode != 0
    assert "all endpoints failed" in r.stderr


def test_run_without_bench_dies(lr):
    r = lr("run", "--prompt", "hi")
    assert r.returncode != 0
    assert "bench" in r.stderr


def test_stdin_prompt(lr):
    benched(lr)
    r = subprocess.run(BIN + ["--config", str(lr.cfg), "run", "--policy", "fastest"],
                       input="piped prompt\n", capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "reply-from-fast"


def test_list(lr):
    benched(lr)
    r = lr("list")
    assert r.returncode == 0
    assert "fast" in r.stdout and "slow" in r.stdout and "2.0/4.0" in r.stdout


def test_bad_config(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[[endpoint]]\nname = 'x'\n")  # missing base_url/model
    r = subprocess.run(BIN + ["--config", str(bad), "list"], capture_output=True, text=True, timeout=30)
    assert r.returncode != 0
    assert "missing required key" in r.stderr
