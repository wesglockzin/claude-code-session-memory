"""Tests for embed() against a fake local HTTP server, and for fixed bugs."""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))
import eval_recall as er

import session_memory as sm


def _serve(routes):
    """routes: path -> (status, body-bytes). Returns (server, base_url)."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            status, body = routes.get(self.path, (404, b"nope"))
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


@pytest.fixture
def fake_ollama(tmp_path, monkeypatch):
    servers = []

    def start(routes):
        server, url = _serve(routes)
        servers.append(server)
        monkeypatch.setattr(sm, "OLLAMA", url)
        return url

    monkeypatch.setattr(sm, "STATE_DIR", str(tmp_path))
    monkeypatch.setattr(sm, "ERR_LOG", str(tmp_path / "err.log"))
    yield start
    for s in servers:
        s.shutdown()
        s.server_close()


def test_embed_parses_api_embed(fake_ollama):
    fake_ollama(
        {"/api/embed": (200, json.dumps({"embeddings": [[0.5, 0.25]]}).encode())}
    )
    assert sm.embed("hi", 5.0) == [0.5, 0.25]


def test_embed_falls_back_to_legacy_endpoint(fake_ollama):
    fake_ollama(
        {"/api/embeddings": (200, json.dumps({"embedding": [1.0, 2.0]}).encode())}
    )
    assert sm.embed("hi", 5.0) == [1.0, 2.0]


def test_embed_http_error_is_logged_as_http_error(fake_ollama, tmp_path):
    fake_ollama({"/api/embed": (404, b"model not found")})
    assert sm.embed("hi", 5.0) is None
    log = (tmp_path / "err.log").read_text()
    assert "HTTP 404" in log and "model not found" in log
    assert "unreachable" not in log


def test_embed_connection_failure_logged_as_unreachable(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "OLLAMA", "http://127.0.0.1:1")
    monkeypatch.setattr(sm, "STATE_DIR", str(tmp_path))
    monkeypatch.setattr(sm, "ERR_LOG", str(tmp_path / "err.log"))
    assert sm.embed("hi", 2.0) is None
    assert "unreachable" in (tmp_path / "err.log").read_text()


def test_embed_bad_json_logged(fake_ollama, tmp_path):
    fake_ollama({"/api/embed": (200, b"not json"), "/api/embeddings": (200, b"{}")})
    assert sm.embed("hi", 5.0) is None
    assert "bad response" in (tmp_path / "err.log").read_text()


def test_save_index_uses_per_process_temp_name(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "STATE_DIR", str(tmp_path))
    monkeypatch.setattr(sm, "INDEX_PATH", str(tmp_path / "index.json"))
    seen = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (seen.append(a), real(a, b))[1])
    sm.save_index({"entries": {}})
    assert seen and seen[0] != str(tmp_path / "index.json") + ".tmp"
    assert str(os.getpid()) in seen[0]
    assert json.loads((tmp_path / "index.json").read_text()) == {"entries": {}}


def test_query_without_text_exits_with_usage(monkeypatch):
    with pytest.raises(SystemExit) as e:
        sm.main(["--query"])
    assert "usage" in str(e.value)


def test_missing_required_settings_name_the_setting(monkeypatch):
    monkeypatch.setattr(sm, "OLLAMA", "")
    monkeypatch.setattr(sm, "MODEL", "")
    with pytest.raises(SystemExit) as e:
        sm.require_settings()
    assert "OLLAMA_HOST" in str(e.value) and "SESSION_MEMORY_MODEL" in str(e.value)


def test_cli_fails_when_settings_unset(tmp_path):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("OLLAMA_HOST", "SESSION_MEMORY_MODEL")
    }
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "session_memory.py"), "--reindex"],
        capture_output=True,
        env=env,
        timeout=30,
    )
    assert r.returncode != 0 and b"OLLAMA_HOST" in r.stderr


def test_diff_ignores_non_json_files(tmp_path, monkeypatch):
    monkeypatch.setattr(er, "MANIFEST_DIR", str(tmp_path))
    for ts, r1 in (("t1", 0.9), ("t2", 0.9)):
        m = {
            "corpus_hash": "abc",
            "doc_hashes": {},
            "metrics": {"full": {"r1": r1, "r3": 1, "mrr": 1}},
        }
        (tmp_path / f"manifest-{ts}.json").write_text(json.dumps(m))
    (tmp_path / "zzz-stray.txt").write_text("not json")
    assert er.diff_mode() == 0


def test_bad_bar_is_a_usage_error(tmp_path, monkeypatch):
    q = tmp_path / "q.jsonl"
    q.write_text(json.dumps({"query": "a", "gold": "d1"}) + "\n")
    c = tmp_path / "c"
    c.mkdir()
    (c / "d1.md").write_text("one")
    monkeypatch.setattr(er, "embed", lambda text, timeout: [1.0, 0.0])
    monkeypatch.setattr(er, "MANIFEST_DIR", str(tmp_path / "m"))
    monkeypatch.setattr(
        sys, "argv", ["x", "--corpus", str(c), "--queries", str(q), "--bar", "garbage"]
    )
    with pytest.raises(SystemExit) as e:
        er.main()
    assert e.value.code == 2
