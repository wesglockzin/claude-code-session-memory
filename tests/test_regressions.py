"""Tests for embed() against a fake local HTTP server, and for fixed bugs."""

import io
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
        check=False,
    )
    assert r.returncode != 0 and b"OLLAMA_HOST" in r.stderr


def _one_entry_index(tmp_path, monkeypatch):
    mem = tmp_path / "a.md"
    mem.write_text("body of a")
    idx = {"entries": {"a": {"path": str(mem), "mtime": 0, "vec": [1.0, 0.0]}}}
    monkeypatch.setattr(sm, "STATE_DIR", str(tmp_path))
    monkeypatch.setattr(sm, "ERR_LOG", str(tmp_path / "err.log"))
    monkeypatch.setattr(sm, "load_index", lambda: idx)
    monkeypatch.setattr(sm, "refresh", lambda *a, **k: False)
    monkeypatch.setattr(sm, "embed", lambda text, timeout: [1.0, 0.0])
    return idx


def test_recall_log_write_failure_is_logged_and_hook_still_succeeds(
    tmp_path, monkeypatch, capsys
):
    _one_entry_index(tmp_path, monkeypatch)
    log_dir = tmp_path / "recall"
    log_dir.mkdir()  # opening a directory for append raises OSError
    monkeypatch.setattr(sm, "RECALL_LOG", str(log_dir))
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"prompt": "x" * 20})))
    sm.hook_mode()
    assert "recall log write failed" in (tmp_path / "err.log").read_text()
    assert "body of a" in capsys.readouterr().out


def test_invalid_hook_payload_is_logged(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "STATE_DIR", str(tmp_path))
    monkeypatch.setattr(sm, "ERR_LOG", str(tmp_path / "err.log"))
    monkeypatch.setattr(sys, "stdin", io.StringIO("not json"))
    sm.hook_mode()
    assert "invalid payload" in (tmp_path / "err.log").read_text()


def test_render_logs_unreadable_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "STATE_DIR", str(tmp_path))
    monkeypatch.setattr(sm, "ERR_LOG", str(tmp_path / "err.log"))
    hit = (0.9, "gone", {"path": str(tmp_path / "missing.md")})
    assert sm.render([hit]) == ("", 0)
    assert "cannot read gone" in (tmp_path / "err.log").read_text()


def test_refresh_logs_file_that_cannot_be_stat(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "STATE_DIR", str(tmp_path))
    monkeypatch.setattr(sm, "ERR_LOG", str(tmp_path / "err.log"))
    monkeypatch.setattr(sm, "corpus_files", lambda: {"a": str(tmp_path / "nope.md")})
    sm.refresh({"entries": {}}, 1.0, 3)
    assert "cannot stat a" in (tmp_path / "err.log").read_text()


def test_query_reports_http_status_not_ollama_down(fake_ollama, tmp_path, monkeypatch):
    fake_ollama(
        {
            "/api/embed": (404, b"model not found"),
            "/api/embeddings": (404, b"model not found"),
        }
    )
    monkeypatch.setattr(
        sm,
        "load_index",
        lambda: {"entries": {"a": {"path": "x", "mtime": 0, "vec": [1.0]}}},
    )
    monkeypatch.setattr(sm, "refresh", lambda *a, **k: False)
    with pytest.raises(SystemExit) as e:
        sm.main(["--query", "anything"])
    msg = str(e.value)
    assert "HTTP 404" in msg and "model not found" in msg
    assert "Ollama up" not in msg


def test_help_works_without_settings(monkeypatch, capsys):
    monkeypatch.setattr(sm, "OLLAMA", "")
    monkeypatch.setattr(sm, "MODEL", "")
    with pytest.raises(SystemExit) as e:
        sm.main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "--reindex" in out and "--query" in out and "--install-hook" in out


def test_install_hook_merges_and_is_idempotent(tmp_path):
    path = tmp_path / "dir" / "settings.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"theme": "dark", "hooks": {"Stop": []}}))
    assert sm.install_hook(str(path), "session-memory") is True
    assert sm.install_hook(str(path), "session-memory") is False
    data = json.loads(path.read_text())
    assert data["theme"] == "dark" and data["hooks"]["Stop"] == []
    groups = data["hooks"]["UserPromptSubmit"]
    assert groups == [{"hooks": [{"type": "command", "command": "session-memory"}]}]


def test_install_hook_creates_missing_settings_file(tmp_path, monkeypatch):
    path = tmp_path / "new" / "settings.json"
    monkeypatch.setenv("CLAUDE_SETTINGS_PATH", str(path))
    sm.main(["--install-hook"])
    data = json.loads(path.read_text())
    hook = data["hooks"]["UserPromptSubmit"][0]["hooks"][0]
    assert hook == {"type": "command", "command": "session-memory"}


def test_install_hook_treats_absolute_path_entry_as_installed(tmp_path):
    path = tmp_path / "settings.json"
    existing = {
        "hooks": {
            "UserPromptSubmit": [
                {"hooks": [{"type": "command", "command": "/v/bin/session-memory"}]}
            ]
        }
    }
    path.write_text(json.dumps(existing))
    assert sm.install_hook(str(path), "session-memory") is False
    assert json.loads(path.read_text()) == existing


def test_embed_empty_vector_records_error(monkeypatch):
    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"embeddings": []}'

    monkeypatch.setattr(sm.urllib.request, "urlopen", lambda *a, **k: Resp())
    assert sm.embed("x", 1) is None
    assert "empty embedding" in sm.LAST_EMBED["error"]


def test_install_hook_refuses_corrupt_settings(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    path.write_text("{not json")
    monkeypatch.setenv("CLAUDE_SETTINGS_PATH", str(path))
    with pytest.raises(SystemExit) as e:
        sm.main(["--install-hook"])
    assert "install-hook failed" in str(e.value)
    assert path.read_text() == "{not json"


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
    assert not (tmp_path / "m").exists()  # failed before embedding or manifest


def test_eval_names_missing_settings_before_doing_anything(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "OLLAMA", "")
    monkeypatch.setattr(sm, "MODEL", "")
    called = []
    monkeypatch.setattr(er, "embed", lambda text, timeout: called.append(1))
    monkeypatch.setattr(er, "MANIFEST_DIR", str(tmp_path / "m"))
    monkeypatch.setattr(sys, "argv", ["x"])
    with pytest.raises(SystemExit) as e:
        er.main()
    assert "OLLAMA_HOST" in str(e.value) and "SESSION_MEMORY_MODEL" in str(e.value)
    assert not called
