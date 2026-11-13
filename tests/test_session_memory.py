"""Tests run WITHOUT Ollama: the embedder is stubbed. What's under test is
everything around the model — the contracts (fail-open, caps, floor), the
index lifecycle, the supersession filter, and the eval math."""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import session_memory as sm

sys.path.insert(0, os.path.join(ROOT, "eval"))
import eval_recall as er

# --- pure functions ---------------------------------------------------------


def test_normed_unit_length():
    v = sm.normed([3.0, 4.0])
    assert abs((v[0] ** 2 + v[1] ** 2) - 1.0) < 1e-9


def test_normed_zero_vector_unchanged():
    assert sm.normed([0.0, 0.0]) == [0.0, 0.0]


# --- floor + render contracts ----------------------------------------------


def _entry(tmp_path, stem, body="hello world", vec=(1.0, 0.0)):
    p = tmp_path / f"{stem}.md"
    p.write_text(body, encoding="utf-8")
    return {"path": str(p), "mtime": 0.0, "vec": list(vec)}


def test_render_floor_blocks_low_similarity(tmp_path):
    hits = [(sm.FLOOR - 0.01, "a", _entry(tmp_path, "a"))]
    out, injected = sm.render(hits)
    assert out == "" and injected == 0


def test_render_caps_at_k_even_with_wider_slice(tmp_path):
    hits = [(0.9, f"m{i}", _entry(tmp_path, f"m{i}")) for i in range(10)]
    out, injected = sm.render(hits)
    assert injected == sm.K
    assert out.count("--- m") == sm.K


def test_render_respects_char_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "TOTAL_CHAR_CAP", 50)
    hits = [(0.9, f"m{i}", _entry(tmp_path, f"m{i}", body="x" * 40)) for i in range(4)]
    _, injected = sm.render(hits)
    assert injected < 4  # budget cut it off
    assert injected >= 1


def test_render_injected_count_matches_output(tmp_path):
    hits = [
        (0.9, "a", _entry(tmp_path, "a")),
        (sm.FLOOR - 0.05, "b", _entry(tmp_path, "b")),
    ]
    out, injected = sm.render(hits)
    assert injected == 1 and "--- a" in out and "--- b" not in out


# --- supersession filter ----------------------------------------------------


def test_superseded_excluded(tmp_path):
    p = tmp_path / "old.md"
    p.write_text("---\nlifecycle: superseded\n---\nbody", encoding="utf-8")
    assert sm.is_superseded(str(p), "old") is True


def test_current_memory_not_excluded(tmp_path):
    p = tmp_path / "cur.md"
    p.write_text("---\ndescription: fine\n---\nbody", encoding="utf-8")
    assert sm.is_superseded(str(p), "cur") is False


def test_unreadable_file_fails_closed(tmp_path):
    assert sm.is_superseded(str(tmp_path / "missing.md"), "missing") is True


def test_superseded_with_inline_comment_still_excluded(tmp_path):
    p = tmp_path / "old.md"
    p.write_text(
        "---\nlifecycle: superseded  # retired 2024\n---\nbody", encoding="utf-8"
    )
    assert sm.is_superseded(str(p), "old") is True


def test_prose_supersedes_mention_is_not_a_marker(tmp_path):
    p = tmp_path / "cur.md"
    p.write_text(
        "---\ndescription: supersedes the old one\n---\nbody", encoding="utf-8"
    )
    assert sm.is_superseded(str(p), "cur") is False


def test_top_k_prefilters_superseded(tmp_path):
    old = tmp_path / "old.md"
    old.write_text("---\nlifecycle: superseded\n---\nbody", encoding="utf-8")
    idx = {
        "entries": {
            "old": {"path": str(old), "mtime": 0.0, "vec": [1.0, 0.0]},
            "cur": {
                "path": str(_entry(tmp_path, "cur")["path"]),
                "mtime": 0.0,
                "vec": [0.9, 0.1],
            },
        }
    }
    idx["entries"]["cur"]["path"] = str(tmp_path / "cur.md")
    got = [stem for _, stem, _ in sm.top_k(idx, [1.0, 0.0])]
    assert "old" not in got and "cur" in got


def test_top_k_skips_dim_mismatch(tmp_path):
    idx = {
        "entries": {
            "bad": {
                "path": str(_entry(tmp_path, "bad")["path"]),
                "mtime": 0.0,
                "vec": [1.0, 0.0, 0.0],
            }
        }
    }
    assert sm.top_k(idx, [1.0, 0.0]) == []


# --- index lifecycle --------------------------------------------------------


def test_wrong_regime_index_rejected_wholesale(tmp_path, monkeypatch):
    idx_path = tmp_path / "index.json"
    idx_path.write_text(
        json.dumps({"model": sm.MODEL, "regime": "raw-v0", "entries": {"a": {}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(sm, "INDEX_PATH", str(idx_path))
    monkeypatch.setattr(sm, "ERR_LOG", str(tmp_path / "err.log"))
    idx = sm.load_index()
    assert idx["entries"] == {} and idx["regime"] == sm.REGIME


def test_refresh_reembed_capped(tmp_path, monkeypatch):
    mem = tmp_path / "mem"
    mem.mkdir()
    for i in range(5):
        (mem / f"m{i}.md").write_text(f"body {i}", encoding="utf-8")
    monkeypatch.setattr(sm, "MEM_DIR", str(mem))
    calls = []
    monkeypatch.setattr(
        sm, "embed", lambda text, timeout: calls.append(1) or [1.0, 0.0]
    )
    idx = {"model": sm.MODEL, "regime": sm.REGIME, "entries": {}}
    dirty = sm.refresh(idx, 1.0, max_reembeds=3)
    assert dirty and len(calls) == 3 and len(idx["entries"]) == 3


def test_refresh_drops_deleted(tmp_path, monkeypatch):
    mem = tmp_path / "mem"
    mem.mkdir()
    monkeypatch.setattr(sm, "MEM_DIR", str(mem))
    idx = {
        "model": sm.MODEL,
        "regime": sm.REGIME,
        "entries": {"gone": {"path": "/nope", "mtime": 0.0, "vec": [1.0]}},
    }
    assert sm.refresh(idx, 1.0, 3) is True
    assert idx["entries"] == {}


# --- hook process contracts (subprocess, env-isolated) ----------------------


def _run_hook(tmp_path, stdin: bytes):
    env = dict(
        os.environ,
        CLAUDE_MEMORY_DIR=str(tmp_path / "mem"),
        SESSION_MEMORY_STATE=str(tmp_path / "state"),
        OLLAMA_HOST="http://127.0.0.1:1",
    )  # nothing listens
    (tmp_path / "mem").mkdir(exist_ok=True)
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "session_memory.py")],
        input=stdin,
        capture_output=True,
        env=env,
        timeout=30,
        check=False,
    )


def test_hook_fail_open_on_garbage_stdin(tmp_path):
    r = _run_hook(tmp_path, b"not json at all")
    assert r.returncode == 0 and r.stdout == b""


def test_hook_fail_open_when_ollama_down(tmp_path):
    # A VALID index must exist, or the hook exits on "index empty" before it
    # ever dials Ollama and the test is vacuous (adversarial-review finding).
    mem = tmp_path / "mem"
    state = tmp_path / "state"
    mem.mkdir()
    state.mkdir()
    doc = mem / "a.md"
    doc.write_text("body", encoding="utf-8")
    (state / "index.json").write_text(
        json.dumps(
            {
                "model": sm.MODEL,
                "regime": sm.REGIME,
                "entries": {
                    "a": {
                        "path": str(doc),
                        "mtime": os.path.getmtime(doc),
                        "vec": [1.0, 0.0],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    payload = json.dumps({"prompt": "how should risky changes be rolled out"})
    r = _run_hook(tmp_path, payload.encode())
    assert r.returncode == 0 and r.stdout == b""
    # and it must have gotten far enough to log the query-embed failure
    err = state / "hook.err.log"
    assert err.exists() and b"query embed failed" in err.read_bytes()


def test_hook_skips_short_prompts(tmp_path):
    r = _run_hook(tmp_path, json.dumps({"prompt": "ok go"}).encode())
    assert r.returncode == 0 and r.stdout == b""


def test_reindex_fails_loud_on_missing_dir(tmp_path):
    env = dict(
        os.environ,
        CLAUDE_MEMORY_DIR=str(tmp_path / "does-not-exist"),
        SESSION_MEMORY_STATE=str(tmp_path / "state"),
    )
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "session_memory.py"), "--reindex"],
        capture_output=True,
        env=env,
        timeout=30,
        check=False,
    )
    assert r.returncode != 0  # CLI modes are NOT fail-open


def test_hook_skips_slash_commands(tmp_path):
    r = _run_hook(tmp_path, json.dumps({"prompt": "/compact please now ok"}).encode())
    assert r.returncode == 0 and r.stdout == b""


# --- eval harness math ------------------------------------------------------


def test_corpus_hash_deterministic_and_sensitive(tmp_path):
    c1 = {"a": "one", "b": "two"}
    h1, _ = er.corpus_hashes(c1)
    h2, _ = er.corpus_hashes(dict(reversed(list(c1.items()))))
    assert h1 == h2  # order-independent
    h3, _ = er.corpus_hashes({"a": "one", "b": "CHANGED"})
    assert h1 != h3


def test_wrong_model_index_rejected(tmp_path, monkeypatch):
    idx_path = tmp_path / "index.json"
    idx_path.write_text(
        json.dumps(
            {"model": "some-other-model", "regime": sm.REGIME, "entries": {"a": {}}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sm, "INDEX_PATH", str(idx_path))
    monkeypatch.setattr(sm, "ERR_LOG", str(tmp_path / "err.log"))
    assert sm.load_index()["entries"] == {}


def test_render_header_count_matches_after_budget_drop(tmp_path, monkeypatch):
    # Header is written from the FINAL injected count, not the pre-budget one.
    monkeypatch.setattr(sm, "TOTAL_CHAR_CAP", 45)
    hits = [(0.9, f"m{i}", _entry(tmp_path, f"m{i}", body="x" * 40)) for i in range(3)]
    out, injected = sm.render(hits)
    assert injected == out.count("--- m")
    assert out.startswith(f"[Memory recall] {injected} memor")


def _write_manifest(dirpath, ts, corpus_hash, doc_hashes, r1, r3, mrr):
    m = {
        "ts": ts,
        "corpus_hash": corpus_hash,
        "doc_hashes": doc_hashes,
        "metrics": {"full": {"r1": r1, "r3": r3, "mrr": mrr}},
    }
    with open(os.path.join(dirpath, f"manifest-{ts}.json"), "w") as f:
        json.dump(m, f)


def test_diff_same_hash_metric_drop_is_regression(tmp_path, monkeypatch):
    monkeypatch.setattr(er, "MANIFEST_DIR", str(tmp_path))
    _write_manifest(tmp_path, "t1", "abc", {"a": "h1"}, 0.9, 1.0, 0.95)
    _write_manifest(tmp_path, "t2", "abc", {"a": "h1"}, 0.8, 1.0, 0.90)
    assert er.diff_mode() == 1  # gates


def test_diff_changed_hash_flags_rebaseline_not_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(er, "MANIFEST_DIR", str(tmp_path))
    _write_manifest(tmp_path, "t1", "abc", {"a": "h1"}, 0.9, 1.0, 0.95)
    _write_manifest(tmp_path, "t2", "xyz", {"a": "h1", "b": "h2"}, 0.8, 1.0, 0.90)
    assert er.diff_mode() == 0  # attributable to corpus drift — no gate
    outp = capsys.readouterr().out
    assert "RE-BASELINE" in outp and "b" in outp


def test_score_metrics_math(monkeypatch):
    # 3 docs on orthogonal axes; queries map deterministically.
    vecs = {
        "d1": [1.0, 0.0, 0.0],
        "d2": [0.0, 1.0, 0.0],
        "d3": [0.0, 0.0, 1.0],
        "q_d1": [1.0, 0.0, 0.0],
        "q_d2_near": [0.1, 0.99, 0.0],
        "q_miss": [0.6, 0.55, 0.5],
    }

    def fake_embed(text, timeout):
        for key, v in vecs.items():
            if text.endswith(key) or key in text:
                return v
        return [0.0, 0.0, 0.0]

    monkeypatch.setattr(er, "embed", fake_embed)
    corpus = {"d1": "d1", "d2": "d2", "d3": "d3"}
    queries = [
        {"query": "q_d1", "gold": "d1"},
        {"query": "q_d2_near", "gold": "d2"},
        {"query": "q_miss", "gold": "d3", "adversarial": True},
    ]
    res = er.score(corpus, queries)
    assert res["full"]["n"] == 3
    assert res["standard"] == {"n": 2, "r1": 1.0, "r3": 1.0, "r5": 1.0, "mrr": 1.0}
    assert res["adversarial"]["n"] == 1
    # q_miss ranks d3 below d1/d2 (0.5 < 0.6, 0.55) → rank 3 exactly:
    row = next(r for r in res["rows"] if r["gold"] == "d3")
    assert row["rank"] == 3 and row["top3"][2] == "d3"
    assert res["full"]["r1"] == round(2 / 3, 3)
    assert res["full"]["mrr"] == round((1 + 1 + 1 / 3) / 3, 3)
