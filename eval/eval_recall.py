#!/usr/bin/env python3
"""
eval_recall.py — measure retrieval quality and attribute any change to a cause.

Scores the hook's actual retrieval path (same embedder, same prefixes, same
cosine) against a labeled query set, then records a per-run MANIFEST so the
next run can tell a retrieval regression from corpus drift.

Why the manifest exists: a metric that cannot be attributed to a cause is a
number, not a signal. On my production corpus, a run once came in 7 points
under the bar while the corpus had grown 13% — and nothing could say how much
of the gap was regression versus 16 new documents competing for rank. Every
run since records the corpus content hash:

  corpus hash IDENTICAL vs last run -> any metric drop is a RETRIEVAL
                                       REGRESSION; `diff` exits 1.
  corpus hash CHANGED               -> not like-for-like. `diff` FLAGS FOR
                                       RE-BASELINE and names what moved.
  (--bar gates a single RUN against a pre-committed floor; `diff` compares
   the last two manifests and needs no bar.)

The hash is an EMBEDDED-CONTENT hash: each document is hashed exactly as the
embedder sees it (first 6 KB), so "hash identical" means the retrieval inputs
were identical — a change past the truncation point is invisible to the
embedder and therefore, deliberately, to the hash.

Usage:
  eval/eval_recall.py --corpus demo-corpus --queries eval/queries.jsonl
      [--bar r1=0.80,r3=0.93,mrr=0.87]   # gate against a pre-committed floor
  eval/eval_recall.py diff               # attribute the last two manifests

Requires Ollama running with the `embeddinggemma` model pulled. Stdlib only.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from session_memory import (
    MODEL,
    QUERY_PREFIX,
    REGIME,
    doc_prefix,
    embed,
    normed,
)

MANIFEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifests")
EMBED_TIMEOUT = 120.0
MAX_CHARS = 6000


def load_corpus(corpus_dir: str) -> dict[str, str]:
    out = {}
    for name in sorted(os.listdir(corpus_dir)):
        if name.endswith(".md"):
            with open(os.path.join(corpus_dir, name), encoding="utf-8") as f:
                out[name[:-3]] = f.read()[:MAX_CHARS]
    return out


def corpus_hashes(corpus: dict[str, str]) -> tuple[str, dict[str, str]]:
    per_doc = {
        s: hashlib.sha256(t.encode()).hexdigest()[:16]
        for s, t in sorted(corpus.items())
    }
    whole = hashlib.sha256(json.dumps(per_doc, sort_keys=True).encode()).hexdigest()[
        :16
    ]
    return whole, per_doc


def score(corpus: dict[str, str], queries: list[dict]) -> dict:
    stems = list(corpus)
    vecs = {}
    for i, s in enumerate(stems, 1):
        v = embed(doc_prefix(s) + corpus[s], EMBED_TIMEOUT)
        if v is None:
            sys.exit(f"embed failed for corpus doc {s!r} — is Ollama up with {MODEL}?")
        vecs[s] = normed(v)
        print(f"\r  embedded {i}/{len(stems)}", end="", flush=True)
    print()

    rows, ranks = [], []
    for q in queries:
        qv = embed(QUERY_PREFIX + q["query"], EMBED_TIMEOUT)
        if qv is None:
            sys.exit(f"embed failed for query {q['query']!r}")
        qv = normed(qv)
        order = sorted(stems, key=lambda s: -sum(a * b for a, b in zip(vecs[s], qv)))
        rank = order.index(q["gold"]) + 1 if q["gold"] in order else None
        rows.append({**q, "rank": rank, "top3": order[:3]})
        ranks.append((rank, q.get("adversarial", False)))

    def metrics(pairs):
        n = len(pairs)
        if not n:
            return {}

        def r_at(k):
            return sum(1 for r, _ in pairs if r and r <= k) / n

        return {
            "n": n,
            "r1": round(r_at(1), 3),
            "r3": round(r_at(3), 3),
            "r5": round(r_at(5), 3),
            "mrr": round(sum(1.0 / r for r, _ in pairs if r) / n, 3),
        }

    return {
        "full": metrics(ranks),
        "standard": metrics([p for p in ranks if not p[1]]),
        "adversarial": metrics([p for p in ranks if p[1]]),
        "rows": rows,
    }


def write_manifest(result: dict, corpus: dict[str, str], args) -> str:
    whole, per_doc = corpus_hashes(corpus)
    manifest = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "regime": REGIME,
        # Repo-relative on purpose: manifests must not record absolute local
        # paths (they leak usernames/layout if a manifest is ever shared).
        "corpus_dir": os.path.relpath(args.corpus),
        "doc_count": len(corpus),
        "corpus_hash": whole,
        "doc_hashes": per_doc,
        "queries": os.path.relpath(args.queries),
        "query_count": sum(len(v) for k, v in result.items() if k == "rows"),
        "metrics": {k: v for k, v in result.items() if k != "rows"},
    }
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    path = os.path.join(
        MANIFEST_DIR, f"manifest-{manifest['ts'].replace(':', '').split('.')[0]}.json"
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    return path


def diff_mode() -> int:
    names = os.listdir(MANIFEST_DIR) if os.path.isdir(MANIFEST_DIR) else []
    files = sorted(n for n in names if n.endswith(".json"))
    if len(files) < 2:
        print("need at least two manifests to diff")
        return 0
    loaded = []
    for f in files[-2:]:
        with open(os.path.join(MANIFEST_DIR, f), encoding="utf-8") as fh:
            loaded.append(json.load(fh))
    a, b = loaded
    print(f"comparing {files[-2]} -> {files[-1]}")
    ma, mb = a["metrics"]["full"], b["metrics"]["full"]
    for k in ("r1", "r3", "mrr"):
        print(f"  {k}: {ma.get(k)} -> {mb.get(k)}")
    if a["corpus_hash"] == b["corpus_hash"]:
        print("corpus hash IDENTICAL -> any metric change is a RETRIEVAL REGRESSION")
        return 1 if any(mb.get(k, 0) < ma.get(k, 0) for k in ("r1", "r3", "mrr")) else 0
    added = sorted(set(b["doc_hashes"]) - set(a["doc_hashes"]))
    removed = sorted(set(a["doc_hashes"]) - set(b["doc_hashes"]))
    changed = sorted(
        s
        for s in set(a["doc_hashes"]) & set(b["doc_hashes"])
        if a["doc_hashes"][s] != b["doc_hashes"][s]
    )
    print("corpus hash CHANGED -> not like-for-like; FLAG FOR RE-BASELINE")
    print(
        f"  added: {added or '-'}\n  removed: {removed or '-'}\n  changed: {changed or '-'}"
    )
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "diff":
        return diff_mode()
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="demo-corpus")
    p.add_argument("--queries", default="eval/queries.jsonl")
    p.add_argument("--bar", help="pre-committed floor, e.g. r1=0.80,r3=0.93,mrr=0.87")
    args = p.parse_args()

    corpus = load_corpus(args.corpus)
    with open(args.queries, encoding="utf-8") as qf:
        queries = [json.loads(line) for line in qf if line.strip()]
    print(
        f"corpus: {len(corpus)} docs | queries: {len(queries)} | model: {MODEL} ({REGIME})"
    )
    result = score(corpus, queries)

    for name in ("full", "standard", "adversarial"):
        m = result[name]
        if m:
            print(
                f"  {name:<12} n={m['n']:>3}  R@1={m['r1']:.0%}  R@3={m['r3']:.0%}  "
                f"R@5={m['r5']:.0%}  MRR={m['mrr']:.3f}"
            )
    misses = [r for r in result["rows"] if r["rank"] is None or r["rank"] > 3]
    for r in misses:
        print(
            f"  miss: rank={r['rank']} gold={r['gold']} q={r['query']!r} top3={r['top3']}"
        )

    path = write_manifest(result, corpus, args)
    print(f"manifest: {path}")

    if args.bar:
        try:
            bar = {
                k: float(v) for k, v in (kv.split("=") for kv in args.bar.split(","))
            }
        except ValueError:
            p.error(f"--bar must look like r1=0.80,r3=0.93,mrr=0.87; got {args.bar!r}")
        full = result["full"]
        fails = [k for k, v in bar.items() if full.get(k, 0) < v]
        if fails:
            print(f"BELOW BAR on {fails} (bar {bar}) — gate FAILS")
            return 1
        print(f"bar {bar} — PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
