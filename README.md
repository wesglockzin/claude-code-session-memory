# claude-code-session-memory

[![ci](https://github.com/wesglockzin/claude-code-session-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/wesglockzin/claude-code-session-memory/actions/workflows/ci.yml)

> **Local RAG session memory for Claude Code — a `UserPromptSubmit` hook that
> retrieves the right memory files by meaning, on-device, with a measured and
> regression-gated eval harness.** No cloud calls: the embedder is Ollama on
> your machine, which is the point — it was built for an environment where
> the memory content cannot leave the host.

I run this in production on every prompt of every Claude Code session, over a
memory corpus of ~140 files. This repo is the same code with paths made
configurable and the eval harness included, plus a synthetic demo corpus so
the whole measurement pipeline reproduces on your machine.

## The problem

Claude Code memory directories grow. A hand-maintained index (`MEMORY.md`)
drifts three ways, and I got bitten by all three: stale one-line summaries
that keep getting acted on, files that never made it into the index and are
therefore invisible, and an index size cap that silently truncates. The root
cause is indexing *by hand*; the fix is indexing *the files on disk* and
retrieving *by meaning*:

- **Unindexed = impossible** — every file on disk gets embedded.
- **No drift** — the vector index is generated, not curated.
- **Retrieval by meaning** — "the user pushed back, what do I do" finds the
  right feedback memory with zero keyword overlap.

The always-loaded index stays for breadth; this hook adds per-prompt depth —
the top-K relevant memories, full text, injected just for that prompt.

## How it works

```
prompt ──► embed (Ollama, embeddinggemma, query prefix)
                 │
                 ▼
      cosine over the index of memory files      index.json refreshes lazily:
      (one 768-dim vector per file,              changed files re-embed at query
       doc-prefixed, regime-stamped)             time (capped), deleted drop out
                 │
                 ▼
      similarity floor + top-K + char caps ──► injected as prompt context
```

Contracts, each of which exists because its absence bit me:

- **Fail-open.** Any failure — Ollama down, index missing, bad payload —
  prints nothing, exits 0, never blocks the prompt. Errors go to a log.
- **Fail-loud where fail-open would lie.** The one exception: an interpreter
  below Python 3.10 dies at import in a way that mimics "no relevant
  memory," so the hook fails CLOSED with a message for that single case.
- **Token-capped.** At most K memories, hard char budget, per-memory cap —
  enforced at the render boundary, not by trusting callers to slice.
- **Floored.** A prompt with no sufficiently-similar memory injects nothing.
  The floor is calibrated, not guessed (see below).
- **Stdlib only.** A hook that fires on every prompt doesn't get a venv.
  Pure-python cosine over a few hundred 768-dim vectors costs ~10 ms.
- **Lifecycle-aware.** A memory marked `lifecycle: superseded` in its
  frontmatter is excluded *before* scoring — read from the file at query
  time, deliberately not cached in the index, because a cached flag is
  silently dropped on the next re-embed. The index is a cache; the file is
  the truth. Unreadable file → excluded, fail-closed, logged.

## The eval story (real numbers, production corpus)

This is the part I'd want to see in someone else's repo, so here is mine —
dates, pre-committed bars, and the misses.

**1. Embedder selection (July 2).** Two candidates, one pre-committed bar
signed off *before* the run (pass = Recall@3 ≥ 85%; pick = higher R@3,
tiebreak MRR, then latency). 108-file corpus, 20 labeled queries:

| model | R@1 | R@3 | MRR | ms/query |
|---|---|---|---|---|
| **embeddinggemma** | **95%** | 100% | **0.97** | 75 |
| nomic-embed-text | 85% | 100% | 0.93 | 18 |

Both cleared the bar; EmbeddingGemma won the tiebreak. The honest caveat I
recorded at the time: I authored both the queries and the memories, so 100%
R@3 partly means the query set was easy.

**2. A harder golden set (Aug 4).** The 20 v1 queries kept verbatim for
continuity, plus 10 new adversarial paraphrases written for near-zero
keyword overlap with their gold memory. Corpus at 124 files:

| subset | n | R@1 | R@3 | MRR |
|---|---|---|---|---|
| full | 30 | 80% | 93% | 0.873 |
| v1 (author-written) | 20 | 90% | 100% | 0.950 |
| adversarial | 10 | 60% | 80% | 0.720 |

The measured baseline became the new pre-committed regression bar
(R@1 ≥ 80%, R@3 ≥ 93%, MRR ≥ 0.87). The adversarial number quantifies known
paraphrase headroom — the trigger for a hybrid BM25+vector upgrade if
real-use misses accumulate (they're logged; see below).

**3. The attribution lesson (Aug 12).** A run came in seven points under the
bar — while the corpus had grown 13%. Nothing could say how much of the gap
was retrieval regression versus sixteen new documents competing for rank.
*A metric that cannot be attributed to a cause is a number, not a signal.*
Since then every eval run records a corpus content hash in a manifest:
hash identical → a metric drop is a regression and gates; hash changed →
flag for re-baseline and name exactly what moved. `eval/eval_recall.py diff`
implements it here.

**4. The regime fix (Aug 13).** EmbeddingGemma is trained with asymmetric
instruction prefixes — queries and documents embed differently, and Ollama
passes text through raw, so the caller owns the prefixes. Adding them:

| regime | R@1 | R@3 | MRR | adversarial R@1 |
|---|---|---|---|---|
| raw | 70% | 90% | 0.810 | 40% |
| **prefixed** | **83%** | **93%** | **0.893** | **60%** |

Prefixed and raw vectors score in different bands, so the index is
regime-stamped and an index from another regime is rejected wholesale rather
than mixed. The similarity floor was then recalibrated **on the live
prefixed index**: gold top-1 scores bottomed at 0.356, negative top-1 scores
peaked at 0.229 → floor 0.30, inside the gap. The old floor (0.35) had been
calibrated on raw embeddings and did not transfer — floors are
regime-specific.

**5. Ongoing.** Every prompt logs its top-10 candidates and scores (prompt
stored as a hash), so below-floor near-misses accumulate as data rather than
anecdotes, and a scheduled job re-runs the golden set against the
pre-committed bar.

## Reproduce it yourself

The demo corpus (16 synthetic memory files for a fictional platform team,
generated by `tools/gen_demo_corpus.py`) and a 20-query labeled set (15
standard + 5 adversarial) ship in the repo:

```bash
ollama pull embeddinggemma
python3 eval/eval_recall.py --corpus demo-corpus --queries eval/queries.jsonl
python3 eval/eval_recall.py diff        # after a second run: attribution
```

Measured on the demo corpus (Aug 23, Apple Silicon, Ollama):

| subset | n | R@1 | R@3 | MRR |
|---|---|---|---|---|
| full | 20 | 95% | 100% | 0.975 |
| standard | 15 | 100% | 100% | 1.000 |
| adversarial | 5 | 80% | 100% | 0.900 |

Demo-corpus numbers are not production numbers — sixteen small, distinct
files retrieve more cleanly than a real, grown corpus does. What reproduces
is the *pipeline*: embed → score → manifest → attribute.

## Install as a Claude Code hook

```bash
git clone https://github.com/wesglockzin/claude-code-session-memory
export CLAUDE_MEMORY_DIR=~/.claude/memory   # your memory directory
python3 session_memory.py --reindex          # bootstrap the index
python3 session_memory.py --query "anything" # sanity-check retrieval
```

Then register it in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 /path/to/session_memory.py"
          }
        ]
      }
    ]
  }
}
```

Requirements: Python ≥ 3.10, [Ollama](https://ollama.com) with
`embeddinggemma` pulled. No Python dependencies.

## Tests

```bash
pip install pytest && python -m pytest -q
```

The embedder is stubbed — what's under test is everything around the model:
the fail-open contract (run as a real subprocess against a dead Ollama
port), the render caps and floor, the supersession filter including its
fail-closed branch, regime rejection, the lazy-refresh cap, and the eval
math. CI runs them on every push, plus a check that the committed demo
corpus matches its generator.

## Known limitations

- **Paraphrase headroom is real and measured**: adversarial R@1 is 60% on my
  corpus. The recall log exists to decide, with data, whether a hybrid
  BM25+vector stage is worth its complexity. Not built until the misses
  justify it.
- **One embedder, pinned.** Changing model or regime invalidates the index
  (by design) and the floor (recalibrate — see the eval story).
- **Whole-file embedding**, truncated at 6 KB — right for atomic memory
  files, wrong for long documents; split those upstream.
- **The index is unencrypted JSON on disk**, same trust domain as the memory
  files themselves.
- **Recall quality is corpus-dependent.** Small, atomic, distinct files
  retrieve beautifully; sprawling multi-topic files don't. The demo corpus
  shows the shape that works.
- **Trust model: only index memory files you trust.** Retrieved bodies are
  injected into Claude's context verbatim (labeled as background, but a
  hostile memory file is still a prompt-injection vector — the hook retrieves,
  it doesn't sanitize). The recall log stores prompt *hashes*, never prompt
  text; the eval manifest stores repo-relative paths only.

## License

MIT — see [LICENSE](LICENSE).

## Author

Wes Glockzin
