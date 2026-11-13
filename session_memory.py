"""
session_memory.py — local RAG memory recall for Claude Code.

A UserPromptSubmit hook: embeds the incoming prompt on-device (Ollama +
EmbeddingGemma), cosine-matches it against a vector index of your memory
directory, and prints the top-K relevant memory files — which Claude Code
injects as context for that prompt. A hand-maintained index gives breadth
every session; this adds depth per prompt, retrieved by meaning.

Contract (load-bearing — every clause exists because its absence bit):
  - FAIL-OPEN: any failure (Ollama down, index missing, bad payload) → print
    nothing, exit 0, never block the prompt. Errors append to ERR_LOG.
  - TOKEN-CAPPED: at most K memories and TOTAL_CHAR_CAP chars injected.
  - FLOOR: prompts with no sufficiently-similar memory inject nothing.
  - ZERO third-party deps: stdlib only. A per-prompt hook must not depend on
    a venv. Pure-python cosine over a few hundred 768-dim vectors is ~10 ms.

Freshness is lazy: at query time, changed/new memory files are re-embedded
(capped at MAX_LAZY_REEMBED per prompt) and deleted files drop out. No
separate file-watcher needed. Full rebuild: --reindex.

Configuration (environment):
  CLAUDE_MEMORY_DIR      directory of *.md memory files
                         (default: ~/.claude/memory)
  SESSION_MEMORY_STATE   where the index + logs live
                         (default: ~/.claude/session-memory)
  SESSION_MEMORY_EXCLUDE comma-separated basenames to skip
                         (default: MEMORY.md)
  OLLAMA_HOST            Ollama base URL (REQUIRED, no default)
  SESSION_MEMORY_MODEL   embedding model name (REQUIRED, no default)

Modes:
  (no args)        hook mode — reads the UserPromptSubmit JSON payload on stdin
  --reindex        full bootstrap/rebuild of the index
  --query "text"   test mode — prints ranked results with scores, injects nothing
"""

import hashlib
import json
import logging
import math
import os
import shlex
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

# --- INTERPRETER GUARD -------------------------------------------------------
# This module uses PEP-604 annotations (`list[float] | None`), which are
# evaluated at def-time. Under Python 3.9 that raises TypeError at import —
# and because this is a fail-open hook, the crash would be indistinguishable
# from "no relevant memory found." macOS still ships /usr/bin/python3 as 3.9,
# so a PATH shift alone could silently kill recall. Fail CLOSED and LOUD here
# instead: this is the one failure mode that must never be quiet.
if sys.version_info < (3, 10):  # noqa: UP036 - hook must fail loud on an old interpreter
    sys.stderr.write(
        "session_memory: FATAL — needs Python >= 3.10; "
        f"got {sys.version.split()[0]} at {sys.executable}. "
        "Point the hook at a newer interpreter. Recall is DISABLED until fixed.\n"
    )
    sys.exit(2)
# -----------------------------------------------------------------------------
# Required settings: no defaults in code (they are checked by require_settings()
# at startup, which names the missing variable).
OLLAMA = os.environ.get("OLLAMA_HOST", "").strip().rstrip("/")
MODEL = os.environ.get("SESSION_MEMORY_MODEL", "").strip()


def require_settings() -> None:
    missing = [
        name
        for name, value in (("OLLAMA_HOST", OLLAMA), ("SESSION_MEMORY_MODEL", MODEL))
        if not value
    ]
    if missing:
        sys.exit(f"session_memory: required setting(s) not set: {', '.join(missing)}")


# --- EMBEDDING REGIME --------------------------------------------------------
# EmbeddingGemma is trained with asymmetric instruction prefixes (per the
# model card): queries and documents embed DIFFERENTLY. Ollama passes text to
# the model raw, so the caller owns the prefixes. On my production corpus the
# prefixed regime beat raw embeddings 83%/93%/0.893 vs 70%/90%/0.810 on the
# full golden set, and doubled adversarial-paraphrase R@1 (60% vs 40%).
#
# REGIME stamps the index. Prefixed and raw vectors score in different bands
# and are NOT comparable, so an index built under another regime is rejected
# wholesale by load_index() — a mixed index is structurally impossible, at
# the cost of recall being disabled until --reindex is run.
REGIME = "prefix-title-v1"
QUERY_PREFIX = "task: search result | query: "


def doc_prefix(stem: str) -> str:
    return f"title: {stem.replace('_', ' ')} | text: "


MEM_DIR = os.path.expanduser(os.environ.get("CLAUDE_MEMORY_DIR", "~/.claude/memory"))
EXCLUDE = {
    s.strip()
    for s in os.environ.get("SESSION_MEMORY_EXCLUDE", "MEMORY.md").split(",")
    if s.strip()
}

STATE_DIR = os.path.expanduser(
    os.environ.get("SESSION_MEMORY_STATE", "~/.claude/session-memory")
)
INDEX_PATH = os.path.join(STATE_DIR, "index.json")
ERR_LOG = os.path.join(STATE_DIR, "hook.err.log")
RECALL_LOG = os.path.join(STATE_DIR, "recall_log.jsonl")

K = 4  # max memories injected per prompt
FLOOR = 0.30  # similarity floor. Calibrated on my production index
# under prefix-title-v1: gold top-1 scores bottomed at
# 0.356, negative top-1 scores peaked at 0.229 —
# floor sits in the (0.229, 0.356) gap. A floor
# calibrated under one regime does NOT transfer to
# another; recalibrate if you change REGIME or MODEL.
TOTAL_CHAR_CAP = 6000  # caps injected memory-BODY chars (~1.5k tokens);
# the one-line header and per-memory separators are
# not counted (bounded: K short lines)
PER_MEM_CHAR_CAP = 2000  # per-memory body cap
MAX_CHARS_EMBED = 6000  # chars of a memory file fed to the embedder
HOOK_TIMEOUT = 3.0  # seconds per Ollama call in hook mode
REINDEX_TIMEOUT = 120.0  # seconds per Ollama call in reindex mode
MAX_LAZY_REEMBED = 3  # changed files re-embedded per prompt; rest wait
MIN_PROMPT_LEN = 12  # skip trivial prompts ("go", "ok", "yes")
LOG_TOP_N = 10  # candidates recorded per prompt in the recall log.
# Injection stays K; the wider slice exists so a
# rank-5..10 miss is visible to the miss log —
# logging only top-K hides exactly the misses the
# log exists to capture (adversarial-review finding).


class _ErrLogHandler(logging.Handler):
    """Appends records to ERR_LOG (resolved at emit time). In hook mode stdout
    is the injection channel, so hook diagnostics go to this file."""

    def emit(self, record: logging.LogRecord) -> None:
        ts = datetime.fromtimestamp(record.created, timezone.utc).isoformat()
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
            with open(ERR_LOG, "a", encoding="utf-8") as f:
                f.write(f"{ts} {record.getMessage()}\n")
        except OSError:
            self.handleError(record)  # the log file is unwritable: report on stderr


logger = logging.getLogger("session_memory")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.addHandler(_ErrLogHandler(level=logging.WARNING))

# The most recent embed failure, so callers can report what actually happened.
LAST_EMBED = {"error": ""}


def log_err(msg: str) -> None:
    logger.error(msg)


def embed(text: str, timeout: float) -> list[float] | None:
    """Embed via Ollama; newer /api/embed with /api/embeddings fallback.

    Failures are logged with their kind: an HTTP error is reported with its
    status and body, distinct from a connection failure. The last failure is
    kept in LAST_EMBED["error"]."""
    LAST_EMBED["error"] = ""
    for url, body, pick in (
        (
            f"{OLLAMA}/api/embed",
            {"model": MODEL, "input": text},
            lambda d: (d.get("embeddings") or [None])[0] or d.get("embedding"),
        ),
        (
            f"{OLLAMA}/api/embeddings",
            {"model": MODEL, "prompt": text},
            lambda d: d.get("embedding"),
        ),
    ):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                vec = pick(json.loads(r.read()))
            if vec:
                return vec
        except urllib.error.HTTPError as e:
            detail = e.read(300).decode("utf-8", errors="replace")
            LAST_EMBED["error"] = f"HTTP {e.code} from {url}: {detail}"
            log_err(f"embed: {LAST_EMBED['error']}")
        except (urllib.error.URLError, OSError) as e:
            LAST_EMBED["error"] = f"{url} unreachable: {type(e).__name__}: {e}"
            log_err(f"embed: {LAST_EMBED['error']}")
        except ValueError as e:
            LAST_EMBED["error"] = f"bad response from {url}: {type(e).__name__}: {e}"
            log_err(f"embed: {LAST_EMBED['error']}")
    return None


def normed(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec))
    return [x / n for x in vec] if n else vec


def read_memory(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def embed_entry(path: str, stem: str, mtime: float, timeout: float) -> dict | None:
    """One body vector per memory file. A dual-vector variant (body +
    frontmatter description, max-pooled) was measured and REJECTED: the
    description vector ranked far worse on paraphrases and scored hotter on
    negatives, which breaks a single similarity floor."""
    body = read_memory(path)
    # Prefix AFTER truncation so the instruction prefix is never cut off.
    vec = embed(doc_prefix(stem) + body[:MAX_CHARS_EMBED], timeout)
    if vec is None:
        return None
    return {"path": path, "mtime": mtime, "vec": normed(vec)}


def load_index() -> dict:
    try:
        with open(INDEX_PATH, encoding="utf-8") as f:
            idx = json.load(f)
        if idx.get("model") == MODEL and idx.get("regime") == REGIME:
            return idx
        log_err(
            f"index model/regime {idx.get('model')!r}/{idx.get('regime')!r} "
            f"!= {MODEL!r}/{REGIME!r}; ignoring index — run --reindex"
        )
    except FileNotFoundError:
        pass
    except Exception as e:  # noqa: BLE001 - fail-open: a corrupt index is logged and rebuilt
        log_err(f"load_index:{type(e).__name__}: {e}")
    return {"model": MODEL, "regime": REGIME, "entries": {}}


def save_index(idx: dict) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = f"{INDEX_PATH}.{os.getpid()}.tmp"  # per-process: parallel hooks don't clash
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(idx, f)
    os.replace(tmp, INDEX_PATH)  # same dir → same device; atomic


def corpus_files() -> dict[str, str]:
    """stem -> absolute path for every eligible memory file."""
    out = {}
    for name in os.listdir(MEM_DIR):
        if name.endswith(".md") and name not in EXCLUDE:
            out[name[:-3]] = os.path.join(MEM_DIR, name)
    return out


def refresh(idx: dict, timeout: float, max_reembeds: int) -> bool:
    """Drop deleted files; re-embed changed/new ones (capped). Returns dirty."""
    files = corpus_files()
    entries = idx["entries"]
    dirty = False

    for stem in list(entries):
        if stem not in files:
            del entries[stem]
            dirty = True

    stale = []
    for stem, path in files.items():
        try:
            mtime = os.path.getmtime(path)
        except OSError as e:
            log_err(f"refresh: cannot stat {stem}: {type(e).__name__}: {e}")
            continue
        e = entries.get(stem)
        if not e or e.get("mtime") != mtime:
            stale.append((stem, path, mtime))

    for stem, path, mtime in stale[:max_reembeds]:
        entry = embed_entry(path, stem, mtime, timeout)
        if entry is None:
            log_err(f"refresh: embed failed for {stem}; keeping stale/absent")
            break  # Ollama likely down — don't hammer it for the rest
        entries[stem] = entry
        dirty = True
    return dirty


def is_superseded(path: str, stem: str) -> bool:
    """Lifecycle read from the SOURCE OF TRUTH (on-disk frontmatter) at query
    time — deliberately NOT cached in the index.

    Why not cached: refresh() rebuilds an entry as {path, mtime, vec} on every
    re-embed, so a cached lifecycle flag would silently vanish the next time
    the file changed — a guard that stops applying without anyone noticing.
    The index is a cache; the file is the truth. The cost is a stdlib
    frontmatter parse, not a model call.

    Three cases, none of which may vanish silently:
      clean read + `lifecycle: superseded`  -> True  (exclude)
      clean read + no marker                -> False (current)
      read error                            -> True + log_err (fail CLOSED)

    Fail-closed is coherent here because render() reads the body from this
    same path: a file that cannot be read cannot be injected anyway, so
    excluding it costs nothing and never risks injecting a retired document.

    Matches ONLY an explicit `lifecycle:` frontmatter field. Prose is not
    parsed — "supersedes X" and "superseded by X" are indistinguishable by
    keyword match.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            head = f.read(4000)
        if not head.startswith("---"):
            return False
        end = head.find("\n---", 3)
        fm = head[3:end] if end != -1 else head[3:]
        for line in fm.splitlines():
            s = line.strip().lower()
            if s.startswith("lifecycle:"):
                val = s.split(":", 1)[1]
                # Strip YAML inline comments: `lifecycle: superseded  # note`
                # must still match, or the filter silently stops applying —
                # the exact failure class this function exists to remove.
                val = val.split("#", 1)[0].strip().strip('"').strip("'")
                return val == "superseded"
        return False
    except Exception as e:  # noqa: BLE001 - logged, and the file is excluded (fail-closed)
        log_err(
            f"is_superseded({stem}): {type(e).__name__}: {e} — EXCLUDED (fail-closed)"
        )
        return True


def top_k(
    idx: dict, qvec: list[float], n: int | None = None
) -> list[tuple[float, str, dict]]:
    """Top-n scored candidates (default K). Injection always uses the first K;
    the wider slice exists only for the recall log (see LOG_TOP_N)."""
    scored = []
    for stem, e in idx["entries"].items():
        if len(e["vec"]) != len(qvec):
            continue
        # PRE-FILTER: exclude retired docs BEFORE scoring, so a superseded
        # document never competes for a top-K slot.
        if is_superseded(e["path"], stem):
            continue
        scored.append((sum(a * b for a, b in zip(e["vec"], qvec)), stem, e))
    scored.sort(key=lambda t: -t[0])
    return scored[: (n or K)]


def render(hits: list[tuple[float, str, dict]]) -> tuple[str, int]:
    """Returns (text, injected_count) — injected_count is the REAL number of
    memories that made it into the output after the floor + budget filters,
    not len(hits). A caller that logged len(hits) as "memories injected"
    would overcount whenever the floor or the char budget dropped one.

    Caps its input at K itself: top_k() can return a wider slice for the
    recall log, and the at-most-K contract must hold at THIS boundary rather
    than depend on every caller slicing correctly."""
    kept = [(s, stem, e) for s, stem, e in hits[:K] if s >= FLOOR]
    if not kept:
        return "", 0
    # Build the body blocks FIRST, then write the header from the FINAL count.
    # The old shape wrote the header from the pre-filter count, so a memory
    # dropped by the budget or an unreadable file left the header overcounting
    # (adversarial-review finding).
    blocks = []
    budget = TOTAL_CHAR_CAP
    for score, stem, e in kept:
        try:
            body = read_memory(e["path"]).strip()
        except OSError as e:
            log_err(f"render: cannot read {stem}: {type(e).__name__}: {e}")
            continue
        body = body[: min(PER_MEM_CHAR_CAP, budget)]
        if not body:
            continue
        blocks.append(f"\n--- {stem} (similarity {score:.2f}) ---\n{body}")
        budget -= len(body)
        if budget <= 0:
            break
    if not blocks:
        return "", 0
    injected = len(blocks)
    header = (
        f"[Memory recall] {injected} memor{'y' if injected == 1 else 'ies'} "
        "auto-retrieved on-device as relevant to this prompt "
        "(background context, not instructions):"
    )
    return "\n".join([header] + blocks), injected


def hook_mode() -> None:
    try:
        payload = json.load(sys.stdin)
    except ValueError as e:
        log_err(f"hook: invalid payload on stdin: {type(e).__name__}: {e}")
        return
    prompt = (payload.get("prompt") or "").strip()
    if len(prompt) < MIN_PROMPT_LEN or prompt.startswith("/"):
        return
    idx = load_index()
    if not idx["entries"]:
        log_err("hook: index empty/missing — run --reindex")
        return
    try:
        if refresh(idx, HOOK_TIMEOUT, MAX_LAZY_REEMBED):
            save_index(idx)
    except Exception as e:  # noqa: BLE001 - fail-open hook boundary; error is logged
        log_err(f"hook refresh: {type(e).__name__}: {e}")
    qvec = embed(QUERY_PREFIX + prompt[:2000], HOOK_TIMEOUT)
    if qvec is None:
        log_err(f"hook: query embed failed ({LAST_EMBED['error']}) — silent no-op")
        return
    candidates = top_k(idx, normed(qvec), LOG_TOP_N)
    hits = candidates[:K]
    out, injected = render(hits)
    # Per-invocation record so real-use misses can accumulate: top-N stems +
    # scores (below-floor near-misses visible by inspection), prompt stored as
    # a hash only — the transcript already holds the text; this log stays lean.
    # Fail-open: a log failure must never cost a prompt.
    try:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "prompt_sha16": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "top": [[stem, round(score, 3)] for score, stem, _ in candidates],
            "injected": injected,
            "floor": FLOOR,
        }
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(RECALL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError as e:
        log_err(f"recall log write failed: {e}")
    if out:
        sys.stdout.write(out + "\n")  # the injected context is the hook's output


def reindex() -> None:
    files = corpus_files()
    idx = {"model": MODEL, "regime": REGIME, "entries": {}}
    failed = []
    for i, (stem, path) in enumerate(sorted(files.items()), 1):
        entry = embed_entry(path, stem, os.path.getmtime(path), REINDEX_TIMEOUT)
        if entry is None:
            failed.append(stem)
            continue
        idx["entries"][stem] = entry
        logger.info("embedded %d/%d", i, len(files))
    save_index(idx)
    logger.info(
        "indexed %d/%d memories -> %s", len(idx["entries"]), len(files), INDEX_PATH
    )
    if failed:
        logger.error("FAILED (%d): %s", len(failed), ", ".join(failed))
        sys.exit(f"reindex failed: {LAST_EMBED['error']}")


def query_mode(text: str) -> None:
    idx = load_index()
    if not idx["entries"]:
        sys.exit("index empty — run --reindex first")
    # Operator mode: refresh EVERYTHING stale (no lazy cap — that cap exists
    # to protect per-prompt latency, which doesn't apply here) and persist,
    # so what this prints is what the hook will actually see.
    if refresh(idx, REINDEX_TIMEOUT, max_reembeds=10**6):
        save_index(idx)
    qvec = embed(QUERY_PREFIX + text, REINDEX_TIMEOUT)
    if qvec is None:
        sys.exit(f"embed failed: {LAST_EMBED['error']}")
    for score, stem, _ in top_k(idx, normed(qvec)):
        marker = "INJECT" if score >= FLOOR else "below-floor"
        sys.stdout.write(f"  {score:.3f}  {marker:<12} {stem}\n")


USAGE = """\
usage: session-memory [--reindex | --query TEXT | --install-hook | --help]

  (no args)       hook mode: read a UserPromptSubmit JSON payload on stdin and
                  print the relevant memories to inject
  --reindex       rebuild the memory index
  --query TEXT    print ranked results with scores; injects nothing
  --install-hook  add the UserPromptSubmit hook to ~/.claude/settings.json
                  (or the file named by CLAUDE_SETTINGS_PATH)
  -h, --help      show this help

Required settings: OLLAMA_HOST, SESSION_MEMORY_MODEL (see .env.example).
"""


def install_hook(settings_path: str, command: str) -> bool:
    """Merge a UserPromptSubmit hook running `command` into a Claude Code
    settings file. Returns False if it was already present (idempotent)."""
    settings: dict = {}
    if os.path.exists(settings_path):
        with open(settings_path, encoding="utf-8") as f:
            settings = json.load(f)
    groups = settings.setdefault("hooks", {}).setdefault("UserPromptSubmit", [])
    for group in groups:
        for h in group.get("hooks", []):
            if h.get("command") == command:
                return False
    groups.append({"hooks": [{"type": "command", "command": command}]})
    os.makedirs(os.path.dirname(settings_path) or ".", exist_ok=True)
    tmp = f"{settings_path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
        f.write("\n")
    os.replace(tmp, settings_path)
    return True


def install_hook_mode() -> None:
    path = os.path.expanduser(
        os.environ.get("CLAUDE_SETTINGS_PATH", "~/.claude/settings.json")
    )
    exe = os.path.abspath(sys.argv[0])
    command = shlex.quote(
        exe if os.path.basename(exe) == "session-memory" else "session-memory"
    )
    try:
        added = install_hook(path, command)
    except (OSError, ValueError) as e:
        sys.exit(f"install-hook failed for {path}: {type(e).__name__}: {e}")
    logger.info(
        "%s hook %s in %s", command, "added" if added else "already present", path
    )


def main(argv: list[str] | None = None) -> None:
    """Console-script entry point (`session-memory`)."""
    argv = sys.argv[1:] if argv is None else argv
    if "-h" in argv or "--help" in argv:
        sys.stdout.write(USAGE)
        sys.exit(0)
    # Operator modes report progress and failures on stderr; hook mode keeps
    # stdout for the injected context and logs to ERR_LOG only.
    if argv and not any(type(h) is logging.StreamHandler for h in logger.handlers):
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console)
    # Missing required settings fail at startup, naming the setting.
    require_settings()
    if "--install-hook" in argv:
        install_hook_mode()
        return
    # Fail-open applies to HOOK MODE ONLY. The CLI modes are operator tools:
    # a --reindex against a missing directory must fail loudly with a nonzero
    # exit, not pretend success (adversarial-review finding: the old shape
    # swallowed CLI errors too, so a broken bootstrap looked like a clean one).
    if "--reindex" in argv:
        reindex()
    elif "--query" in argv:
        pos = argv.index("--query") + 1
        if pos >= len(argv):
            sys.exit("usage: session-memory --query TEXT")
        query_mode(argv[pos])
    else:
        try:
            hook_mode()
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001 - last-resort boundary; hook is fail-open and logs
            log_err(f"hook top-level: {type(e).__name__}: {e}")
        sys.exit(0)  # fail-open, always


if __name__ == "__main__":
    main()
