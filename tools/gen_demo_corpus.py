#!/usr/bin/env python3
"""Generate the demo corpus + labeled query set.

The demo corpus is a fictional platform team's memory directory — the same
shape as a real one (frontmatter, four memory types, one fact per file) with
none of anyone's real content. Committed to the repo so the eval is
reproducible byte-for-byte; this generator is the single source of truth.

Usage:  python3 tools/gen_demo_corpus.py   (writes demo-corpus/ and
        eval/queries.jsonl relative to the repo root)
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "demo-corpus")
QUERIES = os.path.join(ROOT, "eval", "queries.jsonl")

MEMORIES = {
    "feedback_deploy_in_phases": (
        "feedback",
        "Ship risky changes in phases with checkpoints, never as one batch",
        "Rolling out the payment-service config change to all regions at once\n"
        "caused the March incident. Rule: non-trivial live changes ship in\n"
        "phases — canary region first, verify dashboards, then the rest.\n"
        "**Why:** a bad batch is a full outage; a bad canary is a blip.\n"
        "**How to apply:** propose the phase plan before the first change.",
    ),
    "feedback_verify_dont_assume": (
        "feedback",
        "Verify against the live system before claiming a config exists",
        "Docs and memory go stale. Before stating that a queue, flag, or\n"
        "endpoint is configured a certain way, read the live config or call\n"
        "the API. **Why:** the team acted twice on a decommissioned worker\n"
        "pool that the wiki still described. **How to apply:** one read call\n"
        "before any assertion about live state.",
    ),
    "feedback_secrets_never_in_logs": (
        "feedback",
        "Secrets flow source-to-consumer in one step; never through logs or files",
        "Provisioning scripts must pass secrets by command substitution into\n"
        "the consuming command. Never stage them in a temp file or env dump —\n"
        "a failed run prints them into CI logs. **Why:** the staging token\n"
        "leaked exactly this way and had to be rotated fleet-wide.",
    ),
    "feedback_small_prs": (
        "feedback",
        "Keep pull requests under ~400 lines; split refactors from behavior changes",
        "Review quality collapses on big diffs. A mechanical rename travels\n"
        "in its own PR; the behavior change rides separately on top.\n"
        "**How to apply:** if a diff mixes both, split before requesting review.",
    ),
    "feedback_tests_before_migrations": (
        "feedback",
        "Schema migrations need a rehearsal run against a copy first",
        "Every schema migration runs against a restored copy of the database\n"
        "before it touches the live one, and the rollback script is written\n"
        "and rehearsed at the same time. **Why:** the widgets table lock in\n"
        "January held for 40 minutes because nobody had timed the migration.",
    ),
    "reference_vpn_cert_bundle": (
        "reference",
        "Corporate VPN does TLS inspection; tools need the internal CA bundle",
        "The corporate VPN intercepts TLS. Python tools must set\n"
        "REQUESTS_CA_BUNDLE to the internal bundle at ~/.corp-ca-bundle.pem\n"
        "or requests fail with certificate errors that look like outages.\n"
        "Node tools need NODE_EXTRA_CA_CERTS pointed at the same file.",
    ),
    "reference_build_cache_flag": (
        "reference",
        "CI build cache is keyed on the lockfile hash; stale caches need the bypass flag",
        "The CI cache key is the dependency lockfile hash. When a native\n"
        "dependency changes underneath the same lockfile, pass\n"
        "--no-dependency-cache once to force a clean build; the next run\n"
        "re-caches correctly. Symptom of a stale cache: linker errors that\n"
        "do not reproduce locally.",
    ),
    "reference_oncall_escalation": (
        "reference",
        "On-call escalation order and the paging threshold",
        "Sev-1 pages the primary on-call immediately; if unacknowledged in\n"
        "15 minutes it escalates to the secondary, then the team lead.\n"
        "Sev-2 waits for business hours. The paging threshold for the API\n"
        "error-rate monitor is 5% over 5 minutes — below that it only alerts\n"
        "the channel.",
    ),
    "reference_feature_flag_service": (
        "reference",
        "Feature flags are served by the flag service; local overrides via env file",
        "Flags come from the central flag service, cached 60 seconds. For\n"
        "local development, an overrides env file at ~/.flags.local takes\n"
        "precedence — remember to clear it, or you will debug a flag state\n"
        "no one else can see.",
    ),
    "reference_object_store_lifecycle": (
        "reference",
        "Uploaded artifacts expire after 90 days unless tagged keep-forever",
        "The artifact bucket has a 90-day lifecycle rule. Anything that must\n"
        "survive (release binaries, audit exports) needs the keep-forever\n"
        "tag at upload time; retro-tagging works only while the object still\n"
        "exists. The weekly job that reaped the compliance exports was this\n"
        "rule, not a bug.",
    ),
    "project_search_reindex": (
        "project",
        "Search reindex project: dual-write cutover, verify counts, then flip reads",
        "Migrating the search cluster: stage 1 dual-writes to old and new\n"
        "indexes; stage 2 compares document counts and spot-checks queries\n"
        "nightly; stage 3 flips reads behind a flag with instant rollback.\n"
        "Currently in stage 2 — counts match within 0.1% for a week.",
    ),
    "project_login_latency": (
        "project",
        "Login latency project: p99 target 800 ms, current bottleneck is the profile call",
        "The login p99 is 2.1 s against an 800 ms target. Tracing shows 60%\n"
        "of the budget in the synchronous profile-service call, which can be\n"
        "deferred. Plan: return the session first, hydrate the profile\n"
        "asynchronously, measure, then decide about the avatar CDN.",
    ),
    "project_cost_dashboard": (
        "project",
        "Cloud cost dashboard: per-team attribution shipped, anomaly alerts next",
        "The cost dashboard now attributes spend per team via resource tags;\n"
        "untagged spend fell from 40% to 6%. Next milestone: anomaly alerts\n"
        "when a team's daily spend deviates more than 3 sigma from its\n"
        "trailing 30-day mean.",
    ),
    "user_maintainer_profile": (
        "user",
        "Maintainer prefers concise answers, phased rollouts, and explicit error handling",
        "The maintainer reviews with three standing expectations: lead with\n"
        "the answer, ship live changes in phases, and treat silent error\n"
        "handling as a defect — every failure path names what broke and how\n"
        "to fix it.",
    ),
    "reference_retired_worker_pool": (
        "reference",
        "RETIRED: the legacy worker pool was decommissioned; jobs run on the queue service",
        "The legacy cron worker pool is gone. All background jobs run on the\n"
        "queue service with per-job retry policies. This file is kept as the\n"
        "historical record of the cutover.",
    ),
    "reference_queue_service_jobs": (
        "reference",
        "Background jobs: queue service, per-job retry policy, dead-letter after 5 attempts",
        "Background jobs run on the queue service. Each job declares its own\n"
        "retry policy; the default is exponential backoff, dead-letter after\n"
        "five attempts. The dead-letter queue is drained by a weekly review,\n"
        "not automatically — a poisoned job must be seen by a human.",
    ),
}

# The retired-pool file gets a lifecycle marker so the supersession filter
# has something real to exclude in the demo, mirroring production behavior.
LIFECYCLE = {"reference_retired_worker_pool": "superseded"}

# Query set: 15 author-style queries + 5 adversarial paraphrases written for
# near-zero keyword overlap with their gold memory (marked adversarial=true).
QUERY_SET = [
    ("how should risky changes be rolled out", "feedback_deploy_in_phases", False),
    (
        "check the live system before saying something is configured",
        "feedback_verify_dont_assume",
        False,
    ),
    (
        "how do secrets get passed in provisioning scripts",
        "feedback_secrets_never_in_logs",
        False,
    ),
    ("how big should a pull request be", "feedback_small_prs", False),
    (
        "what do we do before running a schema migration",
        "feedback_tests_before_migrations",
        False,
    ),
    (
        "python requests certificate errors on the corporate network",
        "reference_vpn_cert_bundle",
        False,
    ),
    (
        "CI build fails with linker errors that work locally",
        "reference_build_cache_flag",
        False,
    ),
    (
        "who gets paged for a sev-1 and when does it escalate",
        "reference_oncall_escalation",
        False,
    ),
    (
        "where do feature flags come from in local development",
        "reference_feature_flag_service",
        False,
    ),
    (
        "uploaded files disappearing after three months",
        "reference_object_store_lifecycle",
        False,
    ),
    ("status of the search cluster migration", "project_search_reindex", False),
    ("why is login slow and what is the plan", "project_login_latency", False),
    ("per-team cloud spend attribution progress", "project_cost_dashboard", False),
    ("how does the maintainer like to work", "user_maintainer_profile", False),
    ("where do background jobs run now", "reference_queue_service_jobs", False),
    # adversarial: paraphrases avoiding the gold memory's vocabulary
    ("blast radius small first then widen", "feedback_deploy_in_phases", True),
    ("trust but confirm against reality", "feedback_verify_dont_assume", True),
    (
        "credentials ended up visible in the pipeline output",
        "feedback_secrets_never_in_logs",
        True,
    ),
    ("sign-in takes seconds; users complain", "project_login_latency", True),
    ("stuff we archive keeps vanishing", "reference_object_store_lifecycle", True),
]


def main() -> None:
    os.makedirs(CORPUS, exist_ok=True)
    os.makedirs(os.path.dirname(QUERIES), exist_ok=True)
    for stem, (mtype, desc, body) in MEMORIES.items():
        lifecycle = ""
        if stem in LIFECYCLE:
            lifecycle = f"lifecycle: {LIFECYCLE[stem]}\n"
        content = (
            "---\n"
            f"name: {stem}\n"
            f'description: "{desc}"\n'
            f"{lifecycle}"
            "metadata:\n"
            f"  type: {mtype}\n"
            "---\n\n"
            f"{body}\n"
        )
        with open(os.path.join(CORPUS, f"{stem}.md"), "w", encoding="utf-8") as f:
            f.write(content)
    with open(QUERIES, "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"query": q, "gold": gold, "adversarial": adversarial})
                + "\n" for q, gold, adversarial in QUERY_SET)
    print(f"wrote {len(MEMORIES)} memories -> {CORPUS}")
    print(f"wrote {len(QUERY_SET)} queries  -> {QUERIES}")


if __name__ == "__main__":
    main()
