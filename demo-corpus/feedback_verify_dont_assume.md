---
name: feedback_verify_dont_assume
description: "Verify against the live system before claiming a config exists"
metadata:
  type: feedback
---

Docs and memory go stale. Before stating that a queue, flag, or
endpoint is configured a certain way, read the live config or call
the API. **Why:** the team acted twice on a decommissioned worker
pool that the wiki still described. **How to apply:** one read call
before any assertion about live state.
