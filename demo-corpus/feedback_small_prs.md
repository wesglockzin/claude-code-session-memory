---
name: feedback_small_prs
description: "Keep pull requests under ~400 lines; split refactors from behavior changes"
metadata:
  type: feedback
---

Review quality collapses on big diffs. A mechanical rename travels
in its own PR; the behavior change rides separately on top.
**How to apply:** if a diff mixes both, split before requesting review.
