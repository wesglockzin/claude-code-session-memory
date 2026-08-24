---
name: project_search_reindex
description: "Search reindex project: dual-write cutover, verify counts, then flip reads"
metadata:
  type: project
---

Migrating the search cluster: stage 1 dual-writes to old and new
indexes; stage 2 compares document counts and spot-checks queries
nightly; stage 3 flips reads behind a flag with instant rollback.
Currently in stage 2 — counts match within 0.1% for a week.
