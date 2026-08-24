---
name: reference_build_cache_flag
description: "CI build cache is keyed on the lockfile hash; stale caches need the bypass flag"
metadata:
  type: reference
---

The CI cache key is the dependency lockfile hash. When a native
dependency changes underneath the same lockfile, pass
--no-dependency-cache once to force a clean build; the next run
re-caches correctly. Symptom of a stale cache: linker errors that
do not reproduce locally.
