---
name: project_login_latency
description: "Login latency project: p99 target 800 ms, current bottleneck is the profile call"
metadata:
  type: project
---

The login p99 is 2.1 s against an 800 ms target. Tracing shows 60%
of the budget in the synchronous profile-service call, which can be
deferred. Plan: return the session first, hydrate the profile
asynchronously, measure, then decide about the avatar CDN.
