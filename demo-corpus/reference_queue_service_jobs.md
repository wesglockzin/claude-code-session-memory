---
name: reference_queue_service_jobs
description: "Background jobs: queue service, per-job retry policy, dead-letter after 5 attempts"
metadata:
  type: reference
---

Background jobs run on the queue service. Each job declares its own
retry policy; the default is exponential backoff, dead-letter after
five attempts. The dead-letter queue is drained by a weekly review,
not automatically — a poisoned job must be seen by a human.
