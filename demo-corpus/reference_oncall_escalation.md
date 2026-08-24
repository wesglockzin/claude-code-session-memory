---
name: reference_oncall_escalation
description: "On-call escalation order and the paging threshold"
metadata:
  type: reference
---

Sev-1 pages the primary on-call immediately; if unacknowledged in
15 minutes it escalates to the secondary, then the team lead.
Sev-2 waits for business hours. The paging threshold for the API
error-rate monitor is 5% over 5 minutes — below that it only alerts
the channel.
