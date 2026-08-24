---
name: feedback_secrets_never_in_logs
description: "Secrets flow source-to-consumer in one step; never through logs or files"
metadata:
  type: feedback
---

Provisioning scripts must pass secrets by command substitution into
the consuming command. Never stage them in a temp file or env dump —
a failed run prints them into CI logs. **Why:** the staging token
leaked exactly this way and had to be rotated fleet-wide.
