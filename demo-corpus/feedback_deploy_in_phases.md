---
name: feedback_deploy_in_phases
description: "Ship risky changes in phases with checkpoints, never as one batch"
metadata:
  type: feedback
---

Rolling out the payment-service config change to all regions at once
caused the March incident. Rule: non-trivial live changes ship in
phases — canary region first, verify dashboards, then the rest.
**Why:** a bad batch is a full outage; a bad canary is a blip.
**How to apply:** propose the phase plan before the first change.
