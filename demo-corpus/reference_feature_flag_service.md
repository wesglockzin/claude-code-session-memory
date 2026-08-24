---
name: reference_feature_flag_service
description: "Feature flags are served by the flag service; local overrides via env file"
metadata:
  type: reference
---

Flags come from the central flag service, cached 60 seconds. For
local development, an overrides env file at ~/.flags.local takes
precedence — remember to clear it, or you will debug a flag state
no one else can see.
