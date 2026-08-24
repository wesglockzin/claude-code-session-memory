---
name: reference_object_store_lifecycle
description: "Uploaded artifacts expire after 90 days unless tagged keep-forever"
metadata:
  type: reference
---

The artifact bucket has a 90-day lifecycle rule. Anything that must
survive (release binaries, audit exports) needs the keep-forever
tag at upload time; retro-tagging works only while the object still
exists. The weekly job that reaped the compliance exports was this
rule, not a bug.
