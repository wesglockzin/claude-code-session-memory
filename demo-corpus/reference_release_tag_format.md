---
name: reference_release_tag_format
description: "Release tags are vMAJOR.MINOR.PATCH and the changelog is generated from them"
metadata:
  type: reference
---

Releases are cut by pushing an annotated tag named vMAJOR.MINOR.PATCH,
for example v2.4.1. The changelog is generated from the commits
between two tags, so a tag with a different shape is skipped and
its changes vanish from the release notes.
