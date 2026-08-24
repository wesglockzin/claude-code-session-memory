---
name: feedback_tests_before_migrations
description: "Schema migrations need a rehearsal run against a copy first"
metadata:
  type: feedback
---

Every schema migration runs against a restored copy of the database
before it touches the live one, and the rollback script is written
and rehearsed at the same time. **Why:** the widgets table lock in
January held for 40 minutes because nobody had timed the migration.
