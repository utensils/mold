- **A queue-media permission problem now says how to fix it, and keeps saying
  it.** The encrypted store under `$MOLD_HOME/queue-media` still refuses any
  path that is not owned by the service user at `0700` — it holds the master
  key — but the refusal now names the observed owner and mode beside the
  expected ones and prints the exact shell-quoted `chown`/`chmod` repair,
  instead of restating the requirement. Anything that walks the Mold data root (an ACL
  pass, `chmod -R`, a restore that drops modes, `rsync` without `-p`) widens
  that directory and switches restart-safe request media off, so the
  degradation is no longer a single startup line: every reason is logged in
  full at `WARN`, authenticated `GET /api/status` gains `durable_media`
  (`available` plus the reasons, retained for the life of the process), and
  auth-exempt `GET /health` gains a body naming the degraded subsystem —
  `{"status":"degraded","degraded":["durable_media"]}`. `/health` still answers
  `200` while degraded and never waits on a lock, because generation is
  unaffected and a failing or blocking check would pull a working server out of
  a load balancer; a host that never offers the feature at all reports nothing
  rather than reading as broken
  ([#1402](https://github.com/utensils/mold/issues/1402)).
