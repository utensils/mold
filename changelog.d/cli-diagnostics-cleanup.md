- **`mold server status` reports on the host you selected.** `--host` (or
  `MOLD_HOST`) now reads the named server's status over HTTP instead of
  answering from this machine's PID file, so a status call against a remote
  host no longer prints "No server running" about a machine it never
  contacted. An unreachable host exits non-zero; PID, port and log path still
  appear only for the local managed daemon.
- **A failed metadata-DB migration reports one line, not a wall of SQL.** A DB
  stuck on a version conflict used to dump the entire failing `CREATE TABLE`
  block into the middle of an ordinary run; it now names the migration, the
  version the DB is stuck at, and what SQLite objected to.
