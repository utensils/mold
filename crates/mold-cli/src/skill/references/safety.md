# Safety and authorization

## Secrets

- Never request that a user paste API keys, bearer tokens, private model
  credentials, or signing material into chat, source files, logs, or command
  history.
- Use an existing environment variable, keychain, credential file, or secret
  manager approved by the user. Confirm only that a value is present; do not
  print it.
- Redact credentials from diagnostics and URLs before reporting them.

## Destructive and paid operations

Read-only inspection does not authorize mutation. Obtain explicit user intent
for each material scope before any of these actions:

- deleting models, gallery media, trash, cloud pods, or network volumes;
- `mold clean --force`, bulk cancellation, or permanent purge;
- creating or resizing paid cloud resources;
- starting, stopping, restarting, or changing a shared/production server.

Resolve exact targets first with list/info/status commands. Prefer a dry run
when the CLI offers one. Before a broad action, state the target set and likely
effect. Afterward, verify the result from live state.

## Source media and privacy

Treat prompts, source images, videos, audio, and generated outputs as user data.
Do not upload or forward them to another host unless that host is in the user's
requested scope. When using a remote Mold server, say which host receives the
media and avoid embedding source bytes in logs or reports.

## Failure boundaries

- A transport error is not proof that a mutation failed; reconcile by exact
  resource or idempotency identity before retrying.
- Do not repeat paid provisioning, generation retry, cancellation, or deletion
  while the first result is uncertain.
- Stop and ask when the target, cost boundary, ownership, or recoverability is
  ambiguous.
