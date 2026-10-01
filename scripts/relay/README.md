# Lambda relay deployment

Terraform in URandom owns AWS configuration. This helper builds separate Node
bundles from `relay/aws/router.mjs` and `relay/aws/frontend.mjs`, packages each
as `index.mjs` with handler `index.handler`, and retains ZIPs for rollback.
It changes Lambda code only, uses RevisionId to reject concurrent updates,
waits for successful completion, and verifies AWS's base64 SHA256 of each ZIP.
It verifies function configuration was preserved. A failed deployment stops;
inspect its state before retrying. Two function updates are not atomic.

Build the browser shell first (`bun run --cwd web build`). To review packages
without contacting AWS:

```sh
python3 scripts/relay/deploy.py --build-only --output /tmp/mold-relay-reviewed-build
```

Deploy reviewed source and the built shell with a fresh artifact directory:

```sh
python3 scripts/relay/deploy.py \
  --output /private/operator/path/relay-release \
  --host-token-file /private/operator/path/host-token \
  --bridge-token-file /private/operator/path/bridge-token
```

Token files must be separate owner-only regular files. Existing SecureStrings
are reused, with absent local files saved mode 600. Missing parameters are
created from an existing token file or a new random token. A mismatch fails;
there is no automatic token rotation. Secret parameter values travel through
private temporary JSON files, never command arguments or logs. Only the host
connector should receive the host token; the bridge token stays in AWS/local
operator storage. Tokens are independent of Mold API keys.

Defaults target profile `dev.urandom.io`, region `us-west-2`, functions
`mold-relay-router`/`mold-relay-frontend`, and bucket
`mold-relay-042506291754-us-west-2`. Override profile/region/bucket only for an
explicitly provisioned deployment. AWS CLI and Bun are required; AWS credentials
must authorize code updates, the two exact SSM parameters, and shell uploads.

Shell upload writes only `shell/` with MIME types and AES256 encryption. It never
sync-deletes assets or modifies uploads/media prefixes. Retaining old hashed
assets allows cached browser shells to continue loading after deployment.
No infrastructure or real-machine registration is performed by this helper.
