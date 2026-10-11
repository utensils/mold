# Lambda relay deployment

Terraform in URandom owns AWS configuration. This helper builds separate Node
bundles for the opaque WebSocket router and metadata-only frontend. Deployment
changes code only, uses RevisionId to reject concurrent updates, verifies ZIP
SHA256 and checks that configuration stayed unchanged. Two function updates are
not atomic; inspect any failed/unconfirmed update before retrying.

Build without contacting AWS:

```sh
python3 scripts/relay/deploy.py --build-only --output /tmp/mold-relay-reviewed-build
```

Deploy reviewed source with a fresh private artifact directory:

```sh
python3 scripts/relay/deploy.py --output /private/operator/path/relay-release
```

Defaults are AWS profile `dev.urandom.io`, region `us-west-2`, and functions
`mold-relay-router`/`mold-relay-frontend`. AWS IAM must permit reading/updating only
those functions. The helper never reads/provisions old SSM tokens or uploads a
browser shell/content to S3. Studio is deployed independently to trusted HTTPS.

Both current Lambda ZIPs are retained mode 600 and verified against AWS hashes
before any code update. Presigned download URLs are never logged/saved. The
private rollback manifest stores deployment identity and code hashes only, plus
whether each previous ZIP has the protocol-3 contract. Keep this directory private.

```sh
python3 scripts/relay/deploy.py --rollback /private/operator/path/relay-release
```

Rollback preflights both functions and refuses unrelated code changes, corrupt
ZIPs, or any pre-cutover/unmarked artifact. The initial hard cutover therefore
cannot roll back to the plaintext proxy; repair forward with a reviewed secure
build. Future protocol-3 rollbacks retain revision/hash safeguards.

## Register a host with operator IAM

Public discovery/enrollment returns metadata only. A self-signed key cannot take
a live host slot until an operator registers its public identity. After the
server creates its persistent host identity, copy the **public Ed25519 key** from
its local pairing metadata and provision:

```sh
python3 scripts/relay/deploy.py provision --host-key <64-lowercase-hex-public-key> --principal <stable-owner-id>
```

This stores only public authorization under the derived namespace in the
`mold-relay-sessions` table, using conditional creation/update. Repeating the same
registration is safe; a different principal/key is refused. The record has no
TTL. Cloud runtime IAM can read registrations but cannot write them; operator IAM
needs GetItem/PutItem on `v3-registration#*`. Do not supply a private identity,
invitation seed, API credential or legacy relay token.

Disable a matching registration with the same command plus `--disable`. Active
traffic/heartbeats recheck it and close disabled sessions. The account/owner ID
is explicit groundwork for future account enrollment; no public endpoint may
create these records. Host API authorization remains local device pairing.

The cutover checklist must verify old plaintext temporary S3 objects have expired
or were explicitly removed, old content permissions are revoked, and every
shipping client passes secure transport tests before claiming end-to-end coverage.

## Read-only production diagnostics

Run from a CLI or headless operator machine with AWS CLI credentials that can
read the two relay functions, the exact host item, CloudWatch metrics and the
router log group. The host ID is the 32-character namespace derived from the
host signing public key in pairing metadata; it is not a credential. This helper
reads only protocol 3 rows:

```sh
python3 scripts/relay/diagnose.py --host-id <32-lowercase-hex-host-id> --hours 24
```

The JSON report shows current host readiness, remaining lease, guest count,
function limits, Lambda throttles/duration, REST failures, WebSocket messages
and DynamoDB read/write usage. CloudWatch metrics explicitly cover the shared
relay fleet. Duration is in milliseconds; percentile summaries are the maximum
hourly percentile, not a combined percentile for the whole interval. Missing
metric samples stay absent rather than being reported as zero.

Log queries match only the exact host and guest connection IDs in that host's
current membership. Only allowlisted aggregate diagnostic categories are
returned. Earlier disconnected sessions and unscoped failures cannot be attributed by
this report; an empty diagnostic list is
not proof of an error-free history. No request payload, credential, connection
ID, session ID, owner verifier or function environment is printed. The helper
never scans the table, reads SSM, invokes a function or changes infrastructure.
CloudWatch Logs Insights queries incur the usual small scan charge, bounded by
the selected 1–168-hour window and one log group.

Run its offline safety contracts with:

```sh
python3 scripts/tests/relay-diagnostics-test.py
```

## Temporary AWS protocol 3 UAT

The staging helper has a fixed isolated stack name `mold-relay-v3-uat`; it creates
only its router Lambda (256 MiB, 10 seconds, concurrency 32), WebSocket API,
on-demand session table, scoped IAM, one-day logs and a private ciphertext-only
S3 bucket. It never modifies production or creates an EC2 host. Use synthetic
data and disposable identities. The bucket allows only the exact Studio/native
and documented local UAT origins, and expires abandoned ciphertext after one day.

```sh
python3 scripts/relay/staging.py create --package /private/build/router.zip --output /private/uat-manifest
python3 scripts/relay/staging.py status --output /private/uat-manifest
python3 scripts/relay/staging.py configure --output /private/uat-manifest
python3 scripts/relay/staging.py update --package /private/next-build/router.zip --output /private/uat-manifest
python3 scripts/relay/deploy.py provision --table mold-relay-v3-uat-sessions --host-key <test-public-key> --principal uat
python3 scripts/relay/staging.py delete --output /private/uat-manifest
```

The retained manifest records the exact stack ARN, every resource ID, verified
router code hash and public
WSS endpoint. Delete uses that identity and refuses a mismatched profile/region
or production stack name. Delete all temporary resources after final production
proof. The stack has no frontend; discovery/redirect contracts are tested separately.
