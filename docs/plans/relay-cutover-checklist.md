# Protocol 3 production cutover and recovery

This is the operator checklist for the authorized hard cutover. Check a box only
after retaining the corresponding evidence. No production action is implied by
a staging result. Account profile is `dev.urandom.io`, region `us-west-2`.

## Reviewed artifacts and current evidence

- [x] Mold PR: <https://github.com/utensils/mold/pull/1867>. Cloud checkpoint
      `1122a2c6b` adds the broker and host-accept gate; later commits must preserve its tests.
- [x] Infrastructure PR: <https://github.com/urandomio/infra/pull/2>, reviewed
      checkpoint `e8afa524dab8ff4bf4fb4a51352745ab4de7a501`.
- [x] Full Terraform plan is confined to `module.mold_relay`: 20 updates and
      one deletion, no creation/replacement or unrelated mutation. The deletion is
      the retired frontend asynchronous invocation configuration. Regenerate at final head.
- [x] Cost Explorer allocation keys `BillingGroup`, `Project`, `Service`,
      `Environment`, `ManagedBy` are Active. Module tests check all supported tags,
      including explicit CloudFormation child tags. Shared Route53 zone costs remain
      shared; individual records do not support tags.
- [x] Staging stack `mold-relay-v3-uat` uses WSS
      `wss://4dp9o267oi.execute-api.us-west-2.amazonaws.com/uat` and isolated bucket
      `mold-relay-v3-uat-042506291754-us-west-2`. No EC2 or production DNS change.
- [x] Reviewed stage broker hash after host-accept, tombstone, atomic allocation and bounded quota compression:
      `7NuM/cdc7x3Ntp3hbROcXHuhAmjlOfqHqE7ByGDpsgc=`. Record superseding hashes here
      before using newer evidence. Production hashes are **not yet assigned**.
- [x] Staged carrier checks: invalid checksum rejected; signed exact PUT/GET
      succeeds; conditional duplicate cannot overwrite; path segments encode exactly
      once; complete ACK clears only its payload into a zero-byte tombstone; replayed
      conditional PUT receives 412; lost ACK recovers without fetching the tombstone.
- [x] Read-only legacy inventory found zero objects in production `uploads/`
      and `_mold/objects/`; recheck after old writes stop. DynamoDB had 11 known
      legacy metadata rows, and exactly two owned old SSM parameters. No secret
      values or host media were read or printed.

- [x] App associations PR <https://github.com/utensils/utensils.github.io/pull/48>
      merged as `b63df2f8bb033e8124a666d86cffc4684e6b446a`. Build/deploy run
      `38172992429` passed; both live HTTPS `.well-known` records contain the
      reviewed iOS bundle IDs and verified Android release certificate.
- [x] Isolated actual Tauri desktop paired to the private-interface fixture
      `192.168.1.142:53372`, showed "on your network", and reconnected with the
      saved pairing after quit/relaunch. This verifies desktop LAN behavior;
      it is not evidence that the installed production host has been upgraded.

- [x] The uninterrupted 200 MiB debug-client transfer completed with exact bytes
      and bounded memory. [Cost evidence](relay-cost-evidence.md) records its
      UTC interval, code hash, measured cloud counts and official regional rates.
      Optimized-build and compact-storage measurements are still separate checks.

## Required release gate

- [ ] Exact final Mold/infra heads, independent review and terminal CI checks
      are green. Windows private-file ACL tests must execute on Windows, not merely
      cross-compile. Record build/install versions for every native client.
- [ ] Actual Rust/CLI, browser/desktop/mobile and native Apple clients pass
      pairing, grant renewal/revocation, simultaneous distinct devices, host restart,
      LAN preference, unavailable cloud, SSE, Range/HEAD, cancellation, uploads and
      media playback. Distinguish simulator proof from physical-device proof.
- [ ] Bidirectional ciphertext objects pass large uploads/downloads, memory
      bounds, checksum/URL rejection, retries and object cleanup. Measure a dedicated
      uninterrupted large transfer with UTC interval, exact bytes, throughput,
      memory and scoped cloud request/cost evidence. Overlapping probes are functional
      evidence only, not an isolated performance baseline.
- [ ] All public executable browser code is served from independently trusted
      HTTPS Studio or the local embedded application. Non-loopback HTTP API is426;
      non-loopback browser HTML redirects to fixed HTTPS Studio without forwarding
      incoming path/query. The secure WebSocket route remains available separately.
- [ ] New client builds are available/installed before removing their old cloud
      path. Include the signed/native apps, headless CLI archives and exact-digest
      Docker images used by RunPod provisioning; an older image cannot consume
      the new one-time secure bootstrap. No compatibility switch or plaintext fallback remains in the release.
- [ ] Halcyon's persistent server-owned identity is read through the secure
      helper, its public fingerprint is independently checked, and the actual public
      key is operator-registered with its intended principal in production. Never use
      the deterministic staging fixture identity, create a second connector identity,
      overwrite an existing pin or print a private key/invitation/API credential.
- [ ] Build final deterministic router/frontend ZIPs and record both SHA256
      values plus source commit. Retain private current-code ZIPs and config hashes;
      the deployment helper must reject stale revisions and unrelated changes.
- [ ] Regenerate/review the complete production Terraform plan from the exact
      infra head immediately before apply. Verify it remains scoped, preserves the
      immutable bootstrap Code, DNS/TLS and real bucket contents.

## Coordinated production execution

1. Announce the short hard-cutover window to the coordinating task. Stop old
   managed connectors from renewing/re-enrolling. Record public host pins and
   currently installed versions; preserve all local operator credentials/media.
2. Provision only the verified production **public** host registration using
   `scripts/relay/deploy.py provision`. Registration is IAM-owned and cannot be
   created by anonymous discovery or the runtime.
3. Deploy final protocol3 router/frontend artifacts with revision/hash guards.
   The code tolerates absent object configuration by using encrypted inline
   frames until the reviewed infrastructure configuration is applied. Old HTTP
   application/media APIs and old WebSocket admission must immediately fail426.
4. Apply only the reviewed complete Mold-scoped Terraform plan. Confirm both
   Lambda CodeSha256 values stayed at the deployed secure artifacts; configuration
   changes must not reinstall bootstrap code. Verify IAM has no old SSM/plaintext
   prefix/invoke authority, frontend is metadata-only, and S3 grants are limited
   to `ciphertext/v3/*` with exact origin CORS and private/TLS-only bucket policy.
5. Start the new Halcyon connector with its existing persistent identity; connect
   new clients over LAN and production cloud. Validate authenticated status,
   distinct devices, pairing/renewal/revocation, uploads, media Range/HEAD, SSE,
   reconnect/cancellation and a measured bulk transfer. Verify that an unaccepted
   or revoked peer cannot allocate an object.
6. After confirming old writes stopped, inventory only the two legacy temporary
   S3 prefixes again. Remove any remaining **relay temporary objects** in those
   exact prefixes, then verify zero. Do not delete other prefixes or host files.
7. Remove only `/mold/relay/host-token` and `/mold/relay/bridge-token` after proving
   both functions no longer reference them. Read parameter metadata only; never
   fetch values. Remove dedicated relay-table records only in the known legacy
   families `managed-roster`, `managed-owner#<hex32>`, `host`,
   `host#tenant#<hex32>`, `connection#<gateway-id>` and
   `upload#<temporary-id>[#tenant#<hex32>]`. Project keys only and inspect any
   unexpected family before action. Never delete `v3-*` or unrelated records.
8. Verify no legacy metadata/parameters/plaintext temporary objects remain, all
   supported production resources carry the expected billing/ownership tags,
   and cost allocation keys stay Active. Capture final deployed hashes, public
   readiness and aggregate failure/cost evidence without private contents.
9. Merge authorized PRs only at their tested heads and synchronize checkouts.
   Delete the isolated staging stack using its retained identity manifest after
   final production proof. The staging helper cleans only its own synthetic
   `ciphertext/v3/` prefix; verify all recorded staging resources are gone.

## Recovery without plaintext rollback

If any update is uncertain, inspect the exact function hash/configuration and
stack status before retrying. A successful PR merge does not prove deployment.
The deploy helper refuses pre-cutover/unmarked ZIP rollback; never restore v2
or old SSM bearers. Keep secure LAN/local access available while repairing
forward. A later rollback may use only a verified protocol3 artifact with
matching revision/hash ownership and a compatible object/admission contract.

On a failed encrypted stream, close that session, clean only its bounded
ciphertext objects, and reconnect with fresh Noise state. Reconcile safe reads
and SSE state; do not replay an uncertain mutation or generate fresh ciphertext
under a reused nonce. If production object transport is temporarily unavailable,
pre-publication fallback may send the retained original ciphertext inline;
after a reference is published, retry only the exact reference or close the peer.
