# One-click native Mac phone pairing

## Acceptance

- Keep the existing native Settings TabView, sizing, tabs and grouped Form.
- Put **Pair your phone** in the first Remote Access section. Clicking opens
  a sheet immediately; preparation, failures and retries stay in that sheet.
- A saved machine uses its existing operator connection and pairing session.
- This Mac opts into a managed outbound connector on that click. The engine
  stays authenticated on numeric loopback; no inbound listener or port change.
- The code contains the engine's short-lived, one-use pairing token and all
  actual advertised usable LAN/Tailscale/HTTPS routes. Phone clients prefer
  direct routes and fall back to the scoped proxy after credential-free proof.
  Never advertise LAN interfaces the engine does not actually listen on.
- Native iOS and Tauri retain the same universal-link payload and claim flow.
- Retain stable host identity and relay origin across restarts. Stop remote
  access explicitly; closing the QR sheet does not unpair a phone.

## Confirmed gap

The shipped gateway has one global host slot and an operator enrollment token.
There is no public enrollment endpoint. A new user's QR cannot safely use that
singleton URL. This work therefore adds managed enrollment and isolated HTTPS
origins, while preserving the existing singleton as a legacy namespace.
Production DNS/TLS and environment configuration must be deployed before the
new native flow can enroll; a runtime PR is not deployment evidence.

## Managed gateway contract

- Enable managed enrollment only with a configured trusted base origin,
  wildcard host domain and verified WSS connector endpoint. Default off in
  an unconfigured gateway, with an actionable unavailable response.
- POST `/_mold/relay/enroll` returns `{host_id, token, public_url,
relay_url, expires_at}`. Generate a random 128-bit lowercase-hex host ID
  and 256-bit bearer owner token. Store only a SHA-256 verifier.
- Public origin is `https://<host_id>.<configured host domain>`; clients
  never invent it. Root-path origins preserve current route-proof, staged
  transfer and client URL contracts.
- Authenticated renew and delete operate on that host ID. Enrollment is
  bounded by 32 simultaneous slots and trusted-source rate limits. Initial
  reservations expire in two minutes; hello/heartbeats maintain 90-second
  live slots separately from established 30-day owner identities. New identity
  admission has a global burst of eight and replenishes once per 30 seconds.
  Never trust X-Forwarded-For for enrollment quotas. Delete releases capacity.
- WSS host headers carry `x-mold-relay-host` plus that host's owner bearer;
  frontend bridges carry the namespace plus the existing internal bearer.
  Persist namespace on each connection and dispatch subsequent frames through
  that namespace. Do not trust a frame to choose a different namespace.
- Namespace host membership, transfer grants and media jobs. Credentials,
  stream IDs, S3 object grants and worker invocations cannot cross namespaces.
  Keep current global infrastructure invocation/concurrency/storage bounds.
- Existing unscoped public origin and host enrollment token continue to work.
  Unknown, expired, malformed or revoked managed origins fail closed.

## Embedded connector and server advertisement

- Reuse mold-relay's AWS connector, including authenticated-target probe,
  request semantics, reconnect backoff and cancellation. Add an optional
  managed host namespace without changing legacy connector behavior.
- Embed a separately cancellable relay task behind a tiny C ABI: start with
  WSS endpoint, bearer, host ID and local port; stop; liveness. Credentials
  are in memory and the existing owner-only secret store, never arguments,
  URLs or logs. Reject a second concurrent connector.
- Expose an app-owned ConnectionAddresses handle for the embedded server.
  A validated managed HTTPS origin is added/withdrawn dynamically so pairing,
  claims and authenticated route catalogs agree without changing process env.
- Stop the connector on quit or engine death; restarting an opted-in app
  reuses its saved enrollment and reconnects after the engine answers.

## Native flow

- Add an observable managed-remote store with injected enrollment, connector
  and readiness operations. Compose and inject it in the existing AppStores
  and Settings environment. Persist owner material in SecretStore.
- Repeated button presses share one preparation operation. Cancellation and
  selection changes never show a code for another host.
- Prepare This Mac: require a running engine, enroll/renew, save credentials,
  start connector, publish the validated origin, mint a local pairing session,
  and verify its instance-bound pairing proof through that exact public route.
  Only then show the QR. Use bounded deadlines with a retryable error state.
- A saved machine immediately uses PairingSheet and its advertised endpoints.
  The sheet must clear stale sessions on a failed replacement and scope errors
  and sessions to its host. Existing expiry/reopen/New Code semantics remain.
- Remote Access shows enabled state and a native Stop Remote Access action;
  paired-device revocation continues through the existing pairing store.

## Validation and documentation

1. TDD: gateway enrollment/renew/delete, quota and two-namespace isolation;
   legacy tests; Rust connector namespace validation and advertisement tests;
   native lifecycle, deduplication, readiness, persistence and error tests.
2. Keep common universal-link fixtures unchanged. Add route selection tests
   proving LAN preference, Tailscale fallback and proxy fallback without
   sending a credential to a failed/foreign candidate.
3. Run gateway tests, affected Rust checks/tests, native lint and Xcode app
   tests/build. Run shared Swift package tests when changing shared contracts.
4. Use an isolated native UAT build/prefs domain to verify the existing Settings
   layout, button, loading/error/retry, QR and revocation controls. Demonstrate
   local gateway integration; distinguish it from undeployed production setup.
5. Update macOS README, root README, relay website/docs, owning rules and
   canonical agent-skill relay instructions, plus a changelog fragment.
6. Independent implementation review, fix findings, rerun affected checks,
   conventional commit, push, open/attach PR and verify exact-head CI.

## Independent plan review

The plan reviewer confirmed the UI reuse and required managed namespaces,
loopback-only forwarding, readiness proof before showing a QR, lifecycle and
isolation tests, explicit production prerequisites, and synchronized rules.
Path-prefix URLs were rejected because current Swift/TS route contracts require
root origins. This plan adopts dedicated managed origins instead.

## Implementation review and evidence

The independent reviewer checked the native lifecycle and the companion
Terraform source. Findings about connector liveness, durable opt-out and
cancelled enrollment cleanup were fixed and covered by regression tests.
The managed owner credential has a declared SecretStore slot; General reset
preserves the explicit remote-access preference.

Validation includes the gateway's 72 tests, connector's 27 tests, dynamic
advertisement's 6 tests, FFI's 15 tests, native Mac's 913 tests, shared client's
1,180 tests, and a generic iOS client-package build. A further regression test
reproduced and fixed pairing during unfinished Stop Remote Access revocation.
The native app builds with
the connector linked. Disposable native UI UAT confirmed the unchanged Settings
tabs and Form, the button for a running This Mac engine, and the QR sheet with
fixture LAN/Tailscale/relay endpoints. This does not demonstrate physical-phone
pairing or production enrollment.

The companion infrastructure source adds wildcard TLS/DNS and API mapping.
Read-only Terraform plans exposed existing AMI lookup failures and unrelated
replacement cascades, so no apply is safe on that evidence. Production rollout
must first resolve those plan blockers, deploy the gateway runtime and verify
two-host isolation and actual wildcard request authority.
