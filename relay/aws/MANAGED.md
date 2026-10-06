# Managed phone pairing gateway

Managed enrollment adds isolated root HTTPS origins to the existing singleton
relay. The singleton keeps its existing operator host token and unscoped
connection, grant and media records. A managed host uses a random 128-bit host
identity and a separate 256-bit owner token; DynamoDB stores only its SHA-256
verifier. TLS terminates at AWS. The connector still forwards only to the
loopback engine after its existing anonymous-authentication probe.

## Required deployment

This change does **not** deploy DNS, certificates or infrastructure. Before
native one-click pairing is usable, configure the URandom Terraform deployment:

- Wildcard DNS and a TLS certificate for `*.mold-link.urandom.io`, in addition
  to the existing central `mold-link.urandom.io` origin. Route every managed
  hostname to the same Regional REST streaming frontend and preserve the
  original hostname in API Gateway's trusted `requestContext.domainName`.
  Verify that value in an integration test; a mapping that replaces it with
  the underlying AWS hostname will fail closed. Do not copy caller `Host`,
  `Forwarded` or `X-Forwarded-For` into trusted request context.
- Set `MANAGED_HOST_DOMAIN=mold-link.urandom.io` and
  `PUBLIC_ORIGIN=https://mold-link.urandom.io` on both frontend and router
  Lambdas. Set the existing `WS_ENDPOINT` to the deployed verified `wss://`
  endpoint on both Lambdas. All three variables are required; incomplete
  configuration disables enrollment. HTTPS origins must have no path,
  credentials, port, query or fragment. The WSS endpoint may have its stage
  path, but no credentials, port, query or fragment.
- Keep existing DynamoDB table access, TTL (`expiresAt`), bridge-token SSM
  permissions, private S3 bucket, staged-prefix lifecycle cleanup, streaming
  Lambda timeouts and global reserved concurrency/invocation limits. Managed
  roster and tenant records use the same table; no plaintext owner secret or
  new IAM public access is needed. Upload admission remains a global 64-grant
  bound across managed hosts and the singleton.
- Bundle/deploy the existing router and frontend entrypoints using
  `scripts/relay/deploy.py`; the bundler includes `managed.mjs` automatically.
  Tokens never belong in Terraform state, deployment logs, URLs or arguments.

A gateway accepts at most 32 managed registrations. Leases last 30 days and are
renewed by the Mac while enabled. Enrollment is limited to five per trusted
source IP per hour. Source-IP verifiers are hashed and held in the same atomic
CAS roster, with at most 1,024 entries and one-hour expiry. Every successful
enroll, renew and delete reclaims expired host and quota entries. DynamoDB TTL
is cleanup only; every authorization checks lease time synchronously. The
singleton retains its own 32 live frontend slots; each managed host retains the
same bounded stream-membership leases and existing global deployment limits.

## Client contract

Central-origin `POST /_mold/relay/enroll` has no body or Mold API key requirement.
It returns HTTP 201:

```json
{
  "host_id": "32 lowercase hexadecimal characters",
  "token": "43 base64url characters representing 32 random bytes",
  "public_url": "https://<host_id>.mold-link.urandom.io",
  "relay_url": "wss://<deployed-websocket-endpoint>/<stage>",
  "expires_at": 1790000000
}
```

Persist the owner token in the platform secret store and retain the host ID and
origin across restarts. Authenticated `POST /_mold/relay/enroll/<host_id>` renews
and returns HTTP 200 with the same schema and owner token, a new expiry and the
configured endpoints. The submitted token is returned unchanged after verifier
validation, and is never persisted as plaintext. `DELETE` on that route with
`Authorization: Bearer <owner-token>` returns HTTP 204 and revokes the lease.
An expired owner lease must enroll again. The enrollment endpoints work only
on the configured central origin, never a tenant origin.

The host WSS handshake sends:

```text
x-mold-relay-role: host
x-mold-relay-host: <host_id>
Authorization: Bearer <owner-token>
```

Frontend bridges use `x-mold-relay-role: frontend`, the same namespace header
and the existing internal bridge bearer. Omit the namespace header for the
legacy singleton. Namespace is persisted on the authenticated connection;
subsequent messages and disconnects use that stored namespace. A frame cannot
select another tenant. Unknown, malformed, revoked or expired tenant origins
and host handshakes fail closed. Viewer headers never select a namespace.

Upload grants, media polling and staging jobs are tenant-bound in DynamoDB.
Managed S3 uploads are `uploads/<host_id>/<uuid>`; responses are
`_mold/objects/<host_id>/<uuid>`. Legacy prefixes remain unchanged. Signed object
URLs remain private bearer capabilities limited to the configured bucket;
frontend signing verifies the requested tenant prefix. Neither an identical
engine API key nor an identical epoch permits transfer-grant reuse across hosts.
Workers verify their saved job's tenant, credential and epoch before forwarding.
Application mutations are never replayed after uncertain delivery.

## Verification

Run `node --test relay/aws/test/*.test.mjs` with the package dependencies
installed. The suite covers concurrent atomic enrollment capacity, source
quotas, lease reclamation, owner renew/delete, trusted hostname selection,
managed/legacy coexistence, cross-tenant credentials and frames, revoked
sessions, upload grants, media polling and worker dispatch. These are local
contract tests; production wildcard DNS, certificate routing, trusted context
and an actual outbound connector must be verified after infrastructure deploy.
