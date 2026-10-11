# Protocol 3 ciphertext object transport, version 1

This is an optional bulk carrier for the **existing** Noise ciphertext stream.
It introduces no application protocol, file key, plaintext staging, HTTP proxy,
or replay of an HTTP operation. Small records and LAN connections stay inline.
Both an admitted host and an admitted device may allocate objects. Invitation
connections may not. All authorization is through the existing admitted WSS
connection; the broker never receives an API credential.

## Trusted storage policy

Clients ship or receive through independently trusted operator configuration
the exact permitted object origin. Do not trust an origin learned only from
cloud `ready`, discovery, an object reference, or a redirect. Production origin:

`https://mold-relay-042506291754-us-west-2.s3.us-west-2.amazonaws.com`

The isolated test fixture explicitly opts into:

`https://mold-relay-v3-uat-042506291754-us-west-2.s3.us-west-2.amazonaws.com`

An optional `objects:1` field in `hello` opts into this carrier. The host's
`ready` advertises availability only when it opted in. A device's `ready` and
the corresponding host `open` advertise it only when both peers opted in;
invites never receive object capability. Host senders must gate each rid on
its `open` policy. Peers without mutual support stay inline.

The negotiated optional `objects` field advertises availability:
`{version:1,origin,maxBytes:262144,maxCount:16,maxInFlightBytes:1048576,maxInFlightObjects:4}`.
It must match the independently configured origin and these version-1 limits.
Absent, unknown or mismatched policy disables object allocation. It never adds
a trusted origin. Direct LAN peers do not advertise this field.

Every object URL must use HTTPS, that exact origin, no userinfo, no fragment,
and the exact path below. Fetch must reject redirects. Never attach API keys,
cookies, ambient credentials or authorization headers to object requests.
Presigned query parameters are opaque and must not be logged or persisted.

`/ciphertext/v3/<namespace>/<sid>/<rid>/<direction>/<id>`

Namespace and id are lowercase hex32. Encode sid and rid as individual UTF-8 URL
path segments using RFC3986 percent encoding (unreserved ASCII only remains
literal). Direction is `host` for host-to-device and `device` for device-to-host.
Each receiver derives every segment from its pinned live session and envelope;
the URL itself supplies no authority. Namespace/session/direction binding also
prevents confused-deputy fetches across another device's objects.

## Allocate and upload

Integers are nonnegative JavaScript safe integers; times are Unix seconds.
`requestId` is exactly 32 lowercase hex characters, generated randomly per
allocation and reused only to retry the identical request. `id` is a separate
32-character random identifier chosen by the broker. SHA-256 is lowercase hex64.

Send on the admitted WSS connection:

```json
{
  "a": "object",
  "v": 3,
  "op": "create",
  "requestId": "<hex32>",
  "sid": "<session>",
  "rid": "<device connection>",
  "seq": 42,
  "count": 16,
  "bytes": 261648,
  "sha256": "<hex64>"
}
```

The router checks fresh registration, current host/session, device membership,
role and expiry exactly as for data frames. It also requires that the host has
sent `accept` for this exact peer/session; possession of an unexpired cloud
certificate alone cannot allocate paid storage. Host cancellation closes and
removes the peer's cloud membership. The allocation is bound to the
sender connection, namespace, sid, rid, direction, starting seq, count, byte
length and checksum. A requestId retry must match all fields. It returns:

```json
{
  "a": "object",
  "v": 3,
  "op": "created",
  "requestId": "<hex32>",
  "sid": "<session>",
  "rid": "<device connection>",
  "id": "<hex32>",
  "seq": 42,
  "count": 16,
  "bytes": 261648,
  "sha256": "<hex64>",
  "putUrl": "<presigned HTTPS>",
  "getUrl": "<presigned HTTPS>",
  "putHeaders": {
    "content-type": "application/octet-stream",
    "x-amz-checksum-sha256": "<base64 of the 32-byte digest>",
    "if-none-match": "*"
  },
  "expiresAt": 1791747600
}
```

Clients validate correlation, all echoed fields, both exact URL paths/origins,
headers and `now < expiresAt <= now + 300` (allow up to 30 seconds clock skew
only for the upper bound). PUT the exact retained ciphertext object bytes with
exact returned headers. The signer binds Content-Length and checksum; browsers
set Content-Length from the binary body. Never use streaming/chunked PUT,
multipart upload, or modify/re-encrypt bytes on retry. The PUT is conditional on
object absence (`If-None-Match: *`, included in `putHeaders`); a retry receiving
412 may proceed to send the reference because the broker verifies stored size
and checksum. All other non-2xx upload statuses fail this attempt.

Presigns expire at the allocation's fixed expiry (at most 300 seconds). There
is no renewal beyond that expiry: cancel the encrypted channel if bytes have
not been acknowledged by then. The application may reconcile safe reads on a
new session; uncertain mutations are never automatically replayed.

## Object bytes and forwarding

An object is a concatenation of **original consecutive reliable data frames**:
`u32BE(chunk length) || raw ciphertext chunk`, repeated `count` times. Each raw
chunk is 1..16384 bytes; count is 1..16, and total object bytes including all
four-byte lengths are 5..262144. A full 16 x 16384-byte bundle exceeds that total
bound and must be split. No trailing bytes, zero-length chunks or padding exist.
The reliable window remains 16 original sequences, not 16 objects.

After successful PUT, send:

```json
{
  "a": "data",
  "v": 3,
  "sid": "<session>",
  "rid": "<device connection>",
  "seq": 42,
  "o": {
    "id": "<hex32>",
    "bytes": 261648,
    "sha256": "<hex64>",
    "url": "<the granted getUrl>",
    "count": 16
  }
}
```

Data has exactly one of `d` or `o`. The broker verifies the matching live
reservation, including seq/count and sender; it checks S3's stored exact length
and SHA-256 before first forwarding. It forwards only its canonical granted URL,
never an arbitrary sender URL. Object bytes never pass through Lambda.

The receiver checks the session, direction, sequence-window bounds and memory
budget before fetching. Fetch at most four objects / 1 MiB total concurrently,
with a bounded body read and no redirects. Verify the complete object length,
SHA-256 and **all** frame lengths/count before exposing any chunk to Noise.
Expand into original seq through seq+count-1, then reorder/deduplicate using the
same reliable window as inline data. Only ACK bytes actually consumed by the
existing parser. A ref and an inline retry at the same sequence must contain
identical ciphertext. Never re-encrypt or decrypt an unverified partial object.

Sender retains original ciphertext until acknowledged and retransmits the same
ref/bytes. A fully old ref is acknowledged without fetching again: it is already
outside the receive window and may have been deleted. Partial old refs still
require a complete bounded fetch/verification before accepting their new tail.

## Release, limits and errors

The broker tracks at most four live allocations / 1 MiB for each
namespace+sid+rid+direction. Limits apply across retries and parallel Lambdas.
Each admitted peer has at most 16 unacknowledged original sequence slots. Hosts
are capped at 32 namespaces and 32 peers each; invite peers cannot allocate.
Expired reservations do not create authorization. Cleanup deletes only objects
under `ciphertext/v3/`; old plaintext prefixes have no runtime IAM access.

Cumulative ACK `next >= seq+count` releases that direction's matching forwarded
objects. Duplicate/stale ACKs cannot delete a newer session or another peer's
objects. Explicit cancellation releases both directions. The sender can abandon
an allocation before forwarding with:

```json
{
  "a": "object",
  "v": 3,
  "op": "release",
  "requestId": "<new hex32>",
  "sid": "<session>",
  "rid": "<device connection>",
  "id": "<hex32>"
}
```

Reply is `{a:"object",v:3,op:"released",requestId,sid,rid,id}`. Release is
idempotent, sender/session fenced, and may not delete a forwarded unacknowledged
object. Cancel the whole channel instead when abandoning forwarded bytes.
Cleanup errors retain quota until retried; a one-day S3 lifecycle is the final
cleanup backstop, not the normal path. Never delete real host media.

Recoverable broker replies are
`{a:"object",v:3,op:"error",requestId,sid,rid,code}` with code exactly one of
`disabled`, `capacity`, `not_ready`, `expired`, `conflict`, `unavailable`.
`capacity`/`not_ready`/`unavailable` may retry the identical request after a
bounded 100–1000 ms delay, within its original expiry. `disabled`, `capacity`, `unavailable` or allocation/PUT network failures may
fall back to the exact retained original inline ciphertext frames after at most
five seconds of retries, only before any object ref is published. Best-effort
release the unforwarded allocation first; do not re-encrypt or replay an API call. `expired` and `conflict` terminate the
channel. Malformed, unauthorized or cross-session frames close the socket under
the normal v3 fail-closed admission rules. No error contains an S3 URL or body.

The private bucket blocks all public access, denies insecure transport, uses
server-side encryption in addition to Noise, and allows only exact production
Studio origins for browser GET/HEAD/PUT CORS with checksum/content headers.
Lambda IAM is limited to `ciphertext/v3/*` object read/write/delete and scoped
admission/quota rows. No S3 list, legacy content, bearer secret or API invoke
authority is required for the runtime.
