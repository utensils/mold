# Universal remote access

A saved machine is one identity and credential with several connection routes.
Existing pairings learn routes from the authenticated server while reachable;
leaving LAN does not require pairing again. A new QR carries an additive,
bounded list of LAN, Tailscale and public HTTPS relay origins. QR version 1 and
its original base_url remain compatible with older clients.

The server advertises only interfaces the listener actually serves, excluding
loopback, wildcard and unusable addresses. Its relay HTTPS origin is explicit
MOLD_PUBLIC_URL configuration, never inferred from request forwarding headers.
The connector remains outbound and does not change the engine listener.

Clients store routes alongside the existing host UUID and secret-store entry.
They prefer usable direct routes and retain healthy routes briefly to avoid
oscillation. Health/reconnect resolves a route before subsequent operations;
there is no automatic mutation replay. Existing durable IDs and read-stream
reconciliation recover admitted work. An authenticated refusal remains visible.
HTTPS browsers skip HTTP routes they cannot access due to mixed-content rules.

Before forwarding a key or one-use token to an alternative origin, a separate
credential-free bounded probe verifies the expected identity and a fresh
HMAC-SHA256 response. Its request contains only a 64-bit hash tag and random
nonce, never the credential or full derived hash. A domain-separated proof uses
the full SHA256 credential digest. Pairing proofs neither consume tokens nor
extend expiry; revoked grants cannot produce proofs. This prevents accidental
credential delivery to a reused address. Plain HTTP retains its existing trusted
network assumption: the proof is not TLS channel binding and does not defeat an
active forwarding MITM. Public relay URLs require HTTPS with verified TLS.

Native macOS Settings has a discoverable Remote Access tab beside Machines,
showing route/address context, an inline QR and existing device revocation. The
private built-in This Mac engine remains private. An absent relay configuration
is shown honestly; clients cannot infer that a cloud gateway hosts this machine.

Validation: server proof/admission/filter tests; shared Swift and TypeScript
wire/proof/selection tests; backwards-compatible QR fixtures; saved-host migration
and route refresh tests; wrong-identity and revocation cases; credential-free
probes; no mutation replay; native macOS rendered UAT and cross-surface CI.
HAL9000 deployment requires operator host selection; cloud gateway already exists.
