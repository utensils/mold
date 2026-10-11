- **Encrypted connections across Mold.** Replace the old proxy and pairing protocol
  with pinned, per-device end-to-end encryption across native apps, Studio, and
  headless CLI hosts. Nearby machines prefer a verified direct connection;
  remote requests and media use the same authenticated transport. Existing remote
  pairings must be replaced with a fresh host-generated invitation after upgrading.
- **Reliable network recovery.** Bound discovery and connection attempts, close
  stale or revoked sessions, preserve streaming and Range requests, and show
  readable offline and retry messages in the iPhone library.
- **Lower-cost bulk transport.** Move large encrypted transfers through private,
  session-scoped bucket objects with checksum verification and acknowledgement
  cleanup. Lambda authorizes and routes small messages; encryption keys stay on
  the host and client. Verified HTTPS pairing links cover Android and both iOS apps.
