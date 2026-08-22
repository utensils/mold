- **Fixed the private H3 claimed-attempt test doubles and the release-contract
  phase gate.** Under `--features h3-private-uat` ten `claimed_h3_*` /
  `claimed_ref2va_*` GPU-worker tests failed against two correct production
  fences: the doubles echoed constant identity digests instead of the run
  binding the owner scope derives, and their scheduler stand-in never answered
  the ledger-aware host-memory recheck the owner blocks on before the CUDA
  allocation boundary. The fixtures now derive the same work-identity,
  cancellation-scope, and ledger-sequence values the real runtime receives and
  answer that recheck while forwarding every other worker event untouched
  ([#1204](https://github.com/utensils/mold/issues/1204)).
- **The private-UAT release contract now enforces its phase ordering.** Its
  VAE-drop → terminal-identity → mux markers also matched the Ref2VA twins, so
  the resolved line numbers were multi-line values that made bash arithmetic
  error out and pass the whole ordering check silently. Every marker is now
  anchored to the block that identifies the FL2VA occurrence, the Ref2VA attempt
  gets its own terminal-before-mux check, and any marker resolving to zero or
  several lines fails the script by name
  ([#1207](https://github.com/utensils/mold/issues/1207)).
