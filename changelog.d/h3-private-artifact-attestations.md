- **MiniMax H3 no longer re-hashes unchanged checkpoints after every NixOS service restart.**
  The NixOS module now keeps digest attestations in owner-private systemd state,
  independent of the intentionally shared `MOLD_HOME`, and Mold warns when no
  secure persistent attestation store is available.
