- **Fetch Nix release dependencies reliably.** Download checksum-verified Cargo
  archives directly from the static crates.io endpoint for the Crane helper and
  desktop package. Report failed cache builds as failures while keeping cache
  publication independent of native release delivery.
