### Fixed

- Automatic cloud provisioning from nightly and source builds now requires a published container digest for the exact source before creating a paid host. Verified stable and scheduled builds publish source readiness and refresh current-main `latest` aliases without downgrading them on older reruns.
- Docker dependency caching now includes the shared secure transport crates.
