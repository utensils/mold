### Fixed

- Automatic cloud provisioning from nightly and source builds now requires a published container digest for the exact source before creating a paid host. Verified stable and scheduled builds publish source readiness and refresh current-main `latest` aliases without downgrading them on older reruns.
- Docker dependency caching now includes the shared secure transport crates.

- HTTPS Studio now tries authenticated encrypted LAN connections before the cloud relay and falls back when browser local-network policy blocks them. Initial pairing allows time to answer the browser permission prompt while ordinary route probes remain fast.

- Report interrupted encrypted responses as errors when forgetting a host or cancelling a request, preventing a cleanup race from appearing as a successfully completed body.
