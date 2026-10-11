- **Verify cloud image readiness before spending.** Automatic cloud provisioning from nightly and source builds now requires a published container digest for the exact source before creating a paid host. Verified stable and scheduled builds publish source readiness and refresh current-main `latest` aliases without downgrading them on older reruns.
- **Build secure containers.** Docker dependency caching now includes the shared secure transport crates.

- **Connect to encrypted LAN from hosted Studio.** HTTPS Studio now tries authenticated encrypted LAN connections before the cloud relay and falls back when browser local-network policy blocks them. Initial pairing allows time to answer the browser permission prompt while ordinary route probes remain fast.

- **Keep cancellation explicit.** Report interrupted encrypted responses as errors when forgetting a host or cancelling a request, preventing a cleanup race from appearing as a successfully completed body.

- **Reliable native Mac packaging.** Builds isolate Swift package checkouts from stale runner caches so signed distributions can resolve Sparkle and the encrypted transport framework reliably.

- **Manage hosted Studio pairings.** Switch, disconnect, or forget the selected machine without losing recoverability when storage fails. Changes in another browser tab cannot silently retarget the mounted workspace. Generation jobs retain their owning machine across selection changes and reconnects, and ambiguous recovered jobs cannot act on another machine.
