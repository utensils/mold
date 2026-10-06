---
paths:
  - "crates/mold-server/src/gallery_*.rs"
  - "crates/mold-db/src/gallery_*.rs"
  - "crates/mold-db/src/generation_queue_media.rs"
  - "crates/mold-cli/src/commands/system.rs"
  - "crates/mold-server/src/lib.rs"
  - "studio/api/gallerySourceMedia.ts"
  - "desktop/src/lib/gallery/sourceMedia.ts"
  - "desktop/src/components/gallery/Lightbox.vue"
---

# Gallery archive authority and durable source media

Moved from the root CLAUDE.md; loaded only when working under the paths above.

## Archive authority storage

Storage version 3 (the append-only delta log) is a property of the
`$MOLD_HOME`, not of the process, so WRITING it is opt-in: `gallery.authority_log`
(`MOLD_GALLERY_AUTHORITY_LOG`), resolved once in `run_server` before anything
opens a gallery. READING v3 is unconditional. The switch exists because a mold
older than 0.29 reads v2 only and refuses to publish against a v3 store — one
new process starting used to upgrade the store in place and lock every older
binary out of the home, with the backup rewritten at v3 too so there was
nothing to roll back to. The upgrade now writes a separate
`gallery-authority-v3` directory and leaves the v2 store frozen intact;
`authority_dir` resolves a store by the presence of its MARKER rather than by
name, which is also what lets it still find and repair a store an earlier build
upgraded in place. Two writers of different versions on one home keep SEPARATE
indexes that drift — that is the stated cost of the switch, not a defect, and
the docs say to enable it only where every binary is new enough. Fresh
initialization takes the same directory rule as the upgrade
(`write_fresh_store_v3`, marker last): a home that opts in before it has any
store must not get v3 bytes under the `-v2` name. `mold system
gallery-authority status|downgrade` is the operator door — read-only status,
and an idempotent downgrade that replays the log, rewrites at v2, parks the v3
directory, verifies by reading back, and refuses on a pending mutation or a
torn tail.

**A LIVE WRITER refuses it too, and a commit can never land v3 bytes under the
v2 name.** Those three guards are crash-recovery conditions, and the
bookkeeping flock the downgrade takes is held by a server only for the length
of one commit — so taking it BLOCKING meant waiting for the gap between two
prints and rewriting the store under a running `mold serve`, whose next
publication then put a delta and a `{"version":3}` marker into the
just-rewritten `gallery-authority-v2` and locked every older binary out of the
home (UAT final-2, D9). Every process that opens the authority for WRITING
(`load_or_initialize_with_authority_log` AND `commit_snapshot` — the
publication gate's cache can be installed by `load_existing_read_only`, so a
commit reaches the store with no recovery of its own) holds a SHARED flock on
`<output_dir>/.mold-gallery-writer.lease` for the life of the
process; two servers still share a home, a crash releases it, and a writer that
cannot take one warns and publishes anyway. **The lease is in the GALLERY ROOT,
never in `.mold-batch-transactions`, and it is REMOVED on a clean stop.** Every
mold validates the transaction root as an inventory of DIRECTORIES — a regular
file it does not recognise is "unrecognized non-directory gallery transaction
entry" during startup recovery — so 42480db8's lease inside it stopped a
pre-0.29 binary from STARTING, which is the rollback the interlock exists to
protect, and it survived SIGTERM and outlived `downgrade` (UAT final-2, E2). The
gallery root is enumerated only for `.mold-batch-attempt-<64 hex>.lock` and for
media extensions, so a dotfile there is invisible to every build.
`release_gallery_writer_leases` (the server's shutdown sequence after the drain,
its hard-exit path, and the CLI's own exit) releases and UNLINKS — upgrading the
shared lock to exclusive first, so a second server sharing the home keeps its
file — and `acquire_writer_lease` re-checks the inode after locking so an
acquirer racing that unlink cannot end up holding a lock on a detached one. A
lease file NOBODY holds is stale, which is a leftover and not a refusal:
`downgrade` takes the lock straight through it and removes the file as its LAST
step, so the home it hands to an older binary is clean, and it removes
42480db8's transaction-root lease too (as does startup recovery). `downgrade`
takes the lease EXCLUSIVE with
`try_lock` AFTER the bookkeeping flock — the order every writer takes them, so
nothing waits on a lock another holder is queueing for — and refuses naming
`mold serve` and the recorded pid (the pid is only quoted when it is ALIVE; a
dead one means the body is another writer's leftover stamp). `status` reports
live / stale / none and is READ-ONLY — it describes the stale file rather than
clearing it. The second
half stands without the lease: `cached_commit_tail` also requires the marker's
VERSION to be the one this process writes (the generation can agree across a
store swap), `recover_storage` routes on whether the RESOLVED store is already
v3 rather than on its checkpoint's version and re-runs the upgrade beside the
frozen v2 store, its crash-recovery marker writes stamp the version the store
IS, and `ensure_v3_store_is_addressable` fails the commit outright rather than
appending v3 bytes to a v2 store.

## Commit fast path and the v3 delta log

**`gallery_authority::commit_snapshot`'s fast path is valid only under the bookkeeping flock, at the marker's own generation.** A commit used to re-read, re-verify and re-parse the whole checkpoint (a serde pass plus a SHA-256 over an index that grows with the gallery) and then read it a second time inside `write_checkpoint`. It now trusts a process-local `AuthorityTail` — the generation and the legacy-evidence epochs — and the three on-disk facts that say the tail still describes the checkpoint: a marker with no pending mutation, whose `committed_generation` is BOTH the caller's expectation and the one this process last wrote, and no unresolved WAL. Any of those failing falls back to the full `recover_storage` read, which is also what a cold process, a foreign writer, or a crash-interrupted mutation gets. The snapshot is serialized ONCE and the same bytes become the marker digest, the WAL, the checkpoint and the backup; `serialize_envelope` builds the envelope by hand so the digest covers the bytes that land, pinned against the serde shape by `the_prebuilt_envelope_matches_the_serde_one`. The same-generation BACKUP is still written on every commit (it is the copy `read_checkpoint` falls back to when `current` is unreadable); only the forensic `previous` is periodic, because `recover_storage` refuses any checkpoint whose generation disagrees with the marker and so can never land on it. **STORAGE_VERSION 3 replaces that whole-snapshot write with an append-only `mutation.log` of DELTAS** — three fsyncs a commit, and bytes proportional to what changed rather than to the size of the library (measured 60.5 ms/commit against 4.4 at 10,000 prints). A record is framed, digested over the bytes that land, and taken on replay only while it is contiguous and intact; the first that is not ends the replay and is truncated, because a log is a sequence and a gap makes its tail meaningless. The checkpoint and its backup are refreshed at COMPACTION (256 records, 8 MiB, or startup), which is also the one-time v2 read-and-upgrade. The delta rests on one contract: **a mutation that edits an entry in place must name it in `exact_names`** — adds and removes come from the key sets, but comparing values would mean serializing every entry, which is the cost the log exists to remove. A process writes v3 only after its own startup recovery has succeeded for that root (`v3_writing_enabled`), because a commit that has not resolved the log's tail could bury a tear under valid-looking bytes; until then it writes v2, which every build still reads.

## Durable gallery source media

Durable queue uploads do not die with their queue row. Publication first pins
the encrypted media set under queue-media storage, commits that exact pin into
the gallery archive authority, projects it into `gallery_media_*`, and only
then settles the queue row. Restart replay performs the same handoff before it
recognizes an already-published job as complete. Trash preserves pins;
permanent deletion removes gallery authority first and releases only that
print's pins afterward, so sibling outputs remain independent.

`GET /api/gallery/source-media/:filename` and its opaque-member download route
require the CALLER to be authorized, which is the same question every other
privileged route asks: on a host with `MOLD_API_KEY` set that is an
authenticated request, and on a keyless host it is every request, exactly as
`DELETE /api/gallery/image/:filename` and device lifecycle already behave. The
gate must never read the server's own configuration as the answer —
`AuthState = None` means "open by policy", and treating it as a refusal made the
whole feature dead on a default server, answering `unavailable_auth` for prints
it never looked at. They never expose paths or store identities and
report explicit `unavailable_auth`, `unavailable_legacy`, and
`unavailable_missing_or_corrupt` states; an empty member list after a CLEAN
resolve is `unavailable_legacy`, never corruption, because `downloadable_role`
filters provenance-text roles by design. **Every client always asks, and the
server is the only authority on what it retained** — `OutputMetadata` records
no marker at all for inline `source_video`, `audio_file`, or `mask_image`
bytes, so a client that skipped the probe on missing markers would silently
lose those restorations. What the metadata decides is DISCLOSURE: the server
cannot tell a pre-feature print from one that never had source media (both
resolve with no pins), so an UNAVAILABLE answer is toasted only when
`retainedSourceMediaDisclosable` in `studio/api/gallerySourceMedia.ts` finds
the print's own recorded conditioning bytes, and a text-to-image print stays
silent. The same module maps a middleware `401` to `unavailable_auth` so a
keyed host reached without a key gets the API-key disclosure instead of a
swallowed error. Desktop's Lightbox
primary button and its right-click item both go through `reuseSettings`, which
is the only path that attaches retained authority (`composer.set` invalidates
it). A same-host reuse session is one-time, short-lived, and bound to the exact
target request — on a keyless host to one stable anonymous subject; cross-host
reuse remains a client download-and-upload relay.

Queue source previews on every GUI surface read the owning host's authenticated
`/api/queue/:id/input-thumbnail` route. Keep source thumbnails distinct from live
render previews, bound response bytes, cancel obsolete requests, and fence results
by host, server instance and job identity. Never infer source bytes from a local
submission cache or a provenance filename. Sibling output retention is covered
through queue cleanup, trash/restore, permanent deletion and startup reconciliation.
Active queue sets remain independently encrypted and job-bound. Future completed
library handoffs canonicalize identical ordered visible-media contracts within
the existing limits (64 entries, 512 MiB total and 64 MiB aggregate Memory payloads). Per-job
provenance and empty-presence markers stay in separate private sets. Publish all
replacement pins in one authority commit before releasing old pins; retry and
SQLite repair preserve that authority. Different accompanying inputs, roles,
positions or sinks, larger sets and historical outputs keep their existing
storage. This is whole-contract sharing, not arbitrary per-member deduplication.

Library mirrors transfer retained inputs through the authenticated, exact-output
`GET`/`PUT /api/gallery/source-media/:filename/transfer` contract. The offer
captures public output digest, recipe, archive identity and input slots together.
The framed import streams bounded private staging, validates every member digest,
pins before committing the destination archive binding, and acknowledges only
after retention is durable. An equal retry repairs SQLite's projection; a changed
output or existing different binding is refused. Imported siblings share an
owner-local synthetic encrypted set, without reusing a generation job's authority.
Native Mac and Tauri desktop copy completion includes this handoff, including
cache, automatic mirrors and existing-file repair paths; capture source archive identity before output download, then verify source
identity and destination output digest/recipe before attaching inputs.

Before importing a source-bearing copy, clients check destination readiness. Windows local destinations currently cannot receive retained inputs, so those copies are refused before creating a local library output. Source-free copies remain supported, and Windows clients can recall retained sources from a supported remote machine. Destination queue-durability support alone does not prove an older server implements the transfer receiver.

`/api/capabilities.retained_media_transfer = { "protocol_version": 1 }` is the explicit receiver-readiness contract. It is absent on older servers and while the encrypted Unix media lifecycle is unready. Clients require this supported version before importing source-bearing outputs; queue `durable_media` alone does not establish transfer support.

Native Mac mirror compatibility keeps exact output digests, sizes and recipes. Embedded metadata may omit archive-only `job_id` and `generation_time_ms`, or carry the same version without its `(revision date)` suffix. Only those differences are accepted; conflicting facts present on both sides and unknown recipe fields remain significant. Existing-copy matching uses the same contract before byte comparison, so retries repair inputs without duplicate outputs.

Retained image previews use authenticated `GET /api/gallery/source-media/:filename/:member_id/thumbnail`. Resolve exact publication-owned members, bound original/decode memory and concurrent rendering, return a 320-pixel PNG with private/no-store headers, and never accept paths or provenance filenames as byte authority. Client response reads are bounded to 2 MiB. Preview bytes stay separate from descriptor/session authority.
