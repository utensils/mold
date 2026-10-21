# MoldClient

The Foundation-only contract and decision layer used by both native apps.
`HTTPBackend` and `MoldClientTesting.FakeBackend` implement the same concern
protocols; app views never downcast a backend to reach a route.

## Duration-based videos

`MoldVideoBackend` follows `/api/video-jobs` parents. Require positive
`Capabilities.canCreateVideoJob`; Library extension additionally requires
`canExtendLibraryVideo`. `VideoDurationDraft` keeps requested milliseconds
independent of model frame grids. Optional scene durations must sum exactly.
The server owns safe clip lengths, overlap, joins, trimming and source eligibility.

`VideoPlanController` debounces display-only plans for 250 milliseconds. Its
request projection removes every input body and media descriptor; the separate
Library filename and archive identity remain. A planning response has an empty
validation revision and cannot admit work. Generate must prepare the captured
media once, fully validate that exact request, and send its nonempty revision.
Prompt strings have literal JSON semantics. UInt64 counts and durations decode
losslessly from decimal strings or integers and encode as decimal strings.

Creation and every mutation carry the captured destination installation.
Persist the operation UUID and scalar owner identity before creation; reconcile
that UUID after a lost response instead of replaying creation. Full snapshot SSE
uses a bounded latest-only buffer. Queue controls use the server's lifecycle flags.

## Restarting a saved chain

`VideoRestartController` reads revision-bound restart/send/discard options.
The export contains private, wire-ready `request_json`: forward its exact UTF-8
bytes without decoding, normalizing backslashes, or persisting it. Restarting
creates a separate attempt and leaves its source unchanged.

Destinations must positively advertise `queue.chain_restart`. Stop additionally
requires `queue.chain_restart_controls` and the captured installation header.
Unacknowledged cancellation intent survives relaunch and replays only the same
idempotent cancellation fence. The scalar recovery journal contains no request
body, media, or credentials. Invalid journals block new admission. A status for
an unexpected job ID cannot publish progress or clear a receipt. Deleting saved
recovery inputs uses the revision-bound discard endpoint, never a legacy DELETE;
saved Library videos remain independently owned.

Run the package tests with `swift test --package-path apps/shared/Packages/MoldClient`.
