import Foundation

/// What one `GET /api/downloads/stream` frame is allowed to do to a client's
/// row list.
///
/// A port of the reducer's own arms (`desktop/src/lib/downloads.ts:54-131`),
/// here because "has an id and is not terminal" is NOT the same question as
/// "names a download job". `catalog_ready`'s `id` is the CATALOG entry's
/// (`types.rs:13118-13125`, emitted at `downloads.rs:521-523` after the
/// group's last terminal frame), so reading it as a job id invents a row --
/// keyed by `hf:owner/repo`, nameless, stuck at "Starting…", with a Cancel
/// that would 404 -- that nothing ever removes, because there is no further
/// frame for that id. Desktop ignores the frame by name
/// (`downloads.ts:127-128`) and so does this.
public extension DownloadEvent {
    enum Effect: Equatable, Sendable {
        /// The first frame on every connection: the whole listing, replacing
        /// what the client held.
        case snapshot
        /// This id IS a download job, and the frame is allowed to create its
        /// row: `enqueued` and `started`.
        case introduce
        /// A row this client already knows, moving. Never a reason to create
        /// one: `downloads.ts:96-97` returns the state untouched for a
        /// `progress` naming a job it has not seen.
        case update
        /// It reached a terminal state and belongs in the finished list.
        case settle
        /// Drop the row without recording an outcome -- `dequeued` follows
        /// the `job_cancelled` that already did (`downloads.rs:586-591`).
        case forget
        /// Says nothing about any row: `catalog_ready`, and any frame a newer
        /// server adds that this build does not know.
        case ignore
    }

    var effect: Effect {
        switch type {
        case "snapshot": listing == nil ? .ignore : .snapshot
        case "enqueued", "started": id == nil ? .ignore : .introduce
        case "progress", "file_done": id == nil ? .ignore : .update
        case "job_done", "job_failed", "job_cancelled": id == nil ? .ignore : .settle
        case "dequeued": id == nil ? .ignore : .forget
        default: .ignore
        }
    }
}
