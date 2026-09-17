import Foundation

/// How much a route's stream will hold for a consumer that has fallen behind.
///
/// Every one of these is consumed on the main actor and fanned out
/// synchronously. During a burst -- a bulk import, an `emptyTrash`, a batch
/// settling, one `gallery_*` frame per print, each carrying a whole
/// `GalleryPrint` -- an `.unbounded` policy makes memory track the BURST
/// rather than what the app can apply, and the UI then spends seconds
/// applying events that are already stale.
///
/// `AsyncStream` has no `yield` that blocks, so a ceiling is the only lever
/// there is at this level: below it the pipeline is lazy (`MoldLines`,
/// `ServerSentEventStream`) and holds nothing at all, and above it these
/// numbers are set where reaching them already means the consumer is further
/// behind than catching up would help.
///
/// EVERY stream states its policy. `StreamBufferingContractTests` reads this
/// package's source and fails on an `AsyncThrowingStream` that does not.
public enum StreamBuffering {
    /// Frames on a route whose messages are each their own fact, so losing
    /// one loses information: a job landing, a print appearing, a download's
    /// progress for one of several models. Generous, because the ceiling is a
    /// last resort rather than a working limit.
    public static let frames = 512

    /// A route whose every message is a COMPLETE picture, where an older one
    /// says nothing the newer one does not: a resources telemetry snapshot, a
    /// batch status (`batchEvents`' own doc comment: "Every frame is a
    /// COMPLETE status, never a delta"). Keeping one is keeping the answer.
    public static let latestOnly = 1
}
