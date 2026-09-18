import Foundation
import MoldClient

/// Making prints bigger, and following the clip jobs that takes.
///
/// It lives beside the queue rather than beside the Library because a clip
/// upscale is WORK: it outlives the pane that started it, it belongs on the
/// machine that holds the print, and the Queue pane is where this app says
/// what is running.
///
/// It is not a queue row: `/api/queue` never lists it. It IS scheduler work
/// -- `upscale_frame` submits every frame through
/// `schedule_standalone_upscale` (`video_upscale.rs:1271-1281`) -- but what
/// `/api/activity` then reports is ONE FRAME, under a fresh uuid each time,
/// with no idea which print it belongs to or how many frames are left. The
/// durable job is the only thing that knows that, and polling it is the only
/// way anybody learns where it got to. `AlsoRunning` is where the two meet.
@MainActor
@Observable
final class UpscaleStore {
    /// One print on one machine. A print exists on every machine that holds
    /// a copy, and the job belongs to exactly one of them.
    struct Key: Hashable {
        let host: MoldHost.ID
        let filename: String
    }

    let hosts: HostStore
    let models: ModelStore
    /// Refreshed when a job lands, because the bigger print is a NEW row in
    /// that machine's gallery and nothing else will go and look.
    let library: LibraryStore

    /// How long between two asks about a running job. A constructor
    /// parameter, never a constant: a test drives a job through four polls
    /// in a millisecond rather than sleeping three seconds.
    let interval: Duration

    /// The job this app is following for each print. A settled job stays
    /// here so the row can say how it ended. `internal(set)` because
    /// `private(set)` does not cross a file boundary, even within one type.
    internal(set) var jobs: [Key: VideoUpscaleJob] = [:]

    /// Prints whose upscale this app has asked for and not yet heard back
    /// about -- a still running inline, or a clip job with no id yet.
    internal(set) var working: Set<Key> = []

    /// What became of a STILL upscale on each print.
    ///
    /// A still is synchronous -- `POST /api/gallery/upscale` publishes the
    /// bigger picture before it answers -- so there is no durable job to
    /// follow and, until this, no feedback of any kind: a person pressed
    /// Make Bigger... and for up to five minutes nothing happened, then a
    /// tile quietly appeared somewhere in the grid. This is what the Also
    /// Running row is drawn from.
    internal(set) var stills: [Key: StillUpscale] = [:]

    /// Prints whose job is being ASKED about on a timer right now.
    ///
    /// Distinct from holding a job in `jobs`: a poll that gave up because the
    /// machine went away leaves the job behind and stops following it, and
    /// recovery has to be able to tell those two apart -- otherwise it looks
    /// at the id it already holds, decides it is up to date, and the job is
    /// never followed again.
    internal(set) var following: Set<Key> = []

    // Not `private`: `UpscaleStore+Jobs.swift` is where every decision about
    // these lives, and `private` does not cross a file boundary.

    /// Bumped by every start and every recovery. An answer arriving under an
    /// older number is an answer about a job nobody is following any more.
    var epochs: [Key: Int] = [:]
    var pollers: [Key: Task<Void, Never>] = [:]

    /// One number per poll, so a poll that is replaced gives back only its
    /// OWN claim on `following`. The epoch cannot answer this: a pause and a
    /// resume replace the poll without replacing the job, so both polls run
    /// under the same epoch and the dying one would retire the live one's
    /// claim.
    var pollTokens: [Key: Int] = [:]

    init(hosts: HostStore, models: ModelStore, library: LibraryStore,
         interval: Duration = .milliseconds(750)) {
        self.hosts = hosts
        self.models = models
        self.library = library
        self.interval = interval
    }
}

/// A still upscale, which has no durable job of its own.
enum StillUpscale: Equatable {
    case working
    /// The machine's own name for the bigger picture it published.
    case done(filename: String)
    /// The machine's own sentence about why it did not.
    case failed(String)
}
