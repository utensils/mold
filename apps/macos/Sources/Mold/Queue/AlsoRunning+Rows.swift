import Foundation
import MoldClient

/// Which of a machine's activity rows belong under **Also Running**.
///
/// Pure, so "what is drawn twice" is a test rather than something you notice
/// by counting rows on screen.
enum AlsoRunning {
    /// The kind an upscale is scheduled as (`scheduler/mod.rs:13959`).
    private static let upscaleKind = "standalone_upscale"

    /// THREE exclusions, all read off the server:
    ///
    /// - a row whose id IS a queue row on that machine. Every `generation`
    ///   comes from the job registry or the restart-paused projection, so
    ///   this is the generation rule -- but written as "the pane is already
    ///   drawing it" rather than as a kind list, because that is the real
    ///   question and it stays true if a future kind takes queue rows too.
    ///   It also covers the ephemeral chain, which reports as a generation.
    /// - `download`, which this app already draws in the downloads popover
    ///   with a byte meter and a cancel of its own.
    /// - a `standalone_upscale` on a machine where this app is FOLLOWING a
    ///   clip job. `upscale_frame` routes every frame through
    ///   `schedule_standalone_upscale` on any host with a v2 scheduler or a
    ///   GPU worker (`video_upscale.rs:1271-1281`), which mints a fresh uuid
    ///   per frame (`routes.rs:2508-2546`) -- so the machine's own row is a
    ///   duplicate of this app's, titled the same, and its identity changes
    ///   on essentially every poll for the length of the job. The job row is
    ///   the better one: it names the print and counts the whole clip.
    ///
    ///   Deliberately NOT a blanket suppression of the kind. A STILL upscale
    ///   is the same scheduler work with no job to follow, and so is one
    ///   somebody started from another client -- there the machine's row is
    ///   the only feedback there is.
    ///
    /// Everything else stands: a durable `sequence`, `prompt_expansion`,
    /// `post_upscale`, `admin_model_load` / `admin_model_unload`, and any
    /// scheduler kind added after this build -- a machine doing something
    /// this app has never heard of is still busy, and saying so is the whole
    /// point of the section.
    static func rows(
        reported: [FleetActiveWork],
        queuedIDs: [MoldHost.ID: Set<String>],
        upscales: [(key: UpscaleStore.Key, job: VideoUpscaleJob)]
    ) -> [AlsoRunningRow] {
        let following = Set(
            upscales.filter { !$0.job.state.isTerminal }.map(\.key.host))
        let fromMachines = reported
            .filter { $0.item.kind != "download" }
            .filter { !(queuedIDs[$0.host] ?? []).contains($0.item.id) }
            .filter { !($0.item.kind == upscaleKind && following.contains($0.host)) }
            .map { AlsoRunningRow(host: $0.host, work: .reported($0)) }
        // Appended, not merged by time: an upscale carries no submission
        // stamp of its own, and inventing one to sort it by would be a
        // number this app made up.
        let mine = upscales
            .sorted { $0.key.filename < $1.key.filename }
            .map { AlsoRunningRow(host: $0.key.host,
                                  work: .upscale(filename: $0.key.filename, job: $0.job)) }
        return fromMachines + mine
    }
}
