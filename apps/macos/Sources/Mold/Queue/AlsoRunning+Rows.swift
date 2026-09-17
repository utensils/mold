import Foundation
import MoldClient

/// Which of a machine's activity rows belong under **Also Running**.
///
/// Pure, so "what is drawn twice" is a test rather than something you notice
/// by counting rows on screen.
enum AlsoRunning {
    /// TWO exclusions, both read off `routes_activity.rs:180-396`:
    ///
    /// - a row whose id IS a queue row on that machine. Every `generation`
    ///   comes from the job registry or the restart-paused projection, so
    ///   this is the generation rule -- but written as "the pane is already
    ///   drawing it" rather than as a kind list, because that is the real
    ///   question and it stays true if a future kind takes queue rows too.
    ///   It also covers the ephemeral chain, which reports as a generation.
    /// - `download`, which this app already draws in the downloads popover
    ///   with a byte meter and a cancel of its own.
    ///
    /// Everything else stands: a durable `sequence`, `prompt_expansion`,
    /// `standalone_upscale`, `post_upscale`, `admin_model_load` /
    /// `admin_model_unload`, and any scheduler kind added after this build --
    /// a machine doing something this app has never heard of is still busy,
    /// and saying so is the whole point of the section.
    static func rows(
        reported: [FleetActiveWork],
        queuedIDs: [MoldHost.ID: Set<String>],
        upscales: [(key: UpscaleStore.Key, job: VideoUpscaleJob)]
    ) -> [AlsoRunningRow] {
        let fromMachines = reported
            .filter { $0.item.kind != "download" }
            .filter { !(queuedIDs[$0.host] ?? []).contains($0.item.id) }
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

    /// The same list, narrowed to one machine -- what a section under that
    /// machine's name draws.
    static func rows(
        on host: MoldHost.ID, reported: [FleetActiveWork],
        queuedIDs: [MoldHost.ID: Set<String>],
        upscales: [(key: UpscaleStore.Key, job: VideoUpscaleJob)]
    ) -> [AlsoRunningRow] {
        rows(reported: reported, queuedIDs: queuedIDs, upscales: upscales)
            .filter { $0.host == host }
    }
}
