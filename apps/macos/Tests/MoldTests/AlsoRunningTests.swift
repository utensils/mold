import Foundation
import MoldClient
import Testing

@testable import Mold

/// What belongs under **Also Running**, and what is already on screen as
/// something else.
///
/// **Fails today**: the Queue pane draws generations and nothing else, so a
/// machine preparing weights or running a durable sequence reads as idle.
@MainActor
struct AlsoRunningTests {

    private let plato = UUID()

    private func item(_ id: String, kind: String, phase: String = "running",
                      current: Int? = nil, total: Int? = nil,
                      canCancel: Bool = true) -> ActiveWorkItem {
        let json = #"""
        {"id": "\#(id)", "kind": "\#(kind)", "phase": "\#(phase)",
         "created_at_unix_ms": 1, "updated_at_unix_ms": 1,
         "current": \#(current.map { "\($0)" } ?? "null"),
         "total": \#(total.map { "\($0)" } ?? "null"),
         "can_cancel": \#(canCancel)}
        """#
        return try! MoldJSON.decoder.decode(ActiveWorkItem.self, from: Data(json.utf8))
    }

    private func reported(_ items: [ActiveWorkItem], stale: Bool = false) -> [FleetActiveWork] {
        items.map { FleetActiveWork(host: plato, item: $0, stale: stale, unavailableKind: false) }
    }

    private func rows(_ items: [ActiveWorkItem], queued: Set<String> = [],
                      upscales: [(key: UpscaleStore.Key, job: VideoUpscaleJob)] = [])
        -> [AlsoRunningRow] {
        AlsoRunning.rows(reported: reported(items), queuedIDs: [plato: queued],
                         upscales: upscales)
    }

    /// The exclusion is "the pane is already drawing it", not a kind list --
    /// which is what covers the ephemeral chain too, since it reports as an
    /// ordinary generation.
    @Test func aRowTheQueueIsAlreadyDrawingIsNotRepeated() {
        let drawn = rows([item("job-1", kind: "generation")], queued: ["job-1"])
        #expect(drawn.isEmpty)
        #expect(rows([item("job-1", kind: "generation")]).count == 1,
                "a generation with no queue row of its own still shows")
    }

    /// Downloads have their own surface with their own byte meter and their
    /// own cancel; repeating them here would be two places to press.
    @Test func downloadsAreNotRepeated() {
        #expect(rows([item("d-1", kind: "download")]).isEmpty)
    }

    /// Everything the scheduler owns stands, INCLUDING a kind added after
    /// this build -- a machine doing something this app has never heard of is
    /// still busy, and saying so is the point.
    @Test func everyOtherKindStands() {
        let kinds = ["sequence", "prompt_expansion", "standalone_upscale", "post_upscale",
                     "admin_model_load", "admin_model_unload", "reticulating"]
        let drawn = rows(kinds.enumerated().map { item("w-\($0.offset)", kind: $0.element) })
        #expect(drawn.count == kinds.count)
        #expect(drawn.first { $0.title == "Prompt rewrite" } != nil)
        #expect(drawn.first { $0.title == "Reticulating" } != nil)
    }

    /// A clip upscale is in NO snapshot -- the host runs it outside the
    /// scheduler -- so this app's own following is the only place it appears.
    @Test func aClipUpscaleThisAppStartedIsARowOfItsOwn() {
        let key = UpscaleStore.Key(host: plato, filename: "clip.mp4")
        let job = FakeFixtures.framewiseJob("vup-1", state: "running", done: 31, total: 124)
        let drawn = rows([], upscales: [(key: key, job: job)])
        #expect(drawn.count == 1)
        #expect(drawn[0].title == "Upscale")
        #expect(drawn[0].subject == "clip.mp4")
        #expect(drawn[0].detail == "Upscaling frame 32 of 124")
        #expect(drawn[0].progress == 0.25)
        #expect(drawn[0].canCancel)
        #expect(drawn[0].canPause)
    }

    /// A clip upscale IS scheduler work: `upscale_frame` routes every frame
    /// through `schedule_standalone_upscale` on any host with a v2 scheduler
    /// or a GPU worker (`video_upscale.rs:1271-1281`), and that mints a NEW
    /// uuid per frame (`routes.rs:2508-2546`). So the host reports a
    /// `standalone_upscale` row beside this app's own -- two rows both
    /// titled "Upscale", the reported one changing identity every poll for
    /// the length of a 124-frame job.
    ///
    /// **Fails today**: `AlsoRunning.rows` draws both.
    @Test func aClipUpscaleDrawsOneRowNotTwo() {
        let key = UpscaleStore.Key(host: plato, filename: "clip.mp4")
        let job = FakeFixtures.framewiseJob("vup-1", state: "running", done: 31, total: 124)
        let drawn = rows([item("standalone-upscale-\(UUID())", kind: "standalone_upscale")],
                         upscales: [(key: key, job: job)])
        #expect(drawn.count == 1)
        #expect(drawn[0].subject == "clip.mp4", "and it is the row that names the print")
    }

    /// NOT a blanket suppression. A still upscale is the same
    /// `standalone_upscale` work and has no job to follow, so that reported
    /// row is the only feedback there is -- and so is one somebody started
    /// from the web UI.
    @Test func aStandaloneUpscaleThisAppIsNotFollowingStands() {
        #expect(rows([item("standalone-upscale-1", kind: "standalone_upscale")]).count == 1)
    }

    /// A settled job of ours stops suppressing: the machine's row is then
    /// about something else.
    @Test func aSettledJobStopsHidingTheMachinesOwnRow() {
        let key = UpscaleStore.Key(host: plato, filename: "clip.mp4")
        let done = FakeFixtures.framewiseJob("vup-1", state: "completed", done: 9, total: 9)
        let drawn = rows([item("standalone-upscale-1", kind: "standalone_upscale")],
                         upscales: [(key: key, job: done)])
        #expect(drawn.count == 2)
    }

    /// And only on the machine that is busy -- another machine's upscale is
    /// its own row.
    @Test func suppressionIsPerMachine() {
        let socrates = UUID()
        let key = UpscaleStore.Key(host: socrates, filename: "clip.mp4")
        let job = FakeFixtures.framewiseJob("vup-1", state: "running", total: 9)
        let drawn = AlsoRunning.rows(
            reported: reported([item("standalone-upscale-1", kind: "standalone_upscale")]),
            queuedIDs: [:], upscales: [(key: key, job: job)])
        #expect(drawn.contains { $0.host == plato && $0.title == "Upscale" })
    }

    /// Cancel is offered only where this app can actually act. A reported row
    /// saying `can_cancel: true` is a durable sequence, whose cancel is an
    /// endpoint family this app does not speak -- a button that quietly does
    /// nothing is worse than no button.
    @Test func aReportedRowOffersNoCancelItCannotPerform() {
        let sequence = rows([item("s-1", kind: "sequence", canCancel: true)])
        #expect(sequence[0].canCancel == false)
        #expect(AlsoRunningActions(sequence[0]).offered().isEmpty)
    }

    /// A settled upscale keeps its row until it is dismissed: somebody who
    /// started it is owed the answer.
    @Test func aSettledUpscaleStaysUntilDismissed() {
        let key = UpscaleStore.Key(host: plato, filename: "clip.mp4")
        let failed = FakeFixtures.framewiseJob(
            "vup-1", state: "failed", error: "ffprobe is required for Framewise upscale")
        let drawn = rows([], upscales: [(key: key, job: failed)])
        #expect(drawn[0].detail == "ffprobe is required for Framewise upscale")
        #expect(drawn[0].isSettled)
        #expect(!drawn[0].canCancel)
        let offered = AlsoRunningActions(drawn[0]).offered().map(\.title)
        #expect(offered == ["Dismiss"])
    }

    /// A paused clip job offers Resume and not Pause.
    @Test func aPausedUpscaleOffersResume() {
        let key = UpscaleStore.Key(host: plato, filename: "clip.mp4")
        let paused = FakeFixtures.framewiseJob("vup-1", state: "paused", done: 60, total: 124)
        let drawn = rows([], upscales: [(key: key, job: paused)])
        let offered = AlsoRunningActions(drawn[0]).offered().map(\.title)
        #expect(offered.contains("Resume"))
        #expect(!offered.contains("Pause"))
        #expect(offered.last == "Cancel", "what cannot be taken back is last")
    }

    /// A stale row says it is the last thing heard rather than pretending to
    /// be current.
    @Test func aStaleRowSaysSo() {
        let drawn = AlsoRunning.rows(
            reported: reported([item("s-1", kind: "sequence")], stale: true),
            queuedIDs: [:], upscales: [])
        #expect(drawn[0].isStale)
    }

    /// Two prints being upscaled keep a stable order, so the list does not
    /// reshuffle on every poll.
    @Test func upscaleRowsKeepAStableOrder() {
        let job = FakeFixtures.framewiseJob("vup", state: "running", total: 9)
        let drawn = rows([], upscales: [
            (key: UpscaleStore.Key(host: plato, filename: "zeta.mp4"), job: job),
            (key: UpscaleStore.Key(host: plato, filename: "alpha.mp4"), job: job),
        ])
        #expect(drawn.map(\.subject) == ["alpha.mp4", "zeta.mp4"])
    }
}
