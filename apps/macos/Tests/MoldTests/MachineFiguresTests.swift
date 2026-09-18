import Foundation
import MoldClient
import Testing

@testable import Mold

/// `MachineFigures` is what the Machines page shows for "Work here" and
/// "Models here" -- pinned here at the pure-function level because a live
/// pane can't distinguish "this store hasn't answered for this host yet"
/// from "it answered and this host has none" without a real network fake.
struct MachineFiguresTests {
    @Test func aStoreThatHasNotAnsweredShowsAnEmDashNotAClaim() {
        #expect(MachineFigures.modelFigure(ready: nil) == "—")
        #expect(MachineFigures.workFigure(live: nil) == "—")
    }

    @Test func anEmptyAnswerIsARealFactNotAnAbsence() {
        #expect(MachineFigures.modelFigure(ready: []) == "None installed")
        #expect(MachineFigures.workFigure(live: []) == "Nothing queued")
    }

    @Test func modelsTotalTheirSize() {
        let ready = [
            FakeFixtures.model("flux-dev:q4", sizeGb: 5.1),
            FakeFixtures.model("flux-dev:q8", sizeGb: 5.1),
        ]
        #expect(MachineFigures.modelFigure(ready: ready) == "2 installed · 10.2 GB")
    }

    @Test func modelsWithNoKnownSizeStillCount() {
        let ready = [FakeFixtures.model("flux-dev:q4")]
        #expect(MachineFigures.modelFigure(ready: ready) == "1 installed")
    }

    @Test func liveEntriesSplitIntoQueuedAndRunning() {
        let live = [
            FakeFixtures.queueEntry("a", state: "queued"),
            FakeFixtures.queueEntry("b", state: "running"),
        ]
        #expect(MachineFigures.workFigure(live: live) == "1 queued, 1 running")
    }

    /// A prompt rewrite never becomes a queue row, and a machine mid-rewrite
    /// read "Nothing queued" on its page and its card (2026-09-17).
    @Test func workNeverBecomingAQueueRowStillCounts() {
        #expect(MachineFigures.workFigure(live: [], alsoRunning: 1) == "Nothing queued · 1 also running")
        let live = [FakeFixtures.queueEntry("a", state: "queued")]
        #expect(MachineFigures.workFigure(live: live, alsoRunning: 2) == "1 queued, 0 running · 2 also running")
        // Unknown queue stays unknown whatever else is reported.
        #expect(MachineFigures.workFigure(live: nil, alsoRunning: 1) == "—")
    }
}
