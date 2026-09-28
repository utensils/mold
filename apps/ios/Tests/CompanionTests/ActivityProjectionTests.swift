import Foundation
import MoldClient
import Testing

@testable import MoldCompanion

/// What the Live Activity says in each phase, and when it goes stale.
struct ActivityProjectionTests {
    private func batch(_ state: String) throws -> BatchStatus {
        try MoldJSON.decoder.decode(BatchStatus.self, from: Data(
            #"{"id":"b1","client_batch_id":"c1","children":[{"index":1,"job_id":"j1","state":"\#(state)"}]}"#.utf8))
    }

    private func progress(_ json: String) throws -> JobProgress {
        try MoldJSON.decoder.decode(JobProgress.self, from: Data(json.utf8))
    }

    @Test func aRunningRenderSaysWhatItIsDoingAndHowFar() throws {
        let now = Date(timeIntervalSince1970: 1_000)
        let state = try #require(ActivityProjection.state(
            for: .running(try batch("running"), try progress(#"{"step":18,"total":28,"stage":"Denoising"}"#)),
            machine: "workstation", waiting: 2, remaining: .seconds(12), preview: "c1.jpg", now: now))
        #expect(state.phase == .running)
        #expect(state.sentence == "Adding detail")
        #expect(state.figure == "denoise 18/28 · workstation")
        #expect(state.fraction == 18.0 / 28.0)
        #expect(state.endsAt == now.addingTimeInterval(12))
        #expect(state.waiting == 2)
        #expect(ActivityProjection.staleDate(for: state, now: now) == now.addingTimeInterval(12 + 300))
    }

    @Test func nothingIsShownWhileIdleOrSubmitting() {
        #expect(ActivityProjection.state(for: .idle, machine: "m", waiting: 0, remaining: nil, preview: nil) == nil)
        #expect(ActivityProjection.state(for: .submitting, machine: "m", waiting: 0, remaining: nil, preview: nil) == nil)
    }

    @Test func aFinishedRenderNamesItsMachineAndNeverGoesStale() throws {
        let outcome = BatchOutcome(chainResults: [BatchResult(filename: "a.png")], failures: [])
        let state = try #require(ActivityProjection.state(for: .finished(outcome, host: UUID()), machine: "workstation",
                                                          waiting: 0, remaining: nil, preview: nil))
        #expect(state.phase == .finished)
        #expect(state.sentence == "Finished on workstation")
        #expect(ActivityProjection.staleDate(for: state) == nil)
    }

    @Test func theStateStaysFarUnderActivityKitsLimit() throws {
        let state = try #require(ActivityProjection.state(
            for: .running(try batch("running"), try progress(#"{"step":1,"total":50,"stage":"Denoising"}"#)),
            machine: String(repeating: "m", count: 64), waiting: 9, remaining: .seconds(600), preview: "\(UUID()).jpg"))
        #expect(try JSONEncoder().encode(state).count < 1024)
    }
}
