import Foundation
import Testing
import MoldClientTesting
@testable import MoldClient

extension VideoJobCoordinatorTests {
    func listing(_ jobs: [VideoJob], cursor: String?) throws -> VideoJobListing {
        let array = String(data: try MoldJSON.encoder.encode(jobs), encoding: .utf8)!
        let suffix = cursor.map { #", "has_more":true,"next_cursor":""# + $0 + #""}"# }
            ?? #", "has_more":false,"next_cursor":null}"#
        return try MoldJSON.decoder.decode(VideoJobListing.self, from: Data(("{\"jobs\":" + array + suffix).utf8))
    }
    @Test func olderRecoverableParentsLoadByStableCursorRatherThanVisibleRowCount() async throws {
        let (backend, owner, saved) = try setup()
        let first = try listing([detail("recent", state: "failed", cancel: false, resume: true, delete: true)], cursor: "opaque-position")
        let older = try listing([detail("older", state: "failed", cancel: false, resume: true, delete: true)], cursor: nil)
        backend.stub("listVideoJobs(limit:cursor:)") { args in
            #expect(args[0] as? Int == 50)
            if args[1] as? String == "opaque-position" { return older }
            return first
        }
        let coordinator = VideoJobCoordinator(defaults: saved)
        await coordinator.discover(owner: owner, backend: backend, current: { true })
        #expect(coordinator.pagination[owner.host.id]?.hasMore == true)
        #expect(coordinator.jobs.map(\.operationId) == ["recent"])
        await coordinator.loadMore(owner: owner, backend: backend, current: { true })
        #expect(coordinator.jobs.map(\.operationId) == ["recent", "older"])
        #expect(coordinator.pagination[owner.host.id]?.hasMore == false)
    }
    @Test func changedOwnerCannotAppendALateHistoryPageOrKeepItsLoadingState() async throws {
        let (backend, owner, saved) = try setup()
        let first = try listing([], cursor: "older"), older = try listing([detail("old", state: "failed")], cursor: nil)
        backend.stub("listVideoJobs(limit:cursor:)", returning: first)
        let coordinator = VideoJobCoordinator(defaults: saved)
        await coordinator.discover(owner: owner, backend: backend, current: { true })
        let gate = AsyncStream<Void>.makeStream()
        backend.stub("listVideoJobs(limit:cursor:)") { _ in
            for await _ in gate.stream { break }; return older
        }
        var current = true
        let old = Task { await coordinator.loadMore(owner: owner, backend: backend, current: { current }) }
        for _ in 0..<100 where backend.count("listVideoJobs(limit:cursor:)") < 2 { await Task.yield() }
        current = false; gate.continuation.finish(); await old.value
        #expect(coordinator.jobs.isEmpty)
        #expect(coordinator.pagination[owner.host.id]?.isLoading == false)
    }
}
