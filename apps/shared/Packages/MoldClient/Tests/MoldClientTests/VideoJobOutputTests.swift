import Foundation
import Testing
import MoldClientTesting
@testable import MoldClient

extension VideoJobCoordinatorTests {
    func finishedOutput(_ id: String = "done") throws -> VideoJob {
        var object = try JSONSerialization.jsonObject(with: MoldJSON.encoder.encode(detail(id, state: "completed", cancel: false, delete: true))) as! [String: Any]
        object["output_filename"] = "movie.mp4"
        return try MoldJSON.decoder.decode(VideoJob.self, from: JSONSerialization.data(withJSONObject: object))
    }
    func outputPrint(trashed: Bool = false, filename: String = "movie.mp4") throws -> GalleryPrint {
        try MoldJSON.decoder.decode(GalleryPrint.self, from: Data("{\"filename\":\"\(filename)\",\"metadata\":{},\"timestamp\":1,\"trashed_at\":\(trashed ? "2" : "null")}".utf8))
    }
    @Test func completedOutputChecksExactOwnerAndDoesNotConfuseMissingWithOffline() async throws {
        let (backend, owner, defaults) = try setup()
        let coordinator = VideoJobCoordinator(defaults: defaults)
        coordinator.publish(try finishedOutput(), owner: owner)
        let row = try #require(coordinator.jobs.first)
        backend.stub("gallery(etag:)", returning: Fetched<[GalleryPrint]>.fresh([], etag: nil))
        #expect(await coordinator.checkOutput(row, backend: backend, current: { false }) == nil)
        #expect(coordinator.jobs.first?.outputAvailable == nil)
        #expect(await coordinator.checkOutput(row, backend: backend, current: { true }) == false)
        #expect(coordinator.jobs.first?.detail?.state == .completed)
        #expect(coordinator.jobs.first?.outputAvailable == false)
        backend.stub("gallery(etag:)", throwing: MoldClientError.unreachable("offline"))
        #expect(await coordinator.checkOutput(row, backend: backend, current: { true }, force: true) == nil)
        #expect(coordinator.jobs.first?.outputMessage != nil)
        #expect(coordinator.jobs.first?.detail?.error == nil)
        backend.stub("gallery(etag:)", returning: Fetched<[GalleryPrint]>.fresh([try outputPrint()], etag: nil))
        #expect(await coordinator.checkOutput(row, backend: backend, current: { true }, force: true) == true)
        coordinator.galleryChanged(.trashed(filename: "movie.mp4"), owner: owner)
        #expect(coordinator.jobs.first?.outputAvailable == false)
        let other = VideoJobOwner(host: owner.host, instanceId: "replacement")
        coordinator.galleryChanged(.restored(filename: "movie.mp4", row: nil), owner: other)
        #expect(coordinator.jobs.first?.outputAvailable == false)
    }
    @Test func deletingAlreadyMissingExactParentSettlesLocallyButGeneric404DoesNot() async throws {
        let (backend, owner, defaults) = try setup()
        let coordinator = VideoJobCoordinator(defaults: defaults)
        coordinator.publish(try finishedOutput(), owner: owner)
        let row = try #require(coordinator.jobs.first)
        backend.stub("deleteVideoJob(id:destinationInstanceId:)", throwing: MoldClientError.http(status: 404, code: "NOT_FOUND", message: "unknown"))
        await coordinator.delete(row, backend: backend, current: { true })
        #expect(coordinator.jobs.count == 1)
        backend.stub("deleteVideoJob(id:destinationInstanceId:)", throwing: MoldClientError.http(status: 404, code: "VIDEO_JOB_NOT_FOUND", message: "gone"))
        await coordinator.delete(row, backend: backend, current: { true })
        #expect(coordinator.jobs.isEmpty)
    }
}

extension VideoJobCoordinatorTests {
    @Test func pendingOutputCheckCoalescesAndCannotRestoreLinkAfterTrashOrDismissal() async throws {
        let (backend, owner, defaults) = try setup()
        let coordinator = VideoJobCoordinator(defaults: defaults)
        coordinator.publish(try finishedOutput(), owner: owner)
        let row = try #require(coordinator.jobs.first)
        let gate = AsyncStream<Void>.makeStream()
        let print = try outputPrint()
        backend.stub("gallery(etag:)") { _ in for await _ in gate.stream { break }; return Fetched<[GalleryPrint]>.fresh([print], etag: nil) }
        let pending = Task { await coordinator.checkOutput(row, backend: backend, current: { true }) }
        for _ in 0..<100 where backend.count("gallery(etag:)") == 0 { await Task.yield() }
        #expect(await coordinator.checkOutput(row, backend: backend, current: { true }) == nil)
        #expect(backend.count("gallery(etag:)") == 1)
        coordinator.galleryChanged(.trashed(filename: "movie.mp4"), owner: owner)
        gate.continuation.finish()
        #expect(await pending.value == nil)
        #expect(coordinator.jobs.first?.outputAvailable == false)
        backend.stub("gallery(etag:)", returning: Fetched<[GalleryPrint]>.fresh([try outputPrint(trashed: true)], etag: nil))
        #expect(await coordinator.checkOutput(row, backend: backend, current: { true }, force: true) == false)
        coordinator.dismiss(row)
        #expect(coordinator.jobs.isEmpty)
        #expect(await coordinator.checkOutput(row, backend: backend, current: { true }, force: true) == nil)
    }
    @Test func deletionHasImmediateBusyStateAndDoesNotBlockLocalCompletedDismissal() async throws {
        let (backend, owner, defaults) = try setup()
        let coordinator = VideoJobCoordinator(defaults: defaults)
        coordinator.publish(try finishedOutput(), owner: owner)
        let row = try #require(coordinator.jobs.first)
        let gate = AsyncStream<Void>.makeStream()
        backend.stub("deleteVideoJob(id:destinationInstanceId:)") { _ in for await _ in gate.stream { break }; return () }
        let deleting = Task { await coordinator.delete(row, backend: backend, current: { true }) }
        for _ in 0..<100 where backend.count("deleteVideoJob(id:destinationInstanceId:)") == 0 { await Task.yield() }
        #expect(coordinator.isDeleting(row))
        coordinator.dismiss(row)
        #expect(coordinator.jobs.isEmpty)
        gate.continuation.finish(); await deleting.value
        #expect(!coordinator.isDeleting(row))
        #expect(coordinator.jobs.isEmpty)
    }
    @Test func staleCompletedGestureCannotDismissNowActiveJob() throws {
        let (_, owner, defaults) = try setup()
        let coordinator = VideoJobCoordinator(defaults: defaults)
        coordinator.publish(try finishedOutput(), owner: owner)
        let stale = try #require(coordinator.jobs.first)
        coordinator.publish(try detail("done", state: "running"), owner: owner)
        coordinator.dismiss(stale)
        #expect(coordinator.jobs.first?.detail?.state == .running)
    }
}

extension VideoJobCoordinatorTests {
    @Test func replacementOutputOwnerDoesNotWaitForStaleGalleryRead() async throws {
        let (old, owner, defaults) = try setup()
        let coordinator = VideoJobCoordinator(defaults: defaults)
        let done = try finishedOutput(), print = try outputPrint()
        coordinator.publish(done, owner: owner)
        let original = try #require(coordinator.jobs.first)
        let gate = AsyncStream<Void>.makeStream()
        old.stub("gallery(etag:)") { _ in for await _ in gate.stream { break }; return Fetched<[GalleryPrint]>.fresh([print], etag: nil) }
        var current = true
        let waiting = Task { await coordinator.checkOutput(original, backend: old, current: { current }) }
        for _ in 0..<100 where old.count("gallery(etag:)") == 0 { await Task.yield() }
        var host = owner.host; host.name = "Fresh metadata"
        let replacement = VideoJobOwner(host: host, instanceId: owner.instanceId)
        let fresh = FakeBackend(host: host)
        fresh.stub("status()", returning: try await old.status())
        fresh.stub("capabilities()", returning: try await old.capabilities())
        fresh.stub("gallery(etag:)", returning: Fetched<[GalleryPrint]>.fresh([print], etag: nil))
        current = false; coordinator.publish(done, owner: replacement)
        let row = try #require(coordinator.jobs.first)
        #expect(await coordinator.checkOutput(row, backend: fresh, current: { true }) == true)
        gate.continuation.finish(); _ = await waiting.value
        #expect(coordinator.jobs.first?.outputAvailable == true)
    }
}

private actor HeldVideoGalleryRead {
    var waiter: CheckedContinuation<Void, Never>?
    var released = false
    func wait() async {
        guard !released else { return }
        await withCheckedContinuation { waiter = $0 }
    }
    func release() { released = true; waiter?.resume(); waiter = nil }
}

extension VideoJobCoordinatorTests {
    @Test func cancelledOutputFlightCanBeReplacedBeforeItsSocketReturns() async throws {
        let (backend, owner, defaults) = try setup()
        let coordinator = VideoJobCoordinator(defaults: defaults)
        coordinator.publish(try finishedOutput(), owner: owner)
        let row = try #require(coordinator.jobs.first), print = try outputPrint()
        let gate = HeldVideoGalleryRead()
        backend.stub("gallery(etag:)") { _ in await gate.wait(); return Fetched<[GalleryPrint]>.fresh([print], etag: nil) }
        let old = Task { await coordinator.checkOutput(row, backend: backend, current: { true }) }
        for _ in 0..<100 where backend.count("gallery(etag:)") == 0 { await Task.yield() }
        old.cancel()
        backend.stub("gallery(etag:)", returning: Fetched<[GalleryPrint]>.fresh([print], etag: nil))
        #expect(await coordinator.checkOutput(row, backend: backend, current: { true }) == true)
        await gate.release(); _ = await old.value
        #expect(coordinator.jobs.first?.outputAvailable == true)
    }
    @Test func replacementServerDuringGalleryReadCannotPublishItsOutputAsOriginals() async throws {
        let (backend, owner, defaults) = try setup()
        let coordinator = VideoJobCoordinator(defaults: defaults)
        coordinator.publish(try finishedOutput(), owner: owner)
        let row = try #require(coordinator.jobs.first), print = try outputPrint()
        let gate = HeldVideoGalleryRead()
        backend.stub("gallery(etag:)") { _ in await gate.wait(); return Fetched<[GalleryPrint]>.fresh([print], etag: nil) }
        let read = Task { await coordinator.checkOutput(row, backend: backend, current: { true }) }
        for _ in 0..<100 where backend.count("gallery(etag:)") == 0 { await Task.yield() }
        backend.stub("status()", returning: try MoldJSON.decoder.decode(ServerStatus.self,
            from: Data(#"{"version":"test","busy":false,"uptime_secs":1,"instance_id":"replacement"}"#.utf8)))
        await gate.release()
        #expect(await read.value == nil)
        #expect(coordinator.jobs.first?.outputAvailable == nil)
    }
}

extension VideoJobCoordinatorTests {
    @Test func queueRecheckRereadsLiveParentsAndOnlyForcedRefreshRechecksSavedVideos() async throws {
        let (backend, owner, defaults) = try setup()
        let coordinator = VideoJobCoordinator(defaults: defaults)
        coordinator.publish(try detail("live"), owner: owner)
        coordinator.publish(try finishedOutput(), owner: owner)
        var progressed = try JSONSerialization.jsonObject(with: MoldJSON.encoder.encode(detail("live"))) as! [String: Any]
        progressed["completed_clips"] = "2"
        progressed["progress_phase"] = "rendering"
        let next = try MoldJSON.decoder.decode(VideoJob.self, from: JSONSerialization.data(withJSONObject: progressed))
        backend.stub("videoJob(id:)", returning: next)
        backend.stub("gallery(etag:)", returning: Fetched<[GalleryPrint]>.fresh([try outputPrint()], etag: nil))
        await coordinator.recheck(backend: { _ in backend }, current: { _ in false })
        #expect(backend.count("videoJob(id:)") == 0, "a refused owner keeps its last-known row")
        await coordinator.recheck(backend: { _ in backend }, current: { _ in true })
        #expect(backend.count("videoJob(id:)") == 1, "a completed parent is not re-read")
        #expect(coordinator.jobs.first { $0.operationId == "live" }?.detail?.completedClips == 2)
        #expect(coordinator.jobs.first { $0.operationId == "done" }?.outputAvailable == true)
        backend.stub("gallery(etag:)", returning: Fetched<[GalleryPrint]>.fresh([], etag: nil))
        await coordinator.recheck(backend: { _ in backend }, current: { _ in true })
        #expect(coordinator.jobs.first { $0.operationId == "done" }?.outputAvailable == true, "poll trusts the known answer")
        await coordinator.recheck(backend: { _ in backend }, current: { _ in true }, force: true)
        #expect(coordinator.jobs.first { $0.operationId == "done" }?.outputAvailable == false)
    }
}
