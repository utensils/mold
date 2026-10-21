import Foundation
import Testing
import MoldClientTesting
@testable import MoldClient

extension VideoJobCoordinatorTests {
    @Test func completionAcrossOwnerDriftSettlesOnlyKnownSameInstallation() async throws {
        let (original, owner, defaults) = try setup()
        let coordinator = VideoJobCoordinator(defaults: defaults)
        coordinator.publish(try detail("done"), owner: owner)
        var host = owner.host; host.name = "Refreshed connection"
        let current = VideoJobOwner(host: host, instanceId: owner.instanceId)
        let backend = FakeBackend(host: host)
        backend.stub("status()", returning: try await original.status())
        backend.stub("capabilities()", returning: try await original.capabilities())
        backend.stub("videoJob(id:)", returning: try finishedOutput())
        backend.stub("gallery(etag:)", returning: Fetched<[GalleryPrint]>.fresh([], etag: nil))
        // Production's recoverable listing no longer includes a completed job.
        backend.stub("listVideoJobs(limit:cursor:)", returning: try listing([], cursor: nil))
        await coordinator.discover(owner: current, backend: backend, current: { true })
        #expect(coordinator.jobs.first?.owner == current)
        #expect(coordinator.jobs.first?.detail?.state == .completed)
        #expect(backend.count("videoJob(id:)") == 1)
        let unknown = VideoJobCoordinator(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        backend.stub("listVideoJobs(limit:cursor:)", returning: try listing([finishedOutput()], cursor: nil))
        await unknown.discover(owner: current, backend: backend, current: { true })
        #expect(unknown.jobs.isEmpty, "a Queue is not arbitrary completed server history")
        let replacement = VideoJobOwner(host: host, instanceId: "different-installation")
        let untouched = VideoJobCoordinator(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        untouched.publish(try detail("done"), owner: replacement)
        backend.stub("listVideoJobs(limit:cursor:)", returning: try listing([], cursor: nil))
        let before = backend.count("videoJob(id:)")
        await untouched.discover(owner: current, backend: backend, current: { true })
        #expect(backend.count("videoJob(id:)") == before)
        #expect(untouched.jobs.first?.owner == replacement)
    }
}
