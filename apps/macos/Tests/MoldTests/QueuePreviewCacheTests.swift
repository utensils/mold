import AppKit
import MoldClient
import Testing
@testable import Mold

@MainActor
struct QueuePreviewCacheTests {
    @Test func routeAndCredentialChangesCannotReuseAnotherOriginsPreview() async throws {
        let host = MoldHost(name: "fixture", baseURL: URL(string: "http://fixture")!, apiKey: "old-key")
        let backend = FakeBackend(host: host)
        let cache = QueuePreviewCache()
        _ = try await cache.preview(for: .init(host: host, instance: "one", job: "job"), backend: backend)
        var moved = host
        moved.baseURL = URL(string: "http://replacement")!
        _ = try await cache.preview(for: .init(host: moved, instance: "one", job: "job"), backend: backend)
        var rekeyed = moved
        rekeyed.apiKey = "replacement-key"
        _ = try await cache.preview(for: .init(host: rekeyed, instance: "one", job: "job"), backend: backend)
        #expect(backend.callCount("queueInputs") == 3)
        _ = try await cache.preview(for: .init(host: host, instance: "one", job: "job"), backend: backend)
        #expect(backend.callCount("queueInputs") == 3)
    }

    @Test func thrownDescriptorReadReleasesRequestAndCanRetry() async throws {
        let host = MoldHost(name: "fixture", baseURL: URL(string: "http://fixture")!)
        let backend = FakeBackend(host: host)
        let cache = QueuePreviewCache()
        let key = QueuePreviewCache.Key(host: host, instance: "one", job: "job")
        backend.plantedErrors["queueInputs"] = MoldClientError.malformedResponse
        do { _ = try await cache.preview(for: key, backend: backend); Issue.record("Expected descriptor failure") }
        catch MoldClientError.malformedResponse {} catch { Issue.record("Unexpected error: \(error)") }
        backend.plantedErrors["queueInputs"] = nil
        _ = try await cache.preview(for: key, backend: backend)
        _ = try await cache.preview(for: key, backend: backend)
        #expect(backend.callCount("queueInputs") == 2)
    }

    @Test func leastRecentlyUsedPreviewIsEvictedAtCapacity() async throws {
        let host = MoldHost(name: "fixture", baseURL: URL(string: "http://fixture")!)
        let backend = FakeBackend(host: host)
        let cache = QueuePreviewCache()
        for index in 0..<128 {
            _ = try await cache.preview(for: .init(host: host, instance: "one", job: "job-\(index)"), backend: backend)
        }
        _ = try await cache.preview(for: .init(host: host, instance: "one", job: "job-0"), backend: backend)
        _ = try await cache.preview(for: .init(host: host, instance: "one", job: "job-128"), backend: backend)
        _ = try await cache.preview(for: .init(host: host, instance: "one", job: "job-0"), backend: backend)
        #expect(backend.callCount("queueInputs") == 129)
        _ = try await cache.preview(for: .init(host: host, instance: "one", job: "job-1"), backend: backend)
        #expect(backend.callCount("queueInputs") == 130)
    }
    @Test func recycledRowsShareDecodedPreviewAndMissingResults() async throws {
        let host = MoldHost(name: "fixture", baseURL: URL(string: "http://fixture")!)
        let backend = FakeBackend(host: host)
        let cache = QueuePreviewCache()
        let key = QueuePreviewCache.Key(host: host, instance: "one", job: "job")
        _ = try await cache.preview(for: key, backend: backend)
        _ = try await cache.preview(for: key, backend: backend)
        #expect(backend.callCount("queueInputThumbnail") == 1)
        let replacement = QueuePreviewCache.Key(host: host, instance: "two", job: "job")
        _ = try await cache.preview(for: replacement, backend: backend)
        #expect(backend.callCount("queueInputThumbnail") == 2)
    }

    @Test func simultaneousRowsShareRequest() async throws {
        let host = MoldHost(name: "fixture", baseURL: URL(string: "http://fixture")!)
        let backend = FakeBackend(host: host)
        backend.queueThumbnailPending = true
        let cache = QueuePreviewCache()
        let key = QueuePreviewCache.Key(host: host, instance: "one", job: "job")
        let first = Task { try await cache.preview(for: key, backend: backend) }
        let second = Task { try await cache.preview(for: key, backend: backend) }
        await settle { backend.callCount("queueInputThumbnail") > 0 }
        backend.queueThumbnailPending = false
        _ = try await first.value
        _ = try await second.value
        #expect(backend.callCount("queueInputThumbnail") == 1)
    }

    @Test func expiredMissingPreviewIsRetried() async throws {
        let host = MoldHost(name: "fixture", baseURL: URL(string: "http://fixture")!)
        let backend = FakeBackend(host: host)
        let cache = QueuePreviewCache(lifetime: 0)
        let key = QueuePreviewCache.Key(host: host, instance: "one", job: "job")
        _ = try await cache.preview(for: key, backend: backend)
        _ = try await cache.preview(for: key, backend: backend)
        #expect(backend.callCount("queueInputThumbnail") == 2)
    }
    @Test func fastScrollingBoundsRequestsAndCancelsOffscreenWaiters() async throws {
        let host = MoldHost(name: "fixture", baseURL: URL(string: "http://fixture")!)
        let backend = FakeBackend(host: host)
        backend.queueThumbnailPending = true
        let cache = QueuePreviewCache()
        let visible = (0..<4).map { index in
            Task { try await cache.preview(for: .init(host: host, instance: "one", job: "job-\(index)"), backend: backend) }
        }
        await settle { backend.callCount("queueInputThumbnail") == 4 }
        let offscreen = Task { try await cache.preview(for: .init(host: host, instance: "one", job: "offscreen"), backend: backend) }
        await Task.yield()
        offscreen.cancel()
        do { _ = try await offscreen.value; Issue.record("Offscreen waiter did not cancel") }
        catch is CancellationError {} catch { Issue.record("Unexpected error: \(error)") }
        #expect(backend.callCount("queueInputThumbnail") == 4)
        backend.queueThumbnailPending = false
        for task in visible { _ = try await task.value }
        #expect(backend.callCount("queueInputThumbnail") == 4)
    }

}
