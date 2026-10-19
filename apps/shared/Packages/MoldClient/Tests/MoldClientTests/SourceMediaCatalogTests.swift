import Foundation
import Testing
import MoldClientTesting
@testable import MoldClient

@MainActor struct SourceMediaCatalogTests {
    @Test func readsBeyondOnePageAndKeepsOwnerScopedIdentities() async {
        let host = UUID(), backend = FakeBackend()
        backend.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: [
            .init(memberId: "same-member", role: "source_image", displayName: "photo.png", sizeBytes: 5)
        ]))
        let owners = (0..<127).map { PrintFixtures.entry("print-\($0).png", host: host) }
        let catalog = SourceMediaCatalog()
        await catalog.load(owners: owners, backend: { _ in backend })
        #expect(catalog.rows.count == 127)
        #expect(Set(catalog.rows.map(\.id)).count == 127)
        #expect(catalog.completed == 127)
        #expect(catalog.failures.isEmpty)
    }
    @Test func absentHostIsUnavailableAndRefreshDropsOldRows() async {
        let owner = PrintFixtures.entry("owner.png", host: UUID())
        let catalog = SourceMediaCatalog()
        await catalog.load(owners: [owner], backend: { _ in nil })
        #expect(catalog.failures[owner.id] != nil)
        await catalog.load(owners: [], backend: { _ in nil })
        #expect(catalog.failures.isEmpty && catalog.rows.isEmpty)
    }
}

private final class SourceInventoryStub: URLProtocol {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let body = Data(#"{"instance_id":"restarted","inventories":{"owner.png":{"availability":"available","members":[{"member_id":"member","role":"source_image","display_name":"image.png","size_bytes":1}]}}}"#.utf8)
        client?.urlProtocol(self, didReceive: HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: ["Content-Type": "application/json"])!, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

extension SourceMediaCatalogTests {
    @Test func restartedBatchCannotPublishRowsUnderOldInstance() async {
        let host = MoldHost(name: "Fixture", baseURL: URL(string: "http://fixture")!)
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [SourceInventoryStub.self]
        let backend = HTTPBackend.testing(host: host, session: URLSession(configuration: configuration))
        let owner = PrintFixtures.entry("owner.png", host: host.id)
        let catalog = SourceMediaCatalog()
        await catalog.load(owners: [owner], backend: { _ in backend }, route: { _ in host }, instance: { _ in "original" })
        #expect(catalog.rows.isEmpty)
        #expect(catalog.failures[owner.id] != nil)
    }

    @Test func changedCredentialCannotPublishCompletedInventory() async {
        let backend = FakeBackend()
        backend.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: [
            .init(memberId: "one", role: "source_image", displayName: "image.png", sizeBytes: 1)
        ]))
        let host = MoldHost(name: "Fixture", baseURL: URL(string: "http://fixture")!, apiKey: "first")
        var changed = host; changed.apiKey = "second"
        let owner = PrintFixtures.entry("owner.png", host: host.id)
        var calls = 0
        let catalog = SourceMediaCatalog()
        await catalog.load(owners: [owner], backend: { _ in backend }, route: { _ in
            calls += 1
            return calls == 1 ? host : changed
        })
        #expect(catalog.rows.isEmpty)
        #expect(catalog.failures[owner.id] != nil)
    }
}

extension SourceMediaCatalogTests {
    @Test func appendAndReorderReuseFreshInventories() async {
        let host = UUID(), backend = FakeBackend()
        backend.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: [
            .init(memberId: "source", role: "source_image", displayName: "photo.png", sizeBytes: 5)
        ]))
        let owners = (0..<100).map { PrintFixtures.entry("print-\($0).png", host: host) }
        let catalog = SourceMediaCatalog()
        await catalog.load(owners: owners, backend: { _ in backend })
        let firstID = catalog.rows.first!.id
        await catalog.load(owners: owners.reversed(), backend: { _ in backend })
        #expect(backend.count("retainedSourceMedia(for:)") == 100)
        #expect(catalog.rows.last!.id == firstID)
        await catalog.load(owners: owners + [PrintFixtures.entry("new.png", host: host)], backend: { _ in backend })
        #expect(backend.count("retainedSourceMedia(for:)") == 101)
    }
}

extension SourceMediaCatalogTests {
    @Test func verifiedBytesDeduplicateAcrossRolesAndKeepEveryUsage() async {
        let backend = FakeBackend(), host = UUID()
        backend.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: [
            .init(memberId: "first", role: "source_image", displayName: "same.png", sizeBytes: 5, contentSha256: String(repeating: "a", count: 64)),
            .init(memberId: "end", role: "last_frame", displayName: "same.png", sizeBytes: 5, contentSha256: String(repeating: "a", count: 64)),
            .init(memberId: "unknown", role: "source_image", displayName: "same.png", sizeBytes: 5)
        ]))
        let catalog = SourceMediaCatalog()
        let owners = (0..<2).map { PrintFixtures.entry("owner-\($0).png", host: host) }
        await catalog.load(owners: owners, backend: { _ in backend })
        #expect(catalog.sources.count == 3)
        #expect(catalog.sources.first?.usages.count == 4)
        let identity = catalog.sources.first?.id
        await catalog.load(owners: [owners[1]], backend: { _ in backend })
        #expect(catalog.sources.first?.id == identity)
        #expect(catalog.sources.first?.usages.count == 2)
        #expect(catalog.rows.allSatisfy { $0.owner.id == owners[1].id })
        await catalog.load(owners: [owners[1]], backend: { _ in backend }, refresh: true)
        #expect(backend.count("retainedSourceMedia(for:)") == 3)
    }
}

extension SourceMediaCatalogTests {
    @Test func freshnessExpiresAndRevokedOwnerIsNotResurrected() async {
        let backend = FakeBackend(), host = UUID(), now = Date()
        backend.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: [
            .init(memberId: "source", role: "source_image", displayName: "photo.png", sizeBytes: 5)
        ]))
        let owner = PrintFixtures.entry("owner.png", host: host), catalog = SourceMediaCatalog()
        await catalog.load(owners: [owner], backend: { _ in backend }, now: now)
        await catalog.load(owners: [owner], backend: { _ in backend }, now: now.addingTimeInterval(59))
        #expect(backend.count("retainedSourceMedia(for:)") == 1)
        await catalog.load(owners: [owner], backend: { _ in backend }, now: now.addingTimeInterval(61))
        #expect(backend.count("retainedSourceMedia(for:)") == 2)
        catalog.retain(authorities: [])
        await catalog.load(owners: [], backend: { _ in backend }, now: now)
        #expect(catalog.sources.isEmpty)
        await catalog.load(owners: [owner], backend: { _ in backend }, now: now)
        #expect(backend.count("retainedSourceMedia(for:)") == 3)
    }
}

extension SourceMediaCatalogTests {
    @Test func thousandOwnersThreeThousandUsesProduceThreeCards() async {
        let backend = FakeBackend(), host = UUID()
        backend.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: (0..<3).map {
            .init(memberId: "member-\($0)", role: "source_image", displayName: "image-\($0).png", sizeBytes: 5, contentSha256: String(repeating: "\($0)", count: 64))
        }))
        let owners = (0..<1_000).map { PrintFixtures.entry("owner-\($0).png", host: host) }
        let catalog = SourceMediaCatalog()
        let began = ContinuousClock.now
        await catalog.load(owners: owners, backend: { _ in backend })
        #expect(catalog.rows.count == 3_000)
        #expect(catalog.sources.count == 3)
        #expect(catalog.sources.allSatisfy { $0.usages.count == 1_000 })
        print("Native fixture: 1000 owners / 3000 uses / 3 cards, cold scan \(ContinuousClock.now - began)")
        let warm = ContinuousClock.now
        await catalog.load(owners: owners.reversed(), backend: { _ in backend })
        #expect(backend.count("retainedSourceMedia(for:)") == 1_000)
        print("Native fixture warm reorder: \(ContinuousClock.now - warm), 0 additional requests")
    }
}

extension SourceMediaCatalogTests {
    @Test func namedGalleryInvalidationOnlyRefetchesOneOwner() async {
        let backend = FakeBackend(), host = UUID()
        backend.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: []))
        let owners = (0..<100).map { PrintFixtures.entry("owner-\($0).png", host: host) }
        let catalog = SourceMediaCatalog()
        await catalog.load(owners: owners, backend: { _ in backend })
        catalog.invalidate(host: host, filename: owners[0].print.filename)
        await catalog.load(owners: owners, backend: { _ in backend })
        #expect(backend.count("retainedSourceMedia(for:)") == 101)
    }
    @Test func galleryInvalidationRevokesFreshAnswersImmediately() async {
        let backend = FakeBackend(), host = UUID()
        backend.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: [
            .init(memberId: "one", role: "source_image", displayName: "image.png", sizeBytes: 1)
        ]))
        let owner = PrintFixtures.entry("owner.png", host: host), catalog = SourceMediaCatalog()
        await catalog.load(owners: [owner], backend: { _ in backend })
        #expect(catalog.rows.count == 1)
        catalog.invalidate(host: host)
        #expect(catalog.rows.isEmpty)
        #expect(catalog.revision == 1)
        await catalog.load(owners: [owner], backend: { _ in backend })
        #expect(catalog.rows.count == 1)
    }
}

private actor CatalogConcurrencyProbe {
    var active = 0
    var peak = 0
    func read() async -> RetainedSourceMedia.Inventory {
        active += 1; peak = max(peak, active)
        try? await Task.sleep(for: .milliseconds(2))
        active -= 1
        return .init(availability: .available, members: [])
    }
}

extension SourceMediaCatalogTests {
    @Test func coldReadsAreConcurrentButBoundedAndWarmReadsAreZero() async {
        let probe = CatalogConcurrencyProbe(), backend = FakeBackend(), host = UUID()
        backend.stub("retainedSourceMedia(for:)") { _ in await probe.read() }
        let owners = (0..<205).map { PrintFixtures.entry("bounded-\($0).png", host: host) }
        let catalog = SourceMediaCatalog()
        await catalog.load(owners: owners, backend: { _ in backend })
        #expect(await probe.peak == 2)
        #expect(catalog.completed == 205)
        await catalog.load(owners: owners, backend: { _ in backend })
        #expect(backend.count("retainedSourceMedia(for:)") == 205)
    }
}

extension SourceMediaCatalogTests {
    @Test func thousandOwnerPlanUsesElevenBoundedHTTPRequests() {
        let host = UUID()
        let owners = (0..<1_000).map { PrintFixtures.entry("bulk-\($0).png", host: host) }
        let batches = SourceMediaCatalog.batches(owners)
        #expect(batches.count == 11) // Previously 41 sequential requests.
        #expect(batches.first?.count == 4)
        #expect(batches.allSatisfy { $0.count <= 100 })
        #expect(batches.flatMap { $0 }.map(\.id) == owners.map(\.id))
        let other = PrintFixtures.entry("other.png", host: UUID())
        let mixed = SourceMediaCatalog.batches(owners + [other])
        #expect(mixed[1].first?.id == other.id) // A slow first host cannot occupy both bootstrap slots.
    }

    @Test func revokedAuthorityCannotReturnFromConcurrentReads() async {
        let probe = CatalogConcurrencyProbe(), backend = FakeBackend(), host = UUID()
        backend.stub("retainedSourceMedia(for:)") { _ in
            _ = await probe.read()
            return RetainedSourceMedia.Inventory(availability: .available, members: [
                .init(memberId: "one", role: "source_image", displayName: "image.png", sizeBytes: 1)
            ])
        }
        let owners = (0..<205).map { PrintFixtures.entry("revoked-\($0).png", host: host) }
        let catalog = SourceMediaCatalog()
        let task = Task { await catalog.load(owners: owners, backend: { _ in backend }) }
        while await probe.active == 0 { await Task.yield() }
        catalog.retain(authorities: [])
        await task.value
        #expect(catalog.rows.isEmpty && catalog.sources.isEmpty)
        #expect(!catalog.loading)
        #expect(backend.count("retainedSourceMedia(for:)") < owners.count)
    }

    @Test func failedExpiredRefreshDropsPreviouslyVisibleMedia() async {
        let backend = FakeBackend(), host = UUID()
        backend.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: [
            .init(memberId: "one", role: "source_image", displayName: "image.png", sizeBytes: 1)
        ]))
        let catalog = SourceMediaCatalog(), owner = PrintFixtures.entry("expired.png", host: host), now = Date()
        await catalog.load(owners: [owner], backend: { _ in backend }, now: now)
        #expect(catalog.rows.count == 1)
        backend.stub("retainedSourceMedia(for:)") { _ in throw MoldClientError.unreachable("offline") }
        await catalog.load(owners: [owner], backend: { _ in backend }, now: now.addingTimeInterval(61))
        #expect(catalog.rows.isEmpty && catalog.sources.isEmpty)
        #expect(catalog.failures[owner.id] != nil)
    }
}

extension SourceMediaCatalogTests {
    @Test func originalRevisionChangesOnlyForItsOwner() {
        let host = UUID(), otherHost = UUID()
        let first = PrintID(host: host, filename: "first.png")
        let second = PrintID(host: host, filename: "second.png")
        let elsewhere = PrintID(host: otherHost, filename: "first.png")
        let catalog = SourceMediaCatalog()
        let before = catalog.contentRevision(for: first)
        catalog.invalidate(host: host, filename: "second.png")
        #expect(catalog.contentRevision(for: first) == before)
        #expect(catalog.contentRevision(for: second) != before)
        catalog.invalidate(host: host)
        #expect(catalog.contentRevision(for: first) != before)
        #expect(catalog.contentRevision(for: elsewhere) == before)
    }
}

extension SourceMediaCatalogTests {
    @Test func expiredSnapshotRemainsVisibleDuringRefreshButRefusalRemovesIt() async {
        let backend = FakeBackend(), host = UUID(), now = Date()
        let inventory = RetainedSourceMedia.Inventory(availability: .available, members: [
            .init(memberId: "source", role: "source_image", displayName: "photo.png", sizeBytes: 5)
        ])
        backend.stub("retainedSourceMedia(for:)", returning: inventory)
        let owner = PrintFixtures.entry("owner.png", host: host), catalog = SourceMediaCatalog()
        await catalog.load(owners: [owner], backend: { _ in backend }, now: now)
        backend.stub("retainedSourceMedia(for:)") { _ in
            await MainActor.run { #expect(catalog.sources.count == 1) }
            return inventory
        }
        await catalog.load(owners: [owner], backend: { _ in backend }, now: now.addingTimeInterval(61))
        #expect(catalog.sources.count == 1)
        backend.stub("retainedSourceMedia(for:)", throwing: MoldClientError.unauthorized)
        await catalog.load(owners: [owner], backend: { _ in backend }, now: now.addingTimeInterval(122))
        #expect(catalog.sources.isEmpty)
        #expect(catalog.failures[owner.id] != nil)
    }
}

extension SourceMediaCatalogTests {
    @Test func refusedRefreshRevokesCapturedMemberAndChangesOwnerRevision() async {
        let backend = FakeBackend(), host = UUID()
        let owner = PrintFixtures.entry("owner.png", host: host)
        backend.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available,
            members: [.init(memberId: "source", role: "source_image", displayName: "photo.png", sizeBytes: 5)]))
        let catalog = SourceMediaCatalog()
        await catalog.load(owners: [owner], backend: { _ in backend })
        let captured = catalog.rows[0], revision = catalog.contentRevision(for: owner.id)
        #expect(catalog.contains(captured))
        backend.stub("retainedSourceMedia(for:)", throwing: MoldClientError.unauthorized)
        await catalog.load(owners: [owner], backend: { _ in backend }, refresh: true)
        #expect(!catalog.contains(captured))
        #expect(catalog.contentRevision(for: owner.id) != revision)
    }
}

extension SourceMediaCatalogTests {
    @Test func noSourceDetailAnswerIsReusedUntilThatOwnerChanges() async throws {
        let backend = FakeBackend()
        backend.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: []))
        let host = MoldHost(name: "Fixture", baseURL: URL(string: "http://fixture")!)
        let owner = PrintFixtures.entry("no-input.png", host: host.id)
        let catalog = SourceMediaCatalog()
        for _ in 0..<10 {
            let inventory = try await catalog.inventory(for: owner, backend: backend, route: host, instance: nil)
            #expect(inventory.members.isEmpty)
        }
        #expect(backend.count("retainedSourceMedia(for:)") == 1)
        catalog.invalidate(host: host.id, filename: "unrelated.png")
        _ = try await catalog.inventory(for: owner, backend: backend, route: host, instance: nil)
        #expect(backend.count("retainedSourceMedia(for:)") == 1)
        catalog.invalidate(host: host.id, filename: owner.print.filename)
        _ = try await catalog.inventory(for: owner, backend: backend, route: host, instance: nil)
        #expect(backend.count("retainedSourceMedia(for:)") == 2)
    }
}

extension SourceMediaCatalogTests {
    @Test func changingFiltersKeepsFreshAuthorizedInventoriesWarm() async {
        let backend = FakeBackend(), host = UUID()
        backend.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: [
            .init(memberId: "source", role: "source_image", displayName: "photo.png", sizeBytes: 5)
        ]))
        let owners = [PrintFixtures.entry("a.png", host: host), PrintFixtures.entry("b.png", host: host)]
        let catalog = SourceMediaCatalog()
        await catalog.load(owners: owners, backend: { _ in backend })
        await catalog.load(owners: [], backend: { _ in backend })
        #expect(catalog.rows.isEmpty)
        await catalog.load(owners: [owners[0]], backend: { _ in backend })
        await catalog.load(owners: owners, backend: { _ in backend })
        #expect(backend.count("retainedSourceMedia(for:)") == 2)
        #expect(catalog.rows.count == 2)
        catalog.retain(authorities: [SourceMediaAuthority(owner: owners[0], route: nil, instance: nil)])
        await catalog.load(owners: owners, backend: { _ in backend })
        #expect(backend.count("retainedSourceMedia(for:)") == 3)
    }
}
