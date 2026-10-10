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
        let backend = HTTPBackend(host: host, session: URLSession(configuration: configuration))
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
