import AppKit
import Foundation
import MoldClient
import Testing

@testable import Mold

/// Thumbnails: the one route in this app that cannot be ticketed.
///
/// **Fails today**: `ThumbnailCache` builds its own `URLSession` in `init`
/// with nothing to stub, so the `X-Api-Key`-on-thumbnails rule the README
/// calls out has never been pinned by anything, and the cache has no test at
/// all.
@MainActor
struct ThumbnailCacheTests {
    private func host(_ name: String, key: String?) -> MoldHost {
        MoldHost(id: UUID(), name: name, baseURL: URL(string: "http://\(name):7680")!,
                 apiKey: key)
    }

    private func entry(_ filename: String, on host: MoldHost) -> LibraryEntry {
        LibraryEntry(host: host, print: FakeFixtures.print(filename))
    }

    private func cache() -> ThumbnailCache {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubProtocol.self]
        return ThumbnailCache(session: URLSession(configuration: configuration))
    }

    @Test func aKeyedMachineGetsTheKeyOnTheThumbnailItself() async {
        StubProtocol.reset()
        let machine = host("plato", key: "secret")

        _ = await cache().image(for: entry("robot.png", on: machine), host: machine, size: 256)

        #expect(StubProtocol.seen.first?.value(forHTTPHeaderField: "X-Api-Key") == "secret")
        #expect(StubProtocol.seen.first?.url?.absoluteString
            == "http://plato:7680/api/gallery/thumbnail/robot.png?size=256")
    }

    /// A keyless host is open by policy. Sending no key is the correct request
    /// there, not a degraded one.
    @Test func aKeylessMachineIsAskedWithoutOne() async {
        StubProtocol.reset()
        let machine = host("hal9000", key: nil)

        _ = await cache().image(for: entry("robot.png", on: machine), host: machine, size: 256)

        #expect(StubProtocol.seen.first?.value(forHTTPHeaderField: "X-Api-Key") == nil)
    }

    @Test func anEmptyKeyIsNotAKey() async {
        StubProtocol.reset()
        let machine = host("plato", key: "")

        _ = await cache().image(for: entry("robot.png", on: machine), host: machine, size: 256)

        #expect(StubProtocol.seen.first?.value(forHTTPHeaderField: "X-Api-Key") == nil)
    }

    @Test func aSecondAskForTheSamePictureIsAnsweredFromMemory() async {
        StubProtocol.reset()
        let machine = host("plato", key: nil)
        let cache = cache()
        let row = entry("robot.png", on: machine)

        _ = await cache.image(for: row, host: machine, size: 256)
        _ = await cache.image(for: row, host: machine, size: 256)

        #expect(StubProtocol.seen.count == 1)
    }

    /// The README's promise -- capped, and emptied when Mold quits -- has to be
    /// true of this cache too.
    @Test func purgingEmptiesWhatWasRemembered() async {
        StubProtocol.reset()
        let machine = host("plato", key: nil)
        let cache = cache()
        let row = entry("robot.png", on: machine)

        _ = await cache.image(for: row, host: machine, size: 256)
        cache.purge()
        _ = await cache.image(for: row, host: machine, size: 256)

        #expect(StubProtocol.seen.count == 2)
    }
}

/// Answers every request with a 1×1 PNG and records what it was asked.
private nonisolated final class StubProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) private(set) static var seen: [URLRequest] = []
    private static let lock = NSLock()

    static func reset() { lock.withLock { seen = [] } }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.lock.withLock { Self.seen.append(request) }
        let image = NSImage(size: NSSize(width: 1, height: 1))
        image.lockFocus()
        image.unlockFocus()
        let data = image.tiffRepresentation ?? Data()
        let response = HTTPURLResponse(url: request.url!, statusCode: 200,
                                       httpVersion: nil, headerFields: nil)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}
