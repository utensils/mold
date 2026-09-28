import Foundation
import MoldClientTesting
import Testing

@testable import MoldClient

/// The fake both apps' store tests stand on: it must refuse what nobody
/// stubbed, answer what was stubbed, and remember every call.
struct FakeBackendTests {
    @Test func anUnstubbedRouteThrowsItsSelector() async {
        let fake = FakeBackend()
        await #expect(throws: FakeBackendError.unstubbed("status()")) {
            _ = try await fake.status()
        }
    }

    @Test func aStubbedRouteAnswersAndIsRecorded() async throws {
        let fake = FakeBackend()
        fake.stub("cancelJob(id:)") { _ in () }
        fake.stub("reorderJob(id:position:)") { _ in () }
        try await fake.cancelJob(id: "j1")
        try await fake.reorderJob(id: "j2", position: 3)
        #expect(fake.calls.map(\.route) == ["cancelJob(id:)", "reorderJob(id:position:)"])
        #expect(fake.calls[1].arguments.first as? String == "j2")
        #expect(fake.calls[1].arguments.last as? Int == 3)
    }

    @Test func overloadsAreDistinctRoutes() async throws {
        let fake = FakeBackend()
        fake.stub("trash(_:)") { _ in () }
        try await fake.trash(["a.png"])
        #expect(fake.count("trash(_:)") == 1)
        await #expect(throws: FakeBackendError.unstubbed("restoreFromTrash(_:)")) {
            try await fake.restoreFromTrash(["a.png"])
        }
    }

    @Test func aStubOfTheWrongTypeSaysSo() async {
        let fake = FakeBackend()
        fake.stub("peers()", returning: "not peers")
        await #expect(throws: FakeBackendError.wrongType("peers()", expected: "Array<DiscoveryPeer>")) {
            _ = try await fake.peers()
        }
    }

    @Test func anUnstubbedStreamEndsEmpty() async throws {
        let fake = FakeBackend()
        var seen = 0
        for try await _ in fake.events() { seen += 1 }
        #expect(seen == 0)
        #expect(fake.count("events()") == 1)
    }

    @Test func aStubbedStreamDeliversWhatItWasGiven() async throws {
        let fake = FakeBackend()
        fake.stubStream("downloadEvents()") { _ -> AsyncThrowingStream<DownloadEvent, Error> in
            AsyncThrowingStream { $0.finish(throwing: MoldClientError.unauthorized) }
        }
        await #expect(throws: MoldClientError.self) {
            for try await _ in fake.downloadEvents() {}
        }
    }
}
