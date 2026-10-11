import Foundation
import MoldClientTesting
import Testing
@testable import MoldClient

@MainActor struct SourceMediaLoadingTests {
    @Test func revokedResponseAndItsErrorCannotPublish() async throws {
        var current = true
        let value = try await SourceMediaLoading.read(isCurrent: { current }) {
            current = false
            return "obsolete"
        }
        #expect(value == nil)
        current = true
        let failed: String? = try await SourceMediaLoading.read(isCurrent: { current }) {
            current = false
            throw URLError(.timedOut)
        }
        #expect(failed == nil)
    }

    @Test func currentFailureStillReachesTheCaller() async {
        await #expect(throws: URLError.self) {
            try await SourceMediaLoading.read(isCurrent: { true }) { () throws -> String in throw URLError(.timedOut) }
        }
    }

    @Test func cancellationPreventsPublicationEvenWhenTheReadIgnoresIt() async throws {
        let task = Task { try await SourceMediaLoading.read(isCurrent: { true }) {
            withUnsafeCurrentTask { $0?.cancel() }
            return "obsolete"
        } }
        #expect(try await task.value == nil)
    }

    @Test func revokedAuthorityDoesNotStartARead() async throws {
        var reads = 0
        let value = try await SourceMediaLoading.read(isCurrent: { false }) { reads += 1; return 1 }
        #expect(value == nil)
        #expect(reads == 0)
    }

    @Test func filteredSearchCannotRevealExcludedOwner() {
        let host = UUID()
        let visible = PrintFixtures.entry("visible.png", host: host)
        let hidden = PrintFixtures.entry("hidden-secret.png", host: host)
        let member = RetainedSourceMedia.Member(memberId: "m", role: "source_image", displayName: "photo.png", sizeBytes: 1,
                                                contentSha256: String(repeating: "a", count: 64))
        let rows = [visible, hidden].map { SourceMediaCatalog.Row(owner: $0, member: member, route: nil, instance: "first") }
        let sources = SourceMediaCatalog.group(rows)
        let allowed = [visible.id: rows[0].authority]
        #expect(SourceMediaDisplay.filter(sources, allowed: allowed, kind: .all, search: "hidden-secret").isEmpty)
        let display = SourceMediaDisplay.filter(sources, allowed: allowed, kind: .image, search: "PHOTO")
        #expect(display.first?.usages == [rows[0]])
        #expect(SourceMediaDisplay.filter(sources, allowed: allowed, kind: .video, search: "").isEmpty)
        let restarted = SourceMediaAuthority(owner: visible, route: nil, instance: "second")
        #expect(SourceMediaDisplay.filter(sources, allowed: [visible.id: restarted], kind: .all, search: "").isEmpty)
    }

    @Test func originalKeyChangesWithRouteRevisionAndRetry() {
        let host = MoldHost(name: "machine", baseURL: URL(string: "http://fixture")!, apiKey: "first")
        var changed = host; changed.apiKey = "second"
        let owner = PrintFixtures.entry("owner.png", host: host.id)
        let member = RetainedSourceMedia.Member(memberId: "m", role: "source_image", displayName: "photo.png", sizeBytes: 1)
        let row = SourceMediaCatalog.Row(owner: owner, member: member, route: host, instance: "first")
        let key = SourceMediaOriginalKey(row: row, revision: 1, retry: 0)
        #expect(key != SourceMediaOriginalKey(row: row, revision: 2, retry: 0))
        #expect(key != SourceMediaOriginalKey(row: row, revision: 1, retry: 1))
        #expect(key != SourceMediaOriginalKey(row: .init(owner: owner, member: member, route: changed, instance: "first"), revision: 1, retry: 0))
    }
}
