import Foundation
import Testing
import MoldClientTesting
@testable import MoldClient

@MainActor struct SourceMediaAspectsTests {
    @Test func measurementsAreBatchedAndUnchangedPixelsDoNotReflow() {
        let cache = SourceMediaAspects(), key = key("v1")
        #expect(cache.aspect(for: "source", key: key) == 1)
        cache.record(id: "source", key: key, aspect: 0.5)
        cache.record(id: "wide", key: key, aspect: 2)
        #expect(cache.revision == 0)
        cache.flush()
        #expect(cache.revision == 1)
        #expect(cache.aspect(for: "source", key: key) == 0.5)
        cache.record(id: "source", key: key, aspect: 0.5)
        cache.flush()
        #expect(cache.revision == 1)
    }
    @Test func changedContentNeverInheritsOldGeometry() {
        let cache = SourceMediaAspects(), first = key("v1"), second = key("v2")
        cache.record(id: "source", key: first, aspect: 0.5); cache.flush()
        #expect(cache.aspect(for: "source", key: second) == 1)
        cache.record(id: "source", key: second, aspect: 2); cache.flush()
        #expect(cache.aspect(for: "source", key: second) == 2)
        #expect(cache.aspect(for: "source", key: first) == 1)
        cache.record(id: "bad", key: first, aspect: .infinity); cache.flush()
        #expect(cache.aspect(for: "bad", key: first) == 1)
    }
    @Test func identicalVerifiedBytesKeepTheirRatioAcrossHealthyCopies() {
        let cache = SourceMediaAspects(), digest = String(repeating: "a", count: 64)
        let member = RetainedSourceMedia.Member(memberId: "input", role: "source_image", displayName: "input.png", sizeBytes: 1, contentSha256: digest)
        let first = SourcePreviewKey(authority: key("first").authority, member: member)
        let second = SourcePreviewKey(authority: key("second").authority, member: member)
        cache.record(id: "sha256/" + digest, key: first, aspect: 0.5); cache.flush()
        #expect(cache.aspect(for: "sha256/" + digest, key: second) == 0.5)
        cache.record(id: "sha256/" + digest, key: second, aspect: 0.5); cache.flush()
        #expect(cache.revision == 1)
    }
    private func key(_ version: String) -> SourcePreviewKey {
        let owner = PrintFixtures.entry("source.png", host: UUID())
        return .init(authority: .init(owner: owner, route: nil, instance: version),
                     member: .init(memberId: "input", role: "source_image", displayName: "input.png", sizeBytes: 1))
    }
}
