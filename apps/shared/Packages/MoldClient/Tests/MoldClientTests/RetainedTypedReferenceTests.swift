import CryptoKit
import Foundation
import Testing
@testable import MoldClient

@Suite struct RetainedTypedReferenceTests {
    private func descriptor(_ data: Data, name: String) -> GenerationReference {
        GenerationReference(kind: "image", media: .init(authority: "descriptor"), mimeType: "image/png",
            provenance: .init(name: name, sha256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()), width: 1, height: 1)
    }
    private func request() -> GenerateRequest {
        GenerateRequest(prompt: "p", model: "m", width: 256, height: 256, steps: 4, guidance: 0, batchSize: 1)
    }
    @Test func retainedReferencesHydrateInOrderAndKeepDescriptorFacts() throws {
        let a = Data([1]), b = Data([2])
        var request = request()
        request.references = [descriptor(a, name: "First"), descriptor(b, name: "Second")]
        let members = [RetainedSourceMedia.Member(memberId: "a", role: "references", displayName: "A", sizeBytes: 1),
                       RetainedSourceMedia.Member(memberId: "b", role: "references", displayName: "B", sizeBytes: 1)]
        #expect(RetainedSourceMedia.members(members, forHydrating: request) == members)
        let result = try RetainedSourceMedia.relayed([(members[0], a), (members[1], b)], into: request)
        #expect(result.references?.map(\.media.data) == [a.base64EncodedString(), b.base64EncodedString()])
        #expect(result.references?.map(\.name) == ["First", "Second"])
        #expect(result.references?.allSatisfy { $0.media.authority == "inline" } == true)
        #expect(RetainedSourceMedia.members(members, forHydrating: result).isEmpty)
    }
    @Test func retainedDescriptorReuseRequiresUnchangedOrderAndMatchingArchive() {
        let refs = [descriptor(Data([1]), name: "First"), descriptor(Data([2]), name: "Second")]
        let members = [RetainedSourceMedia.Member(memberId: "a", role: "references", displayName: "A", sizeBytes: 1),
                       RetainedSourceMedia.Member(memberId: "b", role: "references", displayName: "B", sizeBytes: 1)]
        #expect(RetainedReferenceGuard.canHydrate(references: refs, original: refs, members: members))
        #expect(!RetainedReferenceGuard.canHydrate(references: refs, original: refs, members: []))
        #expect(!RetainedReferenceGuard.canHydrate(references: refs, original: refs, members: [
            .init(memberId: "a", role: "source_image", displayName: "A", sizeBytes: 1)]))
        #expect(!RetainedReferenceGuard.canHydrate(references: [refs[0]], original: refs, members: members))
        #expect(!RetainedReferenceGuard.canHydrate(references: refs.reversed(), original: refs, members: members))
        var mixed = refs; mixed[0].media = .init(authority: "inline", data: "AQ==")
        #expect(!RetainedReferenceGuard.canHydrate(references: mixed, original: refs, members: members))
        #expect(!RetainedReferenceGuard.canHydrate(references: refs, original: refs, members: [members[0]]))
    }
    @Test func mismatchedCountAndContentAndNewUserInputsAreNeverOverwritten() throws {
        let a = Data([1]), member = RetainedSourceMedia.Member(memberId: "a", role: "references", displayName: "A", sizeBytes: 1)
        var request = request()
        request.references = [descriptor(a, name: "A"), descriptor(a, name: "B")]
        #expect(throws: (any Error).self) { try RetainedSourceMedia.relayed([(member, a)], into: request) }
        request.references = [descriptor(a, name: "A")]
        #expect(throws: (any Error).self) { try RetainedSourceMedia.relayed([(member, Data([9]))], into: request) }
        request.references?[0].media = .init(authority: "inline", data: a.base64EncodedString())
        #expect(RetainedSourceMedia.members([member], forHydrating: request).isEmpty)
        #expect(throws: (any Error).self) { try RetainedSourceMedia.relayed([(member, a)], into: request) }
    }
}
