import Foundation
import Testing
import MoldClientTesting
@testable import MoldClient

struct InputMediaPreviewTests {
    @Test func prefersAuthoringOriginalWithoutChangingSubmission() {
        let submitted = "fitted"
        #expect(InputMediaPreview.image(original: "original", submitted: submitted) == "original")
        #expect(InputMediaPreview.image(original: nil, submitted: submitted) == submitted)
        #expect(submitted == "fitted")
    }
    @Test func stagesExactBytesAndReleasesPrivateFiles() async throws {
        let bytes = Data([1, 2, 3, 4])
        var file: SourceMediaFile? = try await InputMediaPreview.file(base64: bytes.base64EncodedString())
        let url = try #require(file?.url)
        #expect(try Data(contentsOf: url) == bytes)
        file = nil
        #expect(!FileManager.default.fileExists(atPath: url.path))
    }
    @Test func malformedEncodingRefuses() async {
        await #expect(throws: (any Error).self) { try await InputMediaPreview.file(base64: "!bad") }
    }
}


extension InputMediaPreviewTests {
    @Test func exactOrderedReferenceMemberIsVerifiedWithoutHydratingDraft() async throws {
        let refs = ["first", "second"].map {
            GenerationReference(kind: "image", media: .init(authority: "descriptor"), mimeType: "image/png", provenance: .init(name: $0))
        }
        let members = ["m-first", "m-second"].map {
            RetainedSourceMedia.Member(memberId: $0, role: "references", displayName: $0, sizeBytes: 4)
        }
        let backend = FakeBackend()
        backend.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: members))
        let file = try await InputMediaPreview.file(base64: Data([1, 2, 3, 4]).base64EncodedString())
        backend.stub("retainedSourceMediaFile(for:member:)", returning: file)
        let loaded = try await InputMediaPreview.retainedFile(reference: refs[1], index: 1, references: refs,
            original: refs, members: members, filename: "owner.png", backend: backend)
        #expect(loaded.url == file.url)
        #expect(backend.calls.last?.arguments[1] as? RetainedSourceMedia.Member == members[1])
        #expect(backend.calls.map(\.route) == ["retainedSourceMedia(for:)", "retainedSourceMediaFile(for:member:)"])
        backend.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: [members[0]]))
        await #expect(throws: MoldClientError.self) {
            try await InputMediaPreview.retainedFile(reference: refs[1], index: 1, references: refs,
                original: refs, members: members, filename: "owner.png", backend: backend)
        }
        #expect(backend.count("retainedSourceMediaFile(for:member:)") == 1)
    }
    @Test func cancelledStagingNeverPublishesFile() async throws {
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await InputMediaPreview.file(base64: Data([1]).base64EncodedString())
        }
        await #expect(throws: CancellationError.self) { try await task.value }
    }
}

extension InputMediaPreviewTests {
    @Test func originalDecodeHasExplicitPixelBudgetBeforeAllocation() throws {
        try SourceMediaOriginal.checkDecodedDimensions(width: 5_000, height: 1)
        #expect(throws: SourceMediaOriginal.InspectionError.imageTooLarge) {
            try SourceMediaOriginal.checkDecodedDimensions(width: 8_192, height: 4_097)
        }
    }
    @Test func pickerAuthorityRevokesOnCredentialRouteAndInstanceChanges() {
        let host = MoldHost(name: "Fixture", baseURL: URL(string: "http://fixture")!, apiKey: "first")
        let authority = InputMediaPreviewAuthority(host: host, instance: "one")
        #expect(authority.isCurrent(host: host, instance: "one"))
        var changed = host
        changed.apiKey = "second"
        #expect(!authority.isCurrent(host: changed, instance: "one"))
        changed = host
        changed.baseURL = URL(string: "http://replacement")!
        #expect(!authority.isCurrent(host: changed, instance: "one"))
        #expect(!authority.isCurrent(host: host, instance: "two"))
        #expect(!authority.isCurrent(host: nil, instance: "one"))
    }
}
