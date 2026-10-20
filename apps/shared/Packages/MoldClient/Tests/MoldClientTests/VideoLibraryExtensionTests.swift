import Foundation
import Testing
import MoldClientTesting
@testable import MoldClient

@MainActor struct VideoLibraryExtensionTests {
    @Test func legacyInventoryCanOpenWithoutInventingHashButAuthAndCorruptionRefuse() async throws {
        let backend = FakeBackend()
        let owner = VideoJobOwner(host: backend.host, instanceId: "installation")
        let metadata = try MoldJSON.decoder.decode(OutputMetadata.self, from: Data(#"{"model":"video","prompt":"literal\\new","width":768,"height":768}"#.utf8))
        backend.stub("status()", returning: try MoldJSON.decoder.decode(ServerStatus.self,
            from: Data(#"{"version":"test","busy":false,"uptime_secs":1,"instance_id":"installation"}"#.utf8)))
        backend.stub("capabilities()", returning: try MoldJSON.decoder.decode(Capabilities.self,
            from: Data(#"{"video_jobs":{"supported":true,"literal_prompts":true,"library_extend":true}}"#.utf8)))
        backend.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .unavailableLegacy))
        let source = try await VideoLibraryExtensionSource.inspect(filename: "ordinary.mp4", title: "Original", metadata: metadata,
            owner: owner, backend: backend, current: { true })
        #expect(source.source.expectedArchiveIdentitySha256 == nil)
        #expect(source.metadata.prompt == metadata.prompt)
        #expect(backend.count("retainedMediaTransferOffer(for:)") == 0)
        for availability: RetainedSourceMedia.Availability in [.unavailableAuth, .unavailableMissingOrCorrupt, .unknown] {
            backend.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: availability))
            await #expect(throws: (any Error).self) {
                try await VideoLibraryExtensionSource.inspect(filename: "ordinary.mp4", title: "Original", metadata: metadata,
                    owner: owner, backend: backend, current: { true })
            }
        }
    }
    @Test func exactArchiveOfferBindsHashAndKeepsTypedInventorySeparateFromEnvelopeDigest() async throws {
        let backend = FakeBackend()
        let owner = VideoJobOwner(host: backend.host, instanceId: "installation")
        let metadata = try MoldJSON.decoder.decode(OutputMetadata.self, from: Data(#"{"model":"video","prompt":"literal\\new","seed":18446744073709551615,"frames":1000,"fps":24,"width":768,"height":768}"#.utf8))
        backend.stub("status()", returning: try MoldJSON.decoder.decode(ServerStatus.self,
            from: Data(#"{"version":"test","busy":false,"uptime_secs":1,"instance_id":"installation"}"#.utf8)))
        backend.stub("capabilities()", returning: try MoldJSON.decoder.decode(Capabilities.self,
            from: Data(#"{"video_jobs":{"supported":true,"literal_prompts":true,"library_extend":true}}"#.utf8)))
        let member = RetainedSourceMedia.Member(memberId: "input", role: "references", displayName: "hero.png", sizeBytes: 50,
            mediaKind: "image", contentSha256: String(repeating: "a", count: 64))
        backend.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: [member]))
        let hash = String(repeating: "c", count: 64)
        backend.stub("retainedMediaTransferOffer(for:)", returning: RetainedSourceMedia.TransferOffer(archiveIdentitySha256: hash,
            members: [.init(memberId: "input", role: "references", position: "item:0", sizeBytes: 50, sha256: String(repeating: "b", count: 64))],
            outputSha256: String(repeating: "d", count: 64), outputSizeBytes: 100, metadata: metadata))
        let source = try await VideoLibraryExtensionSource.inspect(filename: "ordinary.mp4", title: "Original", metadata: metadata,
            owner: owner, backend: backend, current: { true })
        #expect(source.source.expectedArchiveIdentitySha256 == hash)
        #expect(source.inventory.members == [member])
        #expect(source.prefilledDraft.seed == UInt64.max)
        #expect(source.prefilledDraft.media.sourceImage == nil)
        #expect(source.prefilledDraft.media.keyframes.isEmpty)
        #expect(source.authoring(seconds: "5").draft.seconds == "5")
        #expect(source.authoring(seconds: "5").sourceOwner?.matches(owner) == true)
        #expect(backend.count("retainedSourceMediaBytes(for:member:)") == 0)
        await #expect(throws: (any Error).self) {
            try await VideoLibraryExtensionSource.inspect(filename: "ordinary.mp4", title: "Original", metadata: metadata,
                owner: owner, backend: backend, current: { false })
        }
    }

    @Test func originalReferenceCardsUseOnlyFinalAuthoredSceneAndPreserveOrder() throws {
        let metadata = try MoldJSON.decoder.decode(OutputMetadata.self, from: Data(#"{"model":"video","chain":{"stages":[{"generation_references":[{"kind":"image","sha256":"early"}]},{"generation_references":[{"kind":"audio","sha256":"last-a"},{"kind":"image","sha256":"last-b"}]}]}}"#.utf8))
        let members = [
            RetainedSourceMedia.Member(memberId: "early", role: "stage_reference:0:0", displayName: "early", sizeBytes: 1, contentSha256: "early"),
            RetainedSourceMedia.Member(memberId: "second", role: "stage_reference:1:1", displayName: "second", sizeBytes: 1, contentSha256: "last-b"),
            RetainedSourceMedia.Member(memberId: "first", role: "stage_reference:1:0", displayName: "first", sizeBytes: 1, contentSha256: "last-a")]
        let cards = VideoOriginalReference.ordered(metadata: metadata, inventory: .init(availability: .available, members: members))
        #expect(cards.map(\.member?.memberId) == ["first", "second"])
        #expect(cards.map(\.descriptor.kind) == ["audio", "image"])
        let missing = VideoOriginalReference.ordered(metadata: metadata, inventory: .init(availability: .unavailableLegacy))
        #expect(missing.count == 2 && missing.allSatisfy { $0.member == nil })
    }

    @Test func coldRestoreRefusesMissingOriginalGalleryPrintBeforeInspectingPrivateSources() async throws {
        let backend = FakeBackend()
        let owner = VideoJobOwner(host: backend.host, instanceId: "installation")
        let authoring = VideoDurationAuthoring(draft: .init(seconds: "5", source: .init(filename: "original.mp4")),
            sourceOwner: .init(owner: owner, title: "Original"))
        let different = try MoldJSON.decoder.decode(GalleryPrint.self,
            from: Data(#"{"filename":"different.mp4","format":"mp4","metadata":{"model":"video"},"timestamp":1}"#.utf8))
        backend.stub("gallery(etag:)", returning: Fetched<[GalleryPrint]>.fresh([different], etag: nil))
        backend.stub("status()", returning: try MoldJSON.decoder.decode(ServerStatus.self,
            from: Data(#"{"version":"test","busy":false,"uptime_secs":1,"instance_id":"installation"}"#.utf8)))
        backend.stub("capabilities()", returning: try MoldJSON.decoder.decode(Capabilities.self,
            from: Data(#"{"video_jobs":{"supported":true,"literal_prompts":true,"library_extend":true}}"#.utf8)))
        backend.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .unavailableLegacy))
        await #expect(throws: (any Error).self) {
            try await VideoLibraryExtensionSource.restore(authoring, owner: owner, backend: backend, current: { true })
        }
        #expect(backend.count("retainedSourceMedia(for:)") == 0)
    }

}
