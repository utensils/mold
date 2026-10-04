import Foundation
import CryptoKit
import MoldClientTesting
import Testing
@testable import MoldClient

struct RetainedMediaTransferTests {
    private func offer(archiveIdentitySha256: String, members: [RetainedSourceMedia.TransferMember]) -> RetainedSourceMedia.TransferOffer {
        RetainedSourceMedia.TransferOffer(archiveIdentitySha256: archiveIdentitySha256, members: members,
            outputSha256: "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81", outputSizeBytes: 3,
            metadata: try! MoldJSON.decoder.decode(OutputMetadata.self, from: Data("{}".utf8)))
    }

    @Test func embeddedRecipeCanReceiveArchiveEnrichedSources() async throws {
        let source = FakeBackend(), target = FakeBackend()
        let embedded = try MoldJSON.decoder.decode(OutputMetadata.self, from: Data(
            #"{"model":"qwen-image-2.1-turbo:q8","seed":2818450336,"version":"0.32.0","edit_image_sha256s":["abc"]}"#.utf8))
        let archived = try MoldJSON.decoder.decode(OutputMetadata.self, from: Data(
            #"{"model":"qwen-image-2.1-turbo:q8","seed":2818450336,"version":"0.32.0 (5b61d17 2026-09-27)","job_id":"original-job","generation_time_ms":8516,"edit_image_sha256s":["abc"]}"#.utf8))
        let base = offer(archiveIdentitySha256: "a".repeated(64), members: [member()])
        source.stub("retainedMediaTransferOffer(for:)", returning:
            RetainedSourceMedia.TransferOffer(archiveIdentitySha256: base.archiveIdentitySha256,
                members: base.members, outputSha256: base.outputSha256,
                outputSizeBytes: base.outputSizeBytes, metadata: archived))
        target.stub("retainedMediaTransferOffer(for:)", returning:
            RetainedSourceMedia.TransferOffer(archiveIdentitySha256: "b".repeated(64),
                members: [], outputSha256: base.outputSha256,
                outputSizeBytes: base.outputSizeBytes, metadata: embedded))
        source.stub("retainedSourceMediaBytes(for:member:)", returning: Data([1, 2, 3]))
        target.stub("importRetainedMedia(_:for:)", returning: ())
        try await RetainedSourceMedia.mirrorSources(for: "old.png", metadata: archived,
            from: source, to: target, as: "copy.png")
        #expect(target.count("importRetainedMedia(_:for:)") == 1)
        target.stub("retainedMediaTransferOffer(for:)", returning:
            RetainedSourceMedia.TransferOffer(archiveIdentitySha256: "b".repeated(64),
                members: base.members.map(\.contentIdentity), outputSha256: base.outputSha256,
                outputSizeBytes: base.outputSizeBytes, metadata: embedded))
        try await RetainedSourceMedia.mirrorSources(for: "old.png", metadata: archived,
            from: source, to: target, as: "copy.png")
        #expect(target.count("importRetainedMedia(_:for:)") == 1)
        #expect(source.count("retainedSourceMediaBytes(for:member:)") == 1)
    }

    @Test func mirrorCompatibilityPreservesRecipesAndConflictingProvenance() {
        let original = Data(#"{"seed":1,"version":"0.32.0 (5b61d17 2026-09-27)","job_id":"one","generation_time_ms":8516,"future_recipe":{"enabled":true}}"#.utf8)
        for changed in [
            #"{"seed":2,"version":"0.32.0","future_recipe":{"enabled":true}}"#,
            #"{"seed":1,"version":"0.31.0","future_recipe":{"enabled":true}}"#,
            #"{"seed":1,"version":"0.32.0","job_id":"two","future_recipe":{"enabled":true}}"#,
            #"{"seed":1,"version":"0.32.0","generation_time_ms":99,"future_recipe":{"enabled":true}}"#,
            #"{"seed":1,"version":"0.32.0","future_recipe":{"enabled":false}}"#,
            #"{"seed":1,"version":"0.32.0 (7efd234 2026-09-27)","future_recipe":{"enabled":true}}"#,
        ] {
            #expect(!RetainedSourceMedia.mirrorMetadataMatches(original, Data(changed.utf8)))
        }
        let embedded = Data(#"{"seed":1,"version":"0.32.0","future_recipe":{"enabled":true}}"#.utf8)
        #expect(RetainedSourceMedia.mirrorMetadataMatches(original, embedded))
        #expect(RetainedSourceMedia.mirrorMetadataMatches(embedded, original))
        #expect(!RetainedSourceMedia.mirrorMetadataMatches(nil as Data?, embedded))
    }

    @Test func destinationRecipeAndSizeConflictsNeverFetchRetainedPayloads() async throws {
        let originalMetadata = try MoldJSON.decoder.decode(OutputMetadata.self, from: Data(
            #"{"seed":1,"edit_image_sha256s":["original"]}"#.utf8))
        for (index, recipe) in [#"{"seed":2,"edit_image_sha256s":["original"]}"#,
                       #"{"seed":1,"edit_image_sha256s":["changed"]}"#,
                       #"{"seed":1,"edit_image_sha256s":["original"]}"#].enumerated() {
            let source = FakeBackend(), target = FakeBackend()
            let base = offer(archiveIdentitySha256: "a".repeated(64), members: [member()])
            source.stub("retainedMediaTransferOffer(for:)", returning:
                RetainedSourceMedia.TransferOffer(archiveIdentitySha256: base.archiveIdentitySha256,
                    members: base.members, outputSha256: base.outputSha256,
                    outputSizeBytes: 3, metadata: originalMetadata))
            target.stub("retainedMediaTransferOffer(for:)", returning:
                RetainedSourceMedia.TransferOffer(archiveIdentitySha256: "b".repeated(64), members: [],
                    outputSha256: base.outputSha256, outputSizeBytes: index == 2 ? 4 : 3,
                    metadata: try MoldJSON.decoder.decode(OutputMetadata.self, from: Data(recipe.utf8))))
            await #expect(throws: (any Error).self) {
                try await RetainedSourceMedia.mirrorSources(for: "old.png", metadata: originalMetadata,
                    from: source, to: target, as: "copy.png")
            }
            #expect(source.count("retainedSourceMediaBytes(for:member:)") == 0)
            #expect(target.count("importRetainedMedia(_:for:)") == 0)
        }
    }

    @Test func unsupportedDestinationRefusesBeforeAnyOutputImport() async throws {
        let source = FakeBackend(), target = FakeBackend()
        source.stub("retainedMediaTransferOffer(for:)", returning: offer(archiveIdentitySha256: "a".repeated(64), members: [member()]))
        target.stub("capabilities()", returning: try MoldJSON.decoder.decode(Capabilities.self, from: Data("{}".utf8)))
        await #expect(throws: (any Error).self) {
            _ = try await RetainedSourceMedia.preflightMirror(for: "original.png", metadata: nil, from: source, to: target)
        }
        #expect(target.count("importPrint(_:as:)") == 0)
        #expect(target.count("importRetainedMedia(_:for:)") == 0)
        #expect(source.count("retainedSourceMediaBytes(for:member:)") == 0)
    }

    @Test func sourceBearingPreflightRequiresExplicitTransferProtocol() async throws {
        for block in [#"{"durable_media":{"protocol_version":2,"encrypted_at_rest":true,"generate_request_media":true,"identity":true}}"#,
                      #"{"retained_media_transfer":{"protocol_version":2}}"#,
                      #"{"retained_media_transfer":{"protocol_version":1}}"#] {
            let source = FakeBackend(), target = FakeBackend()
            source.stub("retainedMediaTransferOffer(for:)", returning: offer(archiveIdentitySha256: "a".repeated(64), members: [member()]))
            target.stub("capabilities()", returning: try MoldJSON.decoder.decode(Capabilities.self, from: Data(block.utf8)))
            if block.contains(#""retained_media_transfer":{"protocol_version":1}"#) {
                #expect(try await RetainedSourceMedia.preflightMirror(for: "original.png", metadata: nil, from: source, to: target) == "a".repeated(64))
            } else {
                await #expect(throws: (any Error).self) {
                    _ = try await RetainedSourceMedia.preflightMirror(for: "original.png", metadata: nil, from: source, to: target)
                }
            }
        }
    }

    @Test func unavailableSourceRefusesBeforeOutputImport() async throws {
        let disclosed = try MoldJSON.decoder.decode(OutputMetadata.self, from: Data(#"{"source_image_sha256":"abc"}"#.utf8))
        for legacy in [true, false] {
            let source = FakeBackend(), target = FakeBackend()
            if legacy {
                source.stub("retainedMediaTransferOffer(for:)", throwing: MoldClientError.http(status: 404, code: nil, message: nil))
                source.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .unavailableLegacy, members: []))
            } else {
                source.stub("retainedMediaTransferOffer(for:)", returning: offer(archiveIdentitySha256: "a".repeated(64), members: []))
            }
            await #expect(throws: (any Error).self) {
                _ = try await RetainedSourceMedia.preflightMirror(for: "original.png", metadata: disclosed, from: source, to: target)
            }
            #expect(target.calls.isEmpty)
        }
    }

    @Test func legacyUnknownRecipeCannotProveSourceFreeCopy() async throws {
        let source = FakeBackend(), target = FakeBackend()
        source.stub("retainedMediaTransferOffer(for:)", throwing: MoldClientError.http(status: 404, code: nil, message: nil))
        source.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .unavailableLegacy, members: []))
        await #expect(throws: (any Error).self) {
            _ = try await RetainedSourceMedia.preflightMirror(for: "original.png", metadata: nil, from: source, to: target)
        }
        await #expect(throws: (any Error).self) {
            try await RetainedSourceMedia.mirrorSources(for: "original.png", metadata: nil, from: source, to: target, as: "copy.png")
        }
        #expect(target.calls.isEmpty)
        let sourceFree = try MoldJSON.decoder.decode(OutputMetadata.self, from: Data("{}".utf8))
        #expect(try await RetainedSourceMedia.preflightMirror(for: "original.png", metadata: sourceFree, from: source, to: target) == nil)
        try await RetainedSourceMedia.mirrorSources(for: "original.png", metadata: sourceFree, from: source, to: target, as: "copy.png")
        #expect(target.calls.isEmpty)
    }

    @Test func sourceFreeOutputNeedsNoDurableDestination() async throws {
        let source = FakeBackend(), target = FakeBackend()
        source.stub("retainedMediaTransferOffer(for:)", returning: offer(archiveIdentitySha256: "a".repeated(64), members: []))
        let identity = try await RetainedSourceMedia.preflightMirror(for: "original.png", metadata: nil, from: source, to: target)
        #expect(identity == "a".repeated(64))
        #expect(target.calls.isEmpty)
    }

    private func member(_ id: String = "source", role: String = "source_image",
                        position: String = "scalar", bytes: Data = Data([1, 2, 3])) -> RetainedSourceMedia.TransferMember {
        .init(memberId: id, role: role, position: position, sizeBytes: bytes.count,
              sha256: SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined())
    }

    @Test func aLibraryMirrorCommitsItsSourceBeforeCompleting() async throws {
        let source = FakeBackend(), target = FakeBackend()
        let member = RetainedSourceMedia.TransferMember(memberId: "source", role: "source_image",
            position: "scalar", sizeBytes: 3,
            sha256: "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81")
        source.stub("retainedMediaTransferOffer(for:)", returning:
            offer(archiveIdentitySha256: "a".repeated(64), members: [member]))
        target.stub("retainedMediaTransferOffer(for:)", returning:
            offer(archiveIdentitySha256: "b".repeated(64), members: []))
        source.stub("retainedSourceMediaBytes(for:member:)", returning: Data([1, 2, 3]))
        target.stub("importRetainedMedia(_:for:)", returning: ())
        try await RetainedSourceMedia.mirrorSources(for: "original.png", metadata: nil,
            from: source, to: target, as: "copy.png")
        #expect(target.count("importRetainedMedia(_:for:)") == 1)
        #expect(source.count("retainedSourceMediaBytes(for:member:)") == 1)
        let transfer = try #require(target.calls.last?.arguments.first as? RetainedSourceMedia.Transfer)
        #expect(transfer.archiveIdentitySha256 == "b".repeated(64))
        #expect(transfer.files.allSatisfy { !FileManager.default.fileExists(atPath: $0.path) })
    }

    @Test func matchingRetainedSourcesDoNotDownloadOrUploadAgain() async throws {
        let source = FakeBackend(), target = FakeBackend()
        let input = member()
        source.stub("retainedMediaTransferOffer(for:)", returning:
            offer(archiveIdentitySha256: "a".repeated(64), members: [input]))
        target.stub("retainedMediaTransferOffer(for:)", returning:
            offer(archiveIdentitySha256: "b".repeated(64), members: [input.contentIdentity]))
        try await RetainedSourceMedia.mirrorSources(for: "old.png", metadata: nil,
            from: source, to: target, as: "copy.png")
        #expect(source.count("retainedSourceMediaBytes(for:member:)") == 0)
        #expect(target.count("importRetainedMedia(_:for:)") == 0)
    }

    @Test func sourceReplacementAfterOutputDownloadCannotAttachNewInputs() async throws {
        let source = FakeBackend(), target = FakeBackend()
        source.stub("retainedMediaTransferOffer(for:)", returning:
            offer(archiveIdentitySha256: "b".repeated(64), members: [member()]))
        await #expect(throws: (any Error).self) {
            try await RetainedSourceMedia.mirrorSources(for: "old.png", metadata: nil,
                from: source, to: target, as: "copy.png", expectedSourceArchiveIdentity: "a".repeated(64))
        }
        #expect(target.calls.isEmpty)
        #expect(source.count("retainedSourceMediaBytes(for:member:)") == 0)
    }

    @Test func destinationBytesMustMatchTheSourceOutputBeforeBindingInputs() async throws {
        let source = FakeBackend(), target = FakeBackend()
        let original = offer(archiveIdentitySha256: "a".repeated(64), members: [member()])
        source.stub("retainedMediaTransferOffer(for:)", returning: original)
        target.stub("retainedMediaTransferOffer(for:)", returning:
            RetainedSourceMedia.TransferOffer(archiveIdentitySha256: "b".repeated(64), members: [],
                outputSha256: "c".repeated(64), outputSizeBytes: 3, metadata: original.metadata))
        await #expect(throws: (any Error).self) {
            try await RetainedSourceMedia.mirrorSources(for: "old.png", metadata: nil,
                from: source, to: target, as: "copy.png")
        }
        #expect(source.count("retainedSourceMediaBytes(for:member:)") == 0)
        #expect(target.count("importRetainedMedia(_:for:)") == 0)
    }

    @Test func duplicateSlotsAreRejectedBeforeFetchingAnyPayload() async throws {
        let source = FakeBackend(), target = FakeBackend()
        source.stub("retainedMediaTransferOffer(for:)", returning:
            offer(archiveIdentitySha256: "a".repeated(64), members: [member("one"), member("two")]))
        await #expect(throws: (any Error).self) {
            try await RetainedSourceMedia.mirrorSources(for: "old.png", metadata: nil,
                from: source, to: target, as: "copy.png")
        }
        #expect(target.calls.isEmpty)
        #expect(source.count("retainedSourceMediaBytes(for:member:)") == 0)
    }

    @Test func changedSourceBytesNeverCommitACompletedCopy() async throws {
        let source = FakeBackend(), target = FakeBackend()
        source.stub("retainedMediaTransferOffer(for:)", returning:
            offer(archiveIdentitySha256: "a".repeated(64), members: [member()]))
        target.stub("retainedMediaTransferOffer(for:)", returning:
            offer(archiveIdentitySha256: "b".repeated(64), members: []))
        source.stub("retainedSourceMediaBytes(for:member:)", returning: Data([4, 5, 6]))
        await #expect(throws: (any Error).self) {
            try await RetainedSourceMedia.mirrorSources(for: "old.png", metadata: nil,
                from: source, to: target, as: "copy.png")
        }
        #expect(target.count("importRetainedMedia(_:for:)") == 0)
    }

    @Test func olderDestinationExplainsAnIncompleteSourceCopy() async throws {
        let source = FakeBackend(), target = FakeBackend()
        source.stub("retainedMediaTransferOffer(for:)", returning:
            offer(archiveIdentitySha256: "a".repeated(64), members: [member()]))
        target.stub("retainedMediaTransferOffer(for:)", throwing: MoldClientError.http(status: 404, code: nil, message: nil))
        do {
            try await RetainedSourceMedia.mirrorSources(for: "old.png", metadata: nil,
                from: source, to: target, as: "copy.png")
            Issue.record("An old destination cannot acknowledge retained sources")
        } catch let error as MoldClientError {
            #expect(error.localizedDescription.contains("Update the destination"))
        }
        #expect(source.count("retainedSourceMediaBytes(for:member:)") == 0)
    }

    @Test func oldSourceWithAvailableRetainedBytesCannotSilentlyLoseThem() async throws {
        let source = FakeBackend(), target = FakeBackend()
        source.stub("retainedMediaTransferOffer(for:)", throwing: MoldClientError.http(status: 404, code: nil, message: nil))
        source.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available))
        await #expect(throws: (any Error).self) {
            try await RetainedSourceMedia.mirrorSources(for: "old.png", metadata: nil,
                from: source, to: target, as: "copy.png")
        }
        #expect(target.calls.isEmpty)
    }

    @Test func transferFrameKeepsReferenceOrderAndStreamsExactlyDeclaredBytes() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let first = root.appendingPathComponent("first"), second = root.appendingPathComponent("second")
        try Data([1, 2, 3]).write(to: first)
        try Data([4, 5]).write(to: second)
        let transfer = RetainedSourceMedia.Transfer(archiveIdentitySha256: "a".repeated(64), members: [
            member("one", role: "references", position: "item:0").contentIdentity,
            member("two", role: "references", position: "item:1", bytes: Data([4, 5])).contentIdentity,
        ], files: [first, second])
        let frame = root.appendingPathComponent("frame")
        try transfer.writeBody(to: frame)
        let body = try Data(contentsOf: frame)
        let length = body.prefix(4).reduce(0) { ($0 << 8) | Int($1) }
        let descriptor = try #require(JSONSerialization.jsonObject(with: body.subdata(in: 4..<4+length)) as? [String: Any])
        #expect(descriptor["archive_identity_sha256"] as? String == "a".repeated(64))
        #expect(Array(body.dropFirst(4+length)) == [1, 2, 3, 4, 5])
        let members = try #require(descriptor["members"] as? [[String: Any]])
        #expect(members.compactMap { $0["position"] as? String } == ["item:0", "item:1"])
    }
}

private extension String {
    func repeated(_ count: Int) -> String { String(repeating: self, count: count) }
}
