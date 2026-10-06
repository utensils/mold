import CryptoKit
import Foundation
import MoldClientTesting
import Testing
@testable import MoldClient

struct RetainedMirrorEmbeddedRecipeTests {
    private func png(_ json: String) -> Data {
        let payload = Data("mold:parameters\0\(json)".utf8)
        var bytes = Data([137, 80, 78, 71, 13, 10, 26, 10])
        let length = UInt32(payload.count)
        bytes.append(contentsOf: [UInt8(length >> 24), UInt8((length >> 16) & 255),
                                  UInt8((length >> 8) & 255), UInt8(length & 255)])
        bytes.append(Data("tEXt".utf8)); bytes.append(payload)
        bytes.append(Data(repeating: 0, count: 4))
        return bytes
    }

    @Test func embeddedMetadataCeilingRejectsOversizedChunksBeforeParsing() {
        let json = "{\"scheduler\":\"ddim\",\"padding\":\"" + String(repeating: "x", count: 1024) + "\"}"
        #expect(EmbeddedPrintMetadata.json(in: png(json), named: "output.png", metadataCeiling: 128) == nil)
        #expect(EmbeddedPrintMetadata.json(in: png(#"{"scheduler":"ddim"}"#), named: "output.png", metadataCeiling: 128) != nil)
        var jpeg = Data([0xFF, 0xD8, 0xFF, 0xFE])
        let jpegComment = Data("mold:parameters \(json)".utf8)
        let length = jpegComment.count + 2
        jpeg.append(contentsOf: [UInt8(length >> 8), UInt8(length & 255)])
        jpeg.append(jpegComment)
        #expect(EmbeddedPrintMetadata.json(in: jpeg, named: "output.jpg", metadataCeiling: 128) == nil)
        #expect(EmbeddedPrintMetadata.json(in: png(json), named: "output.png", metadataCeiling: -1) == nil)
        var gif = Data("GIF89a".utf8)
        gif.append(contentsOf: [0x21, 0xFE])
        let comment = Data("mold:parameters \(json)".utf8)
        for start in stride(from: 0, to: comment.count, by: 255) {
            let block = comment[start..<min(start + 255, comment.count)]
            gif.append(UInt8(block.count)); gif.append(contentsOf: block)
        }
        gif.append(0)
        #expect(EmbeddedPrintMetadata.json(in: gif, named: "output.gif", metadataCeiling: 128) == nil)
    }

    @Test func omittedRecipeFactsRequireExactEmbeddedOutputEvidence() async throws {
        for field in [#""scheduler":"ddim""#, #""transparent_background":true"#] {
            for scenario in ["matching", "wrongEmbedded", "changedOutput", "conflictingArchive"] {
                let source = FakeBackend(), target = FakeBackend()
                let embedded = scenario == "wrongEmbedded" ? #"{"seed":42}"# : "{\"seed\":42,\(field)}"
                let bytes = png(embedded)
                let hash = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
                let archiveJSON = scenario == "conflictingArchive"
                    ? (field.contains("scheduler") ? #"{"seed":42,"scheduler":"euler"}"# : #"{"seed":42,"transparent_background":false}"#)
                    : #"{"seed":42}"#
                let archive = try MoldJSON.decoder.decode(OutputMetadata.self, from: Data(archiveJSON.utf8))
                let local = try MoldJSON.decoder.decode(OutputMetadata.self, from: Data("{\"seed\":42,\(field)}".utf8))
                let input = Data([1, 2, 3])
                let member = RetainedSourceMedia.TransferMember(memberId: "source", role: "source_image", position: "scalar", sizeBytes: 3,
                    sha256: SHA256.hash(data: input).map { String(format: "%02x", $0) }.joined())
                let offer = RetainedSourceMedia.TransferOffer(archiveIdentitySha256: String(repeating: "a", count: 64), members: [member],
                    outputSha256: scenario == "changedOutput" ? String(repeating: "c", count: 64) : hash,
                    outputSizeBytes: bytes.count, metadata: archive)
                source.stub("retainedMediaTransferOffer(for:)", returning: offer)
                target.stub("retainedMediaTransferOffer(for:)", returning:
                    RetainedSourceMedia.TransferOffer(archiveIdentitySha256: String(repeating: "b", count: 64), members: [],
                        outputSha256: offer.outputSha256, outputSizeBytes: bytes.count, metadata: local))
                let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".png")
                try bytes.write(to: file)
                defer { try? FileManager.default.removeItem(at: file) }
                source.stub("mediaFile(_:trashed:)", returning: file)
                source.stub("retainedSourceMediaBytes(for:member:)", returning: input)
                target.stub("importRetainedMedia(_:for:)", returning: ())
                if scenario == "matching" {
                    try await RetainedSourceMedia.mirrorSources(for: "original.png", metadata: archive,
                        from: source, to: target, as: "copy.png", expectedSourceArchiveIdentity: offer.archiveIdentitySha256)
                    #expect(target.count("importRetainedMedia(_:for:)") == 1)
                } else {
                    await #expect(throws: (any Error).self) {
                        try await RetainedSourceMedia.mirrorSources(for: "original.png", metadata: archive,
                            from: source, to: target, as: "copy.png", expectedSourceArchiveIdentity: offer.archiveIdentitySha256)
                    }
                    #expect(target.count("importRetainedMedia(_:for:)") == 0)
                    #expect(source.count("retainedSourceMediaBytes(for:member:)") == 0)
                }
            }
        }
    }
}
