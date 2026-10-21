import Foundation
import Testing
@testable import MoldClient

private final class TransferRecipeTransport: StubTransport {
    override class func response(for path: String) -> (status: Int, body: Data)? {
        if path == "/api/capabilities" { return (200, Data(#"{"retained_media_transfer":{"protocol_version":1}}"#.utf8)) }
        guard path.hasSuffix("/transfer") else { return nil }
        let changed = path.contains("copy.mp4")
        let archive = String(repeating: changed ? "b" : "a", count: 64)
        let digest = String(repeating: "c", count: 64)
        return (200, Data("""
        {"archive_identity_sha256":"\(archive)","output_sha256":"\(digest)","output_size_bytes":3,"metadata":{"prompt":"literal\\\\nline\\r\\nnext","seed":18446744073709551615,"video_job":{"id":"parent","source_frames":"0","total_frames":"144","authoring":{"duration_ms":"6000","boundary_frames":124,"seed":"18446744073709551615"},"artifacts":{"provenance_sha256":"\(changed ? "different" : "original")"}}},"members":[{"member_id":"member","role":"source_image","position":"0","sink":"memory","size_bytes":3,"sha256":"\(digest)"}]}
        """.utf8))
    }
}

struct RetainedTransferRecipeTests {
    @Test func unknownPublicationFactsCannotDisappearFromMirrorVerification() async throws {
        let backend = TransferRecipeTransport.backend()
        await #expect(throws: (any Error).self) {
            try await RetainedSourceMedia.mirrorSources(for: "original.mp4", metadata: nil,
                from: backend, to: backend, as: "copy.mp4")
        }
    }
    @Test func authenticatedArchiveRecipePreservesCompleteVideoMetadataAndCompletionFacts() async throws {
        let literal = "literal\\nline\r\nnext"
        let embedded = Data(#"{"prompt":"literal\\nline\r\nnext","model":"video","seed":18446744073709551615,"width":768,"height":768,"steps":3,"guidance":0,"frames":144,"fps":24}"#.utf8)
        var object = try #require(JSONSerialization.jsonObject(with: embedded) as? [String:Any])
        object["enable_audio"] = true
        object["spatial_upscale"] = "2x"
        object["temporal_upscale"] = "2x"
        object["future_recipe"] = ["retain":"exact"]
        object["video_job"] = ["id":"parent","source_frames":"24","total_frames":"168",
            "authoring":["duration_ms":"6000","boundary_frames":124,"input_frames":107,"seed":"18446744073709551615",
                "scenes":[["prompt":literal,"duration_ms":"3000"],["prompt":"second\\n","duration_ms":"3000"]]],
            "artifacts":["version":1,"bundle_id":"opaque","future_fact":"must survive"]] as [String:Any]
        let embeddedRecipe = try JSONSerialization.data(withJSONObject: object, options: .sortedKeys)
        object["job_id"] = "published-completion"
        object["generation_time_ms"] = 1234
        let raw = try JSONSerialization.data(withJSONObject: object, options: .sortedKeys)
        let metadata = try MoldJSON.decoder.decode(OutputMetadata.self, from: raw)
        let listing = try JSONSerialization.data(withJSONObject: ["filename":"original.png","timestamp":1700000000,"metadata":object])
        let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: listing)
        let payload = Data("mold:parameters".utf8) + Data([0]) + embeddedRecipe
        var png = Data([137,80,78,71,13,10,26,10])
        withUnsafeBytes(of: UInt32(payload.count).bigEndian) { png.append(contentsOf: $0) }
        png.append(Data("tEXt".utf8)); png.append(payload); png.append(Data(repeating: 0, count: 4))
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("original.png"); try png.write(to: file)
        var source = RetainedSourceMedia.TransferOffer(archiveIdentitySha256: String(repeating:"a",count:64), members: [],
            outputSha256: TransportDigest.sha256(png), outputSizeBytes: png.count, metadata: metadata)
        source.rawMetadataJSON = raw
        let item = try await LibraryMirrorFiles.importDescriptor(for: print, file: file, sourceOffer: source)
        let framedFile = root.appendingPathComponent("framed"); try item.writeBody(to: framedFile)
        let framed = try Data(contentsOf: framedFile)
        let count = framed[0..<4].reduce(0) { $0 << 8 | Int($1) }
        let descriptor = try #require(JSONSerialization.jsonObject(with: Data(framed[12..<12+count])) as? [String:Any])
        let copied = try #require(descriptor["metadata"] as? [String:Any])
        #expect(NSDictionary(dictionary: copied).isEqual(to: object))
        #expect((copied["seed"] as? NSNumber)?.uint64Value == UInt64.max)
        let copiedMetadata = try MoldJSON.decoder.decode(OutputMetadata.self, from: JSONSerialization.data(withJSONObject: copied))
        let reuse = try VideoSettingsReuse(metadata: copiedMetadata)
        #expect(reuse.draft.prompt == literal)
        #expect(reuse.draft.seed == UInt64.max && reuse.draft.locksSeed)
        #expect(reuse.draft.frames == 107)
        #expect(reuse.draft.spatialUpscale == "2x" && reuse.draft.temporalUpscale == "2x")
        #expect(reuse.duration?.draft.scenes.map(\.prompt) == [literal,"second\\n"])
        #expect(reuse.duration?.draft.source == nil)
        #expect(reuse.notice?.contains("original video is not attached") == true)
        try Data([1,2,3]).write(to: file)
        await #expect(throws: (any Error).self) {
            try await LibraryMirrorFiles.importDescriptor(for: print, file: file, sourceOffer: source)
        }
    }

    @Test func capturedSourceRecipeIsRecheckedEvenWhenTheOutputIdentityDidNotChange() async throws {
        let backend = TransferRecipeTransport.backend()
        var captured = try await RetainedSourceMedia.prepareMirror(for: "original.mp4", metadata: nil, from: backend, to: backend).sourceOffer!
        let object = try #require(JSONSerialization.jsonObject(with: captured.rawMetadataJSON!) as? [String:Any])
        var changed = object; changed["future_recipe"] = "changed"
        captured.rawMetadataJSON = try JSONSerialization.data(withJSONObject: changed)
        let frozen = RetainedSourceMedia.MirrorSourcePreparation(sourceOffer: captured)
        await #expect(throws: (any Error).self) {
            try await RetainedSourceMedia.revalidateMirrorSource(frozen, filename: "original.mp4", from: backend)
        }
    }

}
