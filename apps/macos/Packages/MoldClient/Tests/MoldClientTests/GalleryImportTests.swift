import Foundation
import Testing

@testable import MoldClient

private final class MirrorListingTransport: StubTransport {
    override class func response(for path: String) -> (status: Int, body: Data)? {
        guard path == "/api/gallery" else { return nil }
        return (200, Data(#"[{"filename":"remote.png","timestamp":1700000000,"metadata_synthetic":true,"metadata":{"prompt":"a fox","model":"flux-dev:q8","seed":42,"steps":24,"guidance":3.5,"width":1024,"height":768,"version":"0.31.0","batch_id":"batch-1","true_cfg":2.75,"future_recipe_field":{"retain_me":true}}}]"#.utf8))
    }
}

private final class UnauthorizedMediaTransport: StubTransport {
    override class func response(for path: String) -> (status: Int, body: Data)? {
        guard path == "/api/gallery/image/clip.mp4" else { return nil }
        return (401, Data(#"{"error":"unauthorized"}"#.utf8))
    }
}

/// The body `PUT /api/gallery/import/:filename` expects.
///
/// A length-prefixed frame rather than multipart, so a host can start writing
/// the file before it has read it: twelve bytes of header, then the
/// descriptor, then the bytes. Getting the widths or the byte order wrong
/// produces a 422 that says nothing useful, so they are pinned here.
@Suite struct GalleryImportSuite {

    @Test func streamedDownloadReportsAuthenticationFailure() async {
        let backend = UnauthorizedMediaTransport.backend()
        do {
            _ = try await backend.mediaFile("clip.mp4", trashed: false)
            Issue.record("expected authentication failure")
        } catch MoldClientError.unauthorized {
            // The file-backed route uses the same refusal handling as media().
        } catch {
            Issue.record("wrong error: \(error)")
        }
    }

    @Test func canonicalRecipeRetainsUnknownFieldsFromNewerHosts() throws {
        let row = Data(#"{"filename":"clip.mp4","timestamp":1000,"metadata":{"prompt":"same"}}"#.utf8)
        var first = try MoldJSON.decoder.decode(GalleryPrint.self, from: row)
        var second = try MoldJSON.decoder.decode(GalleryPrint.self, from: row)
        first.rawMetadataJSON = Data(#"{"prompt":"same","future_field":1}"#.utf8)
        second.rawMetadataJSON = Data(#"{"future_field":2,"prompt":"same"}"#.utf8)

        #expect(first.metadata == second.metadata)
        #expect(first.canonicalMetadataJSON != second.canonicalMetadataJSON)
        second.rawMetadataJSON = Data(#"{"future_field":1,"prompt":"same"}"#.utf8)
        #expect(first.canonicalMetadataJSON == second.canonicalMetadataJSON)
    }

    private func body(_ import_: GalleryImport) throws -> [UInt8] {
        Array(try import_.body())
    }

    @Test func theHeaderIsFourBytesOfDescriptorThenEightOfFile() throws {
        let bytes = try body(GalleryImport(prompt: "owls", model: "import", version: "0.29.0",
                                           file: Data([1, 2, 3])))
        let descriptorLength = bytes[0..<4].reduce(0) { $0 << 8 | Int($1) }
        let fileLength = bytes[4..<12].reduce(0) { $0 << 8 | Int($1) }
        #expect(fileLength == 3)
        #expect(bytes.count == 12 + descriptorLength + 3)
        // Big endian, which is what "three" looks like in the last byte.
        #expect(Array(bytes[4..<12]) == [0, 0, 0, 0, 0, 0, 0, 3])
    }

    @Test func theFileBytesRideAtTheEndUntouched() throws {
        let file = Data([0xDE, 0xAD, 0xBE, 0xEF])
        let bytes = try body(GalleryImport(prompt: "owls", model: "import", version: "0.29.0", file: file))
        #expect(Array(bytes.suffix(4)) == [0xDE, 0xAD, 0xBE, 0xEF])
    }

    @Test func theDescriptorCarriesTheMetadataTheServerRequires() throws {
        let bytes = try body(GalleryImport(prompt: "a brass helmet", model: "import", width: 1024,
                                           height: 768, version: "0.29.0", file: Data([1])))
        let length = bytes[0..<4].reduce(0) { $0 << 8 | Int($1) }
        let json = try JSONSerialization.jsonObject(
            with: Data(bytes[12..<(12 + length)])) as? [String: Any]
        let metadata = json?["metadata"] as? [String: Any]
        // `prompt` and `model` are not optional on the wire; a descriptor
        // without them is a 422.
        #expect(metadata?["prompt"] as? String == "a brass helmet")
        #expect(metadata?["model"] as? String == "import")
        // Required on the wire, and meaningless for a file nobody rendered.
        #expect(metadata?["seed"] as? Int == 0)
        #expect(metadata?["steps"] as? Int == 0)
        #expect(metadata?["version"] as? String == "0.29.0")
        // The picture's real shape, where it is knowable.
        #expect(metadata?["width"] as? Int == 1024)
        #expect(metadata?["height"] as? Int == 768)
        // We invented this metadata; saying so is what stops a host treating
        // an imported picture as something mold rendered.
        #expect(json?["metadata_synthetic"] as? Bool == true)
    }

    /// An imported file keeps the date it was made, so it lands where it
    /// belongs in a day-sectioned timeline rather than at the top of today.
    @Test func aStatedDateTravelsWithIt() throws {
        let made = Date(timeIntervalSince1970: 1_700_000_000)
        let bytes = try body(GalleryImport(prompt: "owls", model: "import", version: "0.29.0",
                                           file: Data([1]), timestamp: made))
        let length = bytes[0..<4].reduce(0) { $0 << 8 | Int($1) }
        let json = try JSONSerialization.jsonObject(
            with: Data(bytes[12..<(12 + length)])) as? [String: Any]
        #expect(json?["timestamp"] as? UInt64 == 1_700_000_000)
    }

    @Test func aPromptlessImportIsStillDescribedBySomething() throws {
        // `prompt` cannot be empty on the wire, and an imported picture has
        // none -- so it is described by where it came from.
        let import_ = GalleryImport(importing: Data([1]), named: "holiday.png",
                                    version: "0.29.0")
        #expect(import_.prompt == "Imported \u{2014} holiday.png")
        #expect(import_.model == "import")
    }

    @Test func mirroringKeepsTheOriginalRecipeAndItsProvenance() throws {
        let source = #"{"filename":"remote.png","timestamp":1700000000,"metadata_synthetic":false,"metadata":{"prompt":"a fox","model":"flux-dev:q8","seed":42,"steps":24,"guidance":3.5,"width":1024,"height":768,"version":"0.31.0"}}"#
        let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(source.utf8))
        let bytes = try body(GalleryImport(mirroring: print, file: Data([1, 2])))
        let length = bytes[0..<4].reduce(0) { $0 << 8 | Int($1) }
        let json = try #require(JSONSerialization.jsonObject(
            with: Data(bytes[12..<(12 + length)])) as? [String: Any])
        let metadata = try #require(json["metadata"] as? [String: Any])
        #expect(metadata["prompt"] as? String == "a fox")
        #expect(metadata["model"] as? String == "flux-dev:q8")
        #expect(metadata["seed"] as? Int == 42)
        #expect(metadata["steps"] as? Int == 24)
        #expect(json["metadata_synthetic"] as? Bool == false)
        #expect(json["timestamp"] as? UInt64 == 1_700_000_000)
        #expect(Array(bytes.suffix(2)) == [1, 2])
    }

    @Test func embeddedRecipeWinsOverAStaleGalleryRow() throws {
        let listing = #"{"filename":"remote.png","timestamp":1700000000,"metadata_synthetic":true,"metadata":{"prompt":"stale","model":"flux-dev:q8","seed":42,"steps":24,"guidance":3.5,"width":1024,"height":768,"version":"0.31.0"}}"#
        let embedded = #"{"prompt":"original","model":"flux-dev:q8","seed":42,"steps":24,"guidance":3.5,"width":1024,"height":768,"version":"0.31.0"}"#
        let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(listing.utf8))
        let text = Array("mold:parameters".utf8) + [0, 0, 0, 0, 0] + Array(embedded.utf8)
        let length = UInt32(text.count).bigEndian
        var png = Data([137, 80, 78, 71, 13, 10, 26, 10])
        withUnsafeBytes(of: length) { png.append(contentsOf: $0) }
        png.append(Data("iTXt".utf8))
        png.append(contentsOf: text)
        png.append(contentsOf: [0, 0, 0, 0])

        let framed = try GalleryImport(mirroring: print, file: png).body()
        let headerLength = framed[0..<4].reduce(0) { $0 << 8 | Int($1) }
        let descriptor = try #require(JSONSerialization.jsonObject(
            with: Data(framed[12..<(12 + headerLength)])) as? [String: Any])
        let recipe = try #require(descriptor["metadata"] as? [String: Any])
        #expect(recipe["prompt"] as? String == "original")
        #expect(descriptor["metadata_synthetic"] as? Bool == false)
    }

    @Test func embeddedRecipeReaderAcceptsTextAndJPEGComments() {
        let json = Data(#"{"prompt":"original","model":"flux-dev:q8"}"#.utf8)
        let payload = Data("mold:parameters".utf8) + Data([0]) + json
        var png = Data([137, 80, 78, 71, 13, 10, 26, 10])
        withUnsafeBytes(of: UInt32(payload.count).bigEndian) { png.append(contentsOf: $0) }
        png.append(Data("tEXt".utf8))
        png.append(payload)
        png.append(contentsOf: [0, 0, 0, 0])
        #expect(EmbeddedPrintMetadata.json(in: png, named: "print.png") == json)

        let comment = Data("mold:parameters ".utf8) + json
        var jpeg = Data([0xFF, 0xD8, 0xFF, 0xFE])
        withUnsafeBytes(of: UInt16(comment.count + 2).bigEndian) { jpeg.append(contentsOf: $0) }
        jpeg.append(comment)
        jpeg.append(contentsOf: [0xFF, 0xD9])
        #expect(EmbeddedPrintMetadata.json(in: jpeg, named: "print.jpg") == json)
        #expect(EmbeddedPrintMetadata.json(in: Data([0xFF]), named: "print.jpg") == nil)
    }

    @Test func fileBackedBodyMatchesTheInMemoryContract() throws {
        let item = GalleryImport(prompt: "a fox", model: "import", version: "test", file: Data([1, 2, 3]))
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        try item.writeBody(to: url)
        #expect(try Data(contentsOf: url) == item.body())
    }

    @Test func fileBackedMirrorFramesAClipWithoutHoldingItsBytesInTheImport() throws {
        let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(#"""
            {
            "filename":"clip.mp4","timestamp":1700000000,
            "metadata":{"prompt":"a fox","model":"ltx","seed":7}
            }
            """#.utf8))
        let source = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let upload = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {
            try? FileManager.default.removeItem(at: source)
            try? FileManager.default.removeItem(at: upload)
        }
        try Data(repeating: 0xAB, count: 2_100_000).write(to: source)
        let item = try GalleryImport(mirroring: print, fileAt: source)
        #expect(item.file.isEmpty)
        try item.writeBody(to: upload)
        let framed = try Data(contentsOf: upload)
        let fileLength = framed[4..<12].reduce(0) { $0 << 8 | UInt64($1) }
        #expect(fileLength == 2_100_000)
        #expect(framed.suffix(2_100_000) == Data(repeating: 0xAB, count: 2_100_000))
    }

    @Test func fetchedMirrorRetainsFieldsUnknownToThisClientAcrossOptimisticEdits() async throws {
        let fetched = try await MirrorListingTransport.backend().gallery(etag: nil)
        guard case let .fresh(prints, _) = fetched else { Issue.record("expected listing"); return }
        var editable = GalleryPrint.Mutable(try #require(prints.first))
        editable.favorite = true
        let print = editable.build()
        #expect(print.metadataSynthetic == true)
        #expect(print.rawMetadataAvailable)
        let bytes = try body(GalleryImport(mirroring: print, file: Data([1])))
        let length = bytes[0..<4].reduce(0) { $0 << 8 | Int($1) }
        let json = try #require(JSONSerialization.jsonObject(
            with: Data(bytes[12..<(12 + length)])) as? [String: Any])
        let metadata = try #require(json["metadata"] as? [String: Any])
        #expect(metadata["batch_id"] as? String == "batch-1")
        #expect(metadata["true_cfg"] as? Double == 2.75)
        #expect((metadata["future_recipe_field"] as? [String: Bool])?["retain_me"] == true)
        #expect(json["metadata_synthetic"] as? Bool == true)
    }
}
