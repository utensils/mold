import Foundation
import Testing

@testable import MoldClient

/// The body `PUT /api/gallery/import/:filename` expects.
///
/// A length-prefixed frame rather than multipart, so a host can start writing
/// the file before it has read it: twelve bytes of header, then the
/// descriptor, then the bytes. Getting the widths or the byte order wrong
/// produces a 422 that says nothing useful, so they are pinned here.
@Suite struct GalleryImportSuite {

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
}
