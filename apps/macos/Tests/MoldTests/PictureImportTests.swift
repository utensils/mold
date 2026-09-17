import AppKit
import ImageIO
import Testing
import UniformTypeIdentifiers

@testable import Mold

/// Conforming an imported picture to something the host can read
/// (finding 02#7). The bytes are built here rather than checked in: what is
/// being pinned is the CONTAINER rule, and a synthesised one-pixel image in
/// each container says it exactly.
@MainActor
struct PictureImportTests {
    private func bytes(_ type: UTType, width: Int = 8, height: Int = 4) throws -> Data {
        let rep = try #require(NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height,
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0))
        let data = NSMutableData()
        let destination = try #require(
            CGImageDestinationCreateWithData(data, type.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, try #require(rep.cgImage), nil)
        #expect(CGImageDestinationFinalize(destination))
        return data as Data
    }

    private func container(of data: Data) throws -> String {
        let source = try #require(CGImageSourceCreateWithData(data as CFData, nil))
        return try #require(CGImageSourceGetType(source) as String?)
    }

    @Test func aFormatTheEngineReadsIsPassedThroughUntouched() throws {
        for type in [UTType.png, .jpeg] {
            let original = try bytes(type)
            let picked = try PictureImport.conform(
                original, name: "a.\(type.preferredFilenameExtension ?? "x")",
                accepting: PictureImport.engineReadable)
            #expect(picked.data == original)
            #expect(picked.name.hasPrefix("a."))
        }
    }

    /// **Fails today**: the open panels offer HEIC -- the default format of
    /// every iPhone photograph -- and the bytes went straight to the host,
    /// which reads a PNG signature and then JPEG markers and nothing else.
    @Test func anIdentityPhotoTheServerCannotReadIsTranscodedToPNG() throws {
        let heic = try bytes(.heic)
        let picked = try PictureImport.conform(
            heic, name: "IMG_4021.HEIC", accepting: PictureImport.identityReadable)

        #expect(try container(of: picked.data) == UTType.png.identifier)
        #expect(picked.name == "IMG_4021.png")
        #expect(!picked.encoded.isEmpty)
        // Same picture, not a placeholder.
        let decoded = try #require(NSImage(data: picked.data))
        #expect(decoded.size.width == 8)
        #expect(decoded.size.height == 4)
    }

    /// The identity contract is stricter than the general one: a TIFF the
    /// `image` crate reads happily is still not something `identity.rs` can
    /// walk, so it transcodes there and passes through elsewhere.
    @Test func theTwoContractsAreReallyDifferent() throws {
        let tiff = try bytes(.tiff)
        let general = try PictureImport.conform(
            tiff, name: "scan.tiff", accepting: PictureImport.engineReadable)
        #expect(general.data == tiff)

        let identity = try PictureImport.conform(
            tiff, name: "scan.tiff", accepting: PictureImport.identityReadable)
        #expect(try container(of: identity.data) == UTType.png.identifier)
    }

    @Test func somethingThatIsNotAPictureIsRefusedByName() {
        #expect(throws: PictureImportError.self) {
            try PictureImport.conform(
                Data("not a picture".utf8), name: "notes.txt",
                accepting: PictureImport.engineReadable)
        }
    }

    @Test func aRefusalNamesTheFile() {
        let sentence = PictureImportError.undecodable(name: "notes.txt").errorDescription
        #expect(sentence?.contains("notes.txt") == true)
    }

    /// **Fails today**: three wells that were added after this type still
    /// hand-roll `Data(contentsOf:)` + `base64EncodedString()` in a `View`
    /// method on the main actor -- the exact stall 02#10 was about -- and
    /// `ControlPictureWell` offers `.heic` in its open panel and never
    /// transcodes it, so an iPhone photograph uploads whole and then 422s.
    ///
    /// A source scan, because the defect is the ABSENCE of a call: a unit
    /// test can only pin what a well does once it goes through the one door,
    /// and nothing stops the next well from opening its own. Over the WHOLE
    /// app, not just `Generate/`: the Library's import read a whole file on
    /// the main actor too, in a different folder.
    @Test func noWellReadsAFileItself() throws {
        let sources = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appending(path: "Sources/Mold")
        let enumerated = FileManager.default.enumerator(at: sources, includingPropertiesForKeys: nil)
        let files = (enumerated?.allObjects as? [URL] ?? []).filter { $0.pathExtension == "swift" }
        // A scan that finds no files passes for the wrong reason.
        #expect(files.count > 100, "the app's source directory was not found")
        var offences: [String] = []
        for file in files {
            // The two readers themselves, and the UAT seeds, which read a
            // path from the environment rather than a person's pick.
            guard !["PictureImport.swift", "MediaImport.swift"]
                .contains(file.lastPathComponent),
                !file.lastPathComponent.hasSuffix("+UAT.swift")
            else { continue }
            let text = try String(contentsOf: file, encoding: .utf8)
            for (number, line) in text.components(separatedBy: "\n").enumerated() {
                let code = line.trimmingCharacters(in: .whitespaces)
                guard code.contains("Data(contentsOf:"), !code.hasPrefix("//") else { continue }
                offences.append("\(file.lastPathComponent):\(number + 1)")
            }
        }
        #expect(offences == [], "a well reading a file on the main actor")
    }
}
