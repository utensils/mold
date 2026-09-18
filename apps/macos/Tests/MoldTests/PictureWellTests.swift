import AppKit
import Foundation
import ImageIO
import MoldClient
import Testing
import UniformTypeIdentifiers

@testable import Mold

/// One picture chooser for every picture well, and wells that say what they
/// are.
///
/// **Fails today**: there are four choosers. `SourceImageWell` has the whole
/// idiom (a click menu, the same list as a contextual menu, a drop, the
/// Library sheet, an open panel, Paste, a single-flight import, a failure
/// sentence beside the well); `ReferenceStrip` carries a second copy of half
/// of it with its own `LibraryPickerSheet`; `ControlPictureWell` a third with
/// no Library door and no menu at all; and `IdentityGroup` a fourth that is an
/// `NSOpenPanel` and nothing else -- no Library, no Paste, no menu -- so a
/// photograph already in the fleet cannot be used as a face without saving it
/// to this Mac first. And every one of them is an anonymous rounded square:
/// on an SD1.5 recipe the source well and the strip's empty add well sit side
/// by side with nothing saying which is which.
@MainActor
struct PictureWellTests {
    // MARK: - One chooser

    /// A source scan, because the defect is the ABSENCE of a call: a unit test
    /// can only pin what a well does once it goes through the one door, and
    /// nothing stops the next well from opening its own panel. The same shape
    /// as `MenuSurfaceTests.everyMenuIsAttachedThroughTheOneModifier`.
    @Test func everyPictureWellChoosesThroughTheOneComponent() throws {
        // `PictureSource` OWNS the panel -- it is the shared door, not a
        // fourth copy -- and `MediaWell` is the non-picture well (a clip,
        // audio, a keyframe still shown by its filename), which draws no
        // preview and deliberately stays apart.
        let panelDoors = ["PictureSource.swift", "MediaWell.swift"]
        // The sheet is the chooser's own, and nothing else may present it.
        let libraryDoors = ["PictureWell.swift"]

        var offences: [String] = []
        let files = try generateSources()
        // A scan that finds no files passes for the wrong reason.
        #expect(files.count > 50, "the Generate source directory was not found")
        for file in files {
            let name = file.lastPathComponent
            let text = try String(contentsOf: file, encoding: .utf8)
            for (number, line) in text.components(separatedBy: "\n").enumerated() {
                let code = line.trimmingCharacters(in: .whitespaces)
                guard !code.hasPrefix("//") else { continue }
                if code.contains("NSOpenPanel()"), !panelDoors.contains(name) {
                    offences.append("\(name):\(number + 1) opens its own panel")
                }
                if code.contains("LibraryPickerSheet("), !libraryDoors.contains(name) {
                    offences.append("\(name):\(number + 1) presents its own Library sheet")
                }
            }
        }
        #expect(offences == [], "a picture well choosing outside the one chooser")
    }

    /// Every well that can TAKE a picture offers the same three doors, in the
    /// same order, by the same names -- the owner's ask, and the reason they
    /// are one list rather than four.
    @Test func everyWellThatTakesAPictureOffersTheSameThreeDoors() {
        let quiet: [[GenerateMenus.Row]] = [
            GenerateMenus.sourceWell(hasPicture: false, canEditMask: false, canPaste: false),
            GenerateMenus.controlWell(hasPicture: false, canPaste: false),
            GenerateMenus.referenceAdd(canPaste: false),
            GenerateMenus.identityAdd(canPaste: false),
        ]
        for rows in quiet {
            #expect(rows.map(\.kind) == [.chooseFile, .chooseFromLibrary])
        }

        let pasteable: [[GenerateMenus.Row]] = [
            GenerateMenus.sourceWell(hasPicture: false, canEditMask: false, canPaste: true),
            GenerateMenus.controlWell(hasPicture: false, canPaste: true),
            GenerateMenus.referenceAdd(canPaste: true),
            GenerateMenus.identityAdd(canPaste: true),
        ]
        for rows in pasteable {
            #expect(rows.map(\.kind) == [.chooseFile, .chooseFromLibrary, .paste])
        }
    }

    /// A picture already staged is REPLACED from either door too -- an
    /// identity photograph's only replacement used to be a file.
    @Test func aStagedPictureIsReplacedFromEitherDoor() {
        #expect(GenerateMenus.identityPhoto().map(\.kind)
            == [.replacePhoto, .replaceFromLibrary, .removePhoto])
        #expect(GenerateMenus.referenceItem(index: 0, count: 1).map(\.kind)
            == [.replacePicture, .replaceFromLibrary, .removeReference])
        // The ends still carry only the move that exists.
        #expect(GenerateMenus.referenceItem(index: 1, count: 3).map(\.kind)
            == [.moveLeft, .moveRight, .replacePicture, .replaceFromLibrary, .removeReference])
    }

    /// The strip's BACKGROUND is about the whole strip. Add and Paste are the
    /// add well's own, and the add well is drawn whenever there is room, so
    /// repeating them here would be two menus offering the same door.
    @Test func theStripBackgroundIsAboutTheWholeStrip() {
        #expect(GenerateMenus.referenceStrip(count: 0).isEmpty)
        #expect(GenerateMenus.referenceStrip(count: 2).map(\.kind) == [.removeAllReferences])
    }

    // MARK: - A Library print is conformed to the well that asked for it

    /// A 1x1 lossless WebP. Checked in as bytes because ImageIO on this Mac
    /// READS WebP and cannot WRITE it, so there is no synthesising it the way
    /// `PictureImportTests` synthesises a HEIC.
    private static let webP = Data(
        base64Encoded: "UklGRhoAAABXRUJQVlA4TA0AAAAvAAAAEAcQERGIiP4HAA==")!

    /// **Fails today**: a print's bytes are base64'd and handed over whole
    /// (`PictureSource.bytes`), with no acceptance policy at all -- so the
    /// Library door hands the identity path a WebP, which reads a PNG
    /// signature and then JPEG markers and nothing else. A file through the
    /// same well is transcoded; the same picture through the Library is not.
    @Test func aLibraryPrintIsConformedToTheWellThatAskedForIt() async throws {
        let workstation = MoldHost(name: "workstation", baseURL: URL(string: "http://w")!)
        let fake = FakeBackend(host: workstation)
        fake.mediaAnswer = Self.webP
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let library = LibraryStore(hosts: hosts)
        let entry = LibraryEntry(host: workstation, print: FakeFixtures.print("a-face.webp"))
        library.items = [entry]

        let forIdentity = try await PictureSource.bytes(
            of: .print(entry.id), accepting: PictureImport.identityReadable,
            hosts: hosts, library: library)
        #expect(try container(of: forIdentity.data) == UTType.png.identifier)
        #expect(forIdentity.name == "a-face.png")

        // The general wells read WebP, so the same print passes through them
        // untouched -- the policy is the WELL's, not the file extension's.
        let forSource = try await PictureSource.bytes(
            of: .print(entry.id), accepting: PictureImport.engineReadable,
            hosts: hosts, library: library)
        #expect(forSource.data == Self.webP)
        #expect(forSource.name == "a-face.webp")
    }

    /// The identity wells' acceptance policy is the STRICT one, and it is the
    /// same one whichever door a photograph arrives through -- the add well,
    /// a staged well's Replace, or a drop on the group. Copying the general
    /// policy here is the one mistake that puts WebP in front of an encoder
    /// that reads PNG and JPEG and nothing else.
    @Test func theIdentityWellsAcceptLessThanEveryOtherWell() {
        #expect(IdentityGroup.accepting == PictureImport.identityReadable)
        #expect(IdentityGroup.accepting.isSubset(of: PictureImport.engineReadable))
        #expect(IdentityGroup.accepting != PictureImport.engineReadable)
        #expect(!IdentityGroup.accepting.contains(UTType.webP.identifier))
    }

    /// What the identity add well does with a picked picture, whichever door
    /// it came through -- the value, so the test needs no view.
    @Test func aPickedPictureIsStagedAsAnIdentityPhotograph() {
        let picked = ImportedPicture(encoded: "AAAA", name: "a-face.png", data: Data())
        var media = DraftMedia()

        media = IdentityGroup.staging(picked, in: media, maxPhotos: 2)
        #expect(media.identity?.photos.map(\.name) == ["a-face.png"])
        #expect(media.identity?.photos.first?.encoded == "AAAA")

        let second = ImportedPicture(encoded: "BBBB", name: "b.png", data: Data())
        media = IdentityGroup.staging(second, in: media, maxPhotos: 2)
        #expect(media.identity?.photos.map(\.name) == ["a-face.png", "b.png"])

        // The host's own limit, checked against the list this picture is
        // joining rather than the one the pick started with: a four-file drop
        // onto a two-photo group must not stage four.
        let third = ImportedPicture(encoded: "CCCC", name: "c.png", data: Data())
        media = IdentityGroup.staging(third, in: media, maxPhotos: 2)
        #expect(media.identity?.photos.map(\.name) == ["a-face.png", "b.png"])
    }

    /// Replace swaps ONE slot in place -- a remove-then-add sent the
    /// replacement to the end of the strip.
    @Test func replacingAStagedPhotographKeepsItsPlace() throws {
        var media = DraftMedia()
        for name in ["a.png", "b.png", "c.png"] {
            media = IdentityGroup.staging(
                ImportedPicture(encoded: name, name: name, data: Data()), in: media, maxPhotos: 4)
        }
        let middle = try #require(media.identity?.photos[1])

        let picked = ImportedPicture(encoded: "NEW", name: "new.png", data: Data())
        media = IdentityGroup.replacing(middle, with: picked, in: media)
        #expect(media.identity?.photos.map(\.name) == ["a.png", "new.png", "c.png"])

        // A photograph that is no longer staged -- removed while the panel was
        // open -- changes nothing rather than appending a stranger.
        let gone = IdentityPhoto(encoded: "gone", name: "gone.png")
        media = IdentityGroup.replacing(gone, with: picked, in: media)
        #expect(media.identity?.photos.map(\.name) == ["a.png", "new.png", "c.png"])
    }

    // MARK: - Wells that say what they are

    /// Opacity is not an explanation: a parked well says so in words.
    @Test func aParkedWellSaysSoInWords() {
        #expect(WellCaption.source(parked: false) == "Source")
        #expect(WellCaption.source(parked: true) == "Source (not used)")
        // One reference is a Reference; a cap above one -- or an absent cap,
        // which is UNBOUNDED -- is References.
        #expect(WellCaption.references(max: 1, parked: false) == "Reference")
        #expect(WellCaption.references(max: 4, parked: false) == "References")
        #expect(WellCaption.references(max: nil, parked: false) == "References")
        #expect(WellCaption.references(max: 4, parked: true) == "References (not used)")
    }

    /// The captions are drawn in a fixed column, so they are MEASURED rather
    /// than trusted: one that wraps grows the prompt bar by a line.
    @Test func everyCaptionFitsOnOneLineInTheColumnItIsDrawnIn() {
        let font = NSFont.preferredFont(forTextStyle: .caption2)
        let captions = [
            WellCaption.source(parked: false), WellCaption.source(parked: true),
            WellCaption.references(max: 1, parked: false),
            WellCaption.references(max: nil, parked: false),
            WellCaption.references(max: nil, parked: true),
            WellCaption.control, WellCaption.identityAdd,
        ]
        // Every caption the app can draw, not the three that happen to be short.
        #expect(captions.count > 6)
        let tooWide = captions.filter {
            NSAttributedString(string: $0, attributes: [.font: font]).size().width
                > WellCaption.width
        }
        #expect(tooWide == [], "a well caption that wraps and grows the prompt bar")
    }

    /// A captioned well is its square plus exactly ONE caption line. The
    /// constant is asserted against the font it is drawn in, in both
    /// directions -- too small clips the caption, too large is a second line's
    /// worth of empty bar.
    @Test func aCaptionedWellIsOneLineTallerThanItsSquare() {
        let line = NSFont.preferredFont(forTextStyle: .caption2).boundingRectForFont.height
        let captioned = WellCaption.height(under: PictureWell.standardSize)
        #expect(captioned >= PictureWell.standardSize + WellCaption.spacing + line)
        #expect(captioned < PictureWell.standardSize + WellCaption.spacing + 2 * line)
    }

    /// The two squares side by side are the owner's actual complaint: they
    /// must not draw the same glyph.
    @Test func theSourceWellAndTheStripsAddWellAreNotTheSameSquare() {
        #expect(SourceImageWell.placeholderGlyph != ReferenceStrip.addGlyph)
    }

    // MARK: - Helpers

    private func container(of data: Data) throws -> String {
        let source = try #require(CGImageSourceCreateWithData(data as CFData, nil))
        return try #require(CGImageSourceGetType(source) as String?)
    }

    private func generateSources() throws -> [URL] {
        let generate = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Tests/MoldTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // apps/macos
            .appending(path: "Sources/Mold/Generate")
        let enumerated = FileManager.default.enumerator(at: generate, includingPropertiesForKeys: nil)
        return (enumerated?.allObjects as? [URL] ?? []).filter { $0.pathExtension == "swift" }
    }
}
