import Foundation
import Testing

@testable import MoldClient

/// A filename off the wire, before this app makes a path out of it.
///
/// **Fails today**: there is no validator at all -- `GalleryPrint.filename`
/// goes straight into `appending(path:)` and `data.write(to:)`
/// (`PrintMaterializer.swift:49,65`) and into `removeItem(at:)`
/// (`LibraryActions.swift:95-97`) in an app that is deliberately not
/// sandboxed and speaks plain HTTP.
@Suite struct SafeFilenameSuite {

    @Test(arguments: [
        "robot.png",
        "flux-dev_20260917_120000_0.mp4",
        "a print with spaces.png",
        "Café ☕.png",
        "100%25 sure.png",
        // Percent-encoding is not decoded by anything between the wire and the
        // file system, so this is a legal single component -- refusing it made
        // the print vanish from the listing with only a private log line.
        "a%2Fb.png",
        "%5Cnot-a-separator.png",
        "%2e%2e%2fevil.png",
        "under_score-and.dots.glb",
    ])
    func anOrdinaryPrintNameIsKept(_ name: String) throws {
        #expect(SafeFilename.isSafe(name))
        #expect(try SafeFilename.validated(name) == name)
    }

    @Test(arguments: [
        ("", SafeFilename.Reason.empty),
        ("   ", .empty),
        ("../../../../Users/you/Library/LaunchAgents/evil.plist", .separator),
        ("..", .relative),
        (".", .relative),
        (".hidden.png", .hidden),
        ("dir/child.png", .separator),
        ("/etc/passwd", .separator),
        ("back\\slash.png", .separator),
        ("colon:name.png", .separator),
        ("nul\0byte.png", .controlCharacter),
        ("bell\u{07}.png", .controlCharacter),
        ("newline\n.png", .controlCharacter),
        ("..%2Fevil.png", .hidden),
    ])
    func aNameThatIsNotOneSafeComponentIsRefused(_ name: String,
                                                 _ reason: SafeFilename.Reason) {
        #expect(!SafeFilename.isSafe(name))
        #expect(throws: SafeFilename.Rejected(name: name, reason: reason)) {
            try SafeFilename.validated(name)
        }
    }

    @Test func aNameLongerThanOneComponentMayBeIsRefused() {
        let name = String(repeating: "a", count: 300) + ".png"
        #expect(!SafeFilename.isSafe(name))
        #expect(SafeFilename.isSafe(String(repeating: "a", count: 251) + ".png"))
    }

    // MARK: - Staying inside the directory

    @Test func aSafeNameResolvesInsideTheDirectoryItWasBuiltFrom() throws {
        let root = URL(filePath: "/tmp/mold-cache")
        let file = try #require(SafeFilename.url("robot.png", in: root))
        #expect(file.path(percentEncoded: false) == "/tmp/mold-cache/robot.png")
    }

    /// A symlink planted in the cache by anything else on this unsandboxed
    /// Mac: `write(to:)` follows one, so a destination that IS one is refused.
    @Test func aSymbolicLinkIsNotAFreshDestination() throws {
        let dir = FileManager.default.temporaryDirectory
            .appending(path: "mold-safe-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let target = dir.appending(path: "target.txt")
        try Data("x".utf8).write(to: target)
        let link = dir.appending(path: "robot.png")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)

        #expect(!SafeFilename.isFreshDestination(link))
        #expect(SafeFilename.isFreshDestination(dir.appending(path: "nothing-here.png")))
        #expect(SafeFilename.isFreshDestination(target))
    }

    /// A link at the DIRECTORY is resolved before containment is judged, or
    /// "inside the cache" is a statement about a path rather than a place.
    @Test func aSymlinkedDirectoryResolvesToWhereItActuallyPoints() throws {
        let base = FileManager.default.temporaryDirectory
            .appending(path: "mold-safe-\(UUID().uuidString)")
        let real = base.appending(path: "real")
        try FileManager.default.createDirectory(at: real, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let link = base.appending(path: "link")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: real)

        let file = try #require(SafeFilename.url("robot.png", in: link))
        #expect(file.deletingLastPathComponent().resolvingSymlinksInPath()
            == real.resolvingSymlinksInPath())
    }

    @Test func aTraversingNameResolvesToNothing() {
        let root = URL(filePath: "/tmp/mold-cache")
        #expect(SafeFilename.url("../escape.png", in: root) == nil)
        #expect(SafeFilename.url("/etc/passwd", in: root) == nil)
        #expect(SafeFilename.url("", in: root) == nil)
    }

    /// A prefix test would pass `/tmp/mold-cache-evil` for `/tmp/mold-cache`.
    /// The containment rule is about the PARENT, not about the string.
    @Test func aSiblingDirectoryIsNotInsideThisOne() throws {
        let file = try #require(SafeFilename.url("robot.png",
                                                 in: URL(filePath: "/tmp/mold-cache-evil")))
        #expect(!file.path(percentEncoded: false).hasPrefix("/tmp/mold-cache/"))
    }

    // MARK: - Folding a value that is not a name

    @Test func aVersionIsFoldedRatherThanRefused() {
        #expect(SafeFilename.folded("sha256:abc", fallback: "1") == "sha256-abc")
        #expect(SafeFilename.folded("../..", fallback: "1") == "-.-..")
        #expect(SafeFilename.folded("", fallback: "1700000000") == "1700000000")
        #expect(SafeFilename.folded(".", fallback: "1700000000") == "-")
        // A folded value is usually part of a longer component.
        #expect(SafeFilename.folded(String(repeating: "v", count: 400),
                                    fallback: "1", limit: 8) == "vvvvvvvv")
    }
}
