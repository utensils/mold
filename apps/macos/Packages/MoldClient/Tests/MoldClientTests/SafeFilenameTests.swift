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
        ("evil%2F..%2Fx.png", .encodedSeparator),
        ("%2e%2e%2fevil.png", .encodedSeparator),
        ("%2Fabsolute.png", .encodedSeparator),
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
    }
}
