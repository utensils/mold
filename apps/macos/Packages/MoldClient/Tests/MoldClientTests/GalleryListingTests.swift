import Foundation
import Testing

@testable import MoldClient

/// A gallery index with a row this client refuses to hold.
///
/// **Fails today**: `galleryListing` decodes `[GalleryPrint]` straight and
/// nothing validates `filename`, so a hostile row is admitted whole -- and
/// once it IS validated, decoding the array as one would lose every other
/// print with it.
@Suite struct GalleryListingSuite {
    private func listing(_ filenames: [String]) throws -> GalleryListing {
        let rows = filenames.map { name in
            ["filename": name, "metadata": [:] as [String: Any], "timestamp": 1_000,
             "format": "png"] as [String: Any]
        }
        let data = try JSONSerialization.data(withJSONObject: rows)
        return try MoldJSON.decoder.decode(GalleryListing.self, from: data)
    }

    @Test func aTraversingRowIsDroppedAndTheRestOfTheListingSurvives() throws {
        let listed = try listing([
            "robot.png",
            "../../../../Users/you/Library/LaunchAgents/evil.plist",
            "turtle.png",
        ])

        #expect(listed.prints.map(\.filename) == ["robot.png", "turtle.png"])
        #expect(listed.rejected.map(\.reason) == [.separator])
    }

    @Test func anHonestListingKeepsEveryRow() throws {
        let listed = try listing(["a.png", "b.mp4", "c.glb"])
        #expect(listed.prints.count == 3)
        #expect(listed.rejected.isEmpty)
    }

    /// A row missing a required field is this app and that server disagreeing
    /// about the wire, not a hostile name -- it still fails the listing, the
    /// same answer as before.
    @Test func aMalformedRowStillFailsTheWholeListing() throws {
        let data = try JSONSerialization.data(withJSONObject: [["filename": "a.png"]])
        #expect(throws: (any Error).self) {
            try MoldJSON.decoder.decode(GalleryListing.self, from: data)
        }
    }

    @Test func aPrintDecodedOnItsOwnRefusesAnUnsafeName() throws {
        let data = try JSONSerialization.data(withJSONObject: [
            "filename": "../evil.png", "metadata": [:] as [String: Any], "timestamp": 1,
        ])
        #expect(throws: SafeFilename.Rejected(name: "../evil.png", reason: .separator)) {
            try MoldJSON.decoder.decode(GalleryPrint.self, from: data)
        }
    }
}
