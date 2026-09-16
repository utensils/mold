import Foundation
import Testing

@testable import MoldClient

private func loadPrints() throws -> [GalleryPrint] {
    try MoldJSON.decoder.decode([GalleryPrint].self, from: RepoFixtures.fixture("gallery.json"))
}

@Test func decodesPrintsCapturedFromALiveGallery() throws {
    let prints = try loadPrints()
    #expect(prints.count == 3)

    let first = try #require(prints.first)
    #expect(first.metadata.prompt != nil)
    #expect(first.metadata.model != nil)
    #expect(first.mediaVersion != nil)
    #expect(first.timestamp > 0)
}

@Test func tellsStillsFromVideo() throws {
    let prints = try loadPrints()
    #expect(prints.contains { $0.isVideo })
    #expect(prints.contains { !$0.isVideo && !$0.isMesh })
}

@Test func absentCollectionFieldsReadAsEmptyRatherThanMissing() throws {
    // Most rows omit `tags`, `favorite` and `collections` entirely -- serde
    // skips them when empty. A client that treated absence as an error, or as
    // anything other than "none", would break on nearly every print.
    let plain = try #require(loadPrints().first { $0.tags == nil })
    #expect(plain.tagList.isEmpty)
    #expect(plain.isFavorite == false)
}

@Test func aFavoriteIsCarriedWhenPresent() throws {
    #expect(try loadPrints().contains { $0.isFavorite })
}

@Test func printIdentityIsHostPlusFilenameNotFilenameAlone() {
    let a = UUID(), b = UUID()
    let onA = PrintID(host: a, filename: "mold-flux-1.png")
    let onB = PrintID(host: b, filename: "mold-flux-1.png")
    // Two machines generate names from the same scheme, so the same filename
    // on two hosts is two different prints.
    #expect(onA != onB)
}

@Test func thumbnailAndMediaURLsCarryTheRightQuery() throws {
    let urls = MediaURL(baseURL: URL(string: "http://host:7680")!)
    #expect(urls.thumbnail("a b.png", size: 512).absoluteString
        == "http://host:7680/api/gallery/thumbnail/a%20b.png?size=512")
    #expect(urls.media("a.png", trashed: true).absoluteString
        == "http://host:7680/api/gallery/image/a.png?view=trash")
}
