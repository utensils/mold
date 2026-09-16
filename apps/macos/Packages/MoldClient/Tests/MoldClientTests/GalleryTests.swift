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

@Test func searchIsFoldedAndEveryTokenMustMatch() throws {
    let raw = try #require(loadPrints().first)
    let item = LibraryEntry(host: MoldHost(id: UUID(), name: "hal9000", baseURL: URL(string: "http://h")!),
                            print: raw)
    let prompt = try #require(item.print.metadata.prompt)
    let word = try #require(prompt.split(separator: " ").first.map(String.init))

    #expect(item.matches(word.uppercased()))
    #expect(item.matches(""))
    // More words narrow rather than widen.
    #expect(!item.matches("\(word) definitelynotpresentxyz"))
}

@Test func searchCoversTheModelAndTheHostNotJustThePrompt() throws {
    let item = try #require(loadPrints().first { $0.metadata.model != nil })
    let library = LibraryEntry(host: MoldHost(id: UUID(), name: "plato", baseURL: URL(string: "http://h")!),
                               print: item)
    #expect(library.matches("plato"))
    #expect(library.matches(try #require(item.metadata.model)))
}

@Test func searchIgnoresDiacritics() {
    let folded = LibraryEntry.fold("Café Übung")
    #expect(folded == "cafe ubung")
}

@Test func aRowRebuiltFromANewPrintRefoldsItsSearchKey() {
    let host = MoldHost(id: UUID(), name: "plato", baseURL: URL(string: "http://h")!)
    let entry = PrintFixtures.entry("a.png", host: host.id, hostName: host.name, prompt: "owls")
    let rebuilt = entry.replacingPrint(PrintFixtures.print("a.png", prompt: "a brass helmet"))

    // The machine identity carries over untouched...
    #expect(rebuilt.hostID == entry.hostID)
    #expect(rebuilt.hostName == entry.hostName)
    // ...but the search key is folded again from the NEW print, not the old one.
    #expect(rebuilt.matches("helmet"))
    #expect(!rebuilt.matches("owls"))
}
