import Foundation
import Testing

@testable import MoldClient

/// UAT 2026-09-17 #9: `is:mesh` typed into the search field matched nothing
/// -- the README promised real tokens and the field was plain text. The
/// vocabulary is pinned here, once, for both the suggestions and the chip a
/// Return commits.
///
/// **Fails today**: there is no syntax; `is:mesh` is three letters and a word.
struct LibrarySearchSyntaxTests {
    private let hal = UUID()
    private var machines: [(id: MoldHost.ID, name: String)] { [(hal, "hal9000"), (UUID(), "Workstation")] }
    private let tags = ["cat", "catalog", "dog"]

    @Test func aPrefixNamesItsField() {
        #expect(LibrarySearchSyntax.split("is:video").field == .kind)
        #expect(LibrarySearchSyntax.split("TAG: cat").term == "cat")
        #expect(LibrarySearchSyntax.split("on:hal9000").field == .machine)
        #expect(LibrarySearchSyntax.split("cat").field == nil)
        // Not a field: a word with a colon in it is still a word.
        #expect(LibrarySearchSyntax.split("note:x").field == nil)
    }

    @Test func aKindIsNamedInAPersonsWordsToo() {
        #expect(LibrarySearchSyntax.kind("video") == .clip)
        #expect(LibrarySearchSyntax.kind("Image") == .picture)
        #expect(LibrarySearchSyntax.kind("3d") == .mesh)
        #expect(LibrarySearchSyntax.kind("song") == nil)
    }

    @Test func aPrefixNarrowsTheSuggestionsToItsOwnField() {
        let kinds = LibrarySearchSyntax.suggestions(for: "is:me", machines: machines, tags: tags, applied: [])
        #expect(kinds == [.kind(.mesh)])
        let videos = LibrarySearchSyntax.suggestions(for: "is:vid", machines: machines, tags: tags, applied: [])
        #expect(videos == [.kind(.clip)])
        let tagged = LibrarySearchSyntax.suggestions(for: "tag:cat", machines: machines, tags: tags, applied: [])
        #expect(tagged == [.tag("cat"), .tag("catalog")])
        let on = LibrarySearchSyntax.suggestions(for: "on:hal", machines: machines, tags: tags, applied: [])
        #expect(on == [.machine(id: hal, name: "hal9000")])
        // No prefix: everything that fits, as before, and never a chip
        // already in the field.
        let all = LibrarySearchSyntax.suggestions(for: "ca", machines: machines, tags: tags, applied: ["tag:cat"])
        #expect(all == [.tag("catalog")])
        // A bare letter is a word: only a kind's own name, never its aliases
        // (`m` is not an offer of Clip by way of "movie").
        #expect(LibrarySearchSyntax.suggestions(for: "m", machines: [], tags: [], applied: []) == [.kind(.mesh)])
        #expect(LibrarySearchSyntax.suggestions(for: "fav", machines: machines, tags: tags, applied: []) == [.favorite])
    }

    @Test func returnCommitsExactlyOneThingOrSearchesTheWords() {
        #expect(LibrarySearchSyntax.committed("is:video", machines: machines, tags: tags) == .kind(.clip))
        #expect(LibrarySearchSyntax.committed("tag:Cat", machines: machines, tags: tags) == .tag("cat"))
        #expect(LibrarySearchSyntax.committed("on:HAL9000", machines: machines, tags: tags)
            == .machine(id: hal, name: "hal9000"))
        // A prefix over nothing the library holds, or a plain word, stays text.
        #expect(LibrarySearchSyntax.committed("tag:ca", machines: machines, tags: tags) == nil)
        #expect(LibrarySearchSyntax.committed("is:song", machines: machines, tags: tags) == nil)
        #expect(LibrarySearchSyntax.committed("cat", machines: machines, tags: tags) == nil)
    }
}
