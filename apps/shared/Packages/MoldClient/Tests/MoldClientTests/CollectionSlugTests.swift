import Testing

@testable import MoldClient

/// Parity with `collection_slug`'s own test
/// (`crates/mold-core/src/organization.rs`): a collection created on the
/// phone must merge with the same name everywhere.
struct CollectionSlugTests {
    @Test func theServersOwnCases() {
        #expect(CollectionShelf.slug(for: "Hello, World!") == "hello-world")
        #expect(CollectionShelf.slug(for: "--") == nil)
        let slug = CollectionShelf.slug(for: String(repeating: "word ", count: 40))
        #expect((slug?.count ?? 99) <= 80)
        #expect(slug?.hasSuffix("-") == false)
    }

    @Test func nonASCIILettersAreSeparators() {
        #expect(CollectionShelf.slug(for: "Café Owls") == "caf-owls")
        #expect(CollectionShelf.slug(for: "  Owls 2  ") == "owls-2")
    }
}
