import Foundation
import Testing
import MoldClientTesting
@testable import MoldClient

@MainActor struct SourceMediaProjectionTests {
    @Test func sourceTextAndKindDoNotExcludeTheirOwningOutputs() {
        let host = UUID()
        let owners = [PrintFixtures.entry("movie.mp4", host: host), PrintFixtures.entry("still.png", host: host)]
        let cache = SourceMediaProjection()
        var query = LibraryQuery()
        query.text = "original input"
        query.tokens = [.kind(.picture)]
        #expect(cache.owners(in: owners, query: query, revision: 1).count == 2)
        query.text = "another source name"
        #expect(cache.owners(in: owners, query: query, revision: 1).count == 2)
        #expect(cache.derivations == 1)
        query.tokens.append(.machine(id: UUID(), name: "Other"))
        #expect(cache.owners(in: owners, query: query, revision: 1).isEmpty)
        #expect(cache.derivations == 2)
    }
}
