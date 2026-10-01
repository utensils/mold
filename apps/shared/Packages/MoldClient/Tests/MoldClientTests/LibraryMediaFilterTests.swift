import Foundation
import Testing
@testable import MoldClient

struct LibraryMediaFilterTests {
    @Test func multipleSearchKindsDoNotPretendToBeAllMedia() {
        var query = LibraryQuery()
        #expect(LibraryMediaFilter.selected(in: query) == .all)
        query.tokens = [.kind(.clip), .kind(.mesh)]
        #expect(LibraryMediaFilter.selected(in: query) == nil)
        #expect(LibraryMediaFilter.selected(in: LibraryMediaFilter.photos.applying(to: query)) == .photos)
    }

    @Test func kindsUseTheLibraryAuthorityAndPreserveOtherFilters() {
        var query = LibraryQuery()
        query.text = "turtle"
        query.sort = .oldest
        query.tokens = [.favorite, .tag("green"), .kind(.clip), .kind(.mesh)]
        for (filter, kind) in [(LibraryMediaFilter.photos, PrintKind.picture), (.videos, .clip), (.meshes, .mesh)] {
            let next = filter.applying(to: query)
            #expect(next.tokens == [.favorite, .tag("green"), .kind(kind)])
            #expect(next.text == query.text)
            #expect(next.sort == query.sort)
        }
        #expect(LibraryMediaFilter.all.applying(to: query).tokens == [.favorite, .tag("green")])
    }
}
