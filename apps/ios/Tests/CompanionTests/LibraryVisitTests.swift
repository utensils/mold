import Testing
@testable import MoldCompanion

@MainActor struct LibraryVisitTests {
    @Test func libraryFamilyIncludesSearchAndShelves() {
        let router = AppRouter()
        router.selection = .go(.library)
        #expect(router.isInLibrary)
        router.selection = .search
        #expect(router.isInLibrary)
        router.selection = .shelf(.all)
        #expect(router.isInLibrary)
        router.selection = .go(.generate)
        #expect(!router.isInLibrary)
    }
}
