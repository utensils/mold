import MoldClient
import Testing
@testable import MoldCompanion

@MainActor @Test func libraryFamilyKeepsOneVisitThroughSearchAndShelves() {
    let router = AppRouter()
    var media = LibraryNewMedia()
    media.markSeen(["old.png"])
    router.selection = .go(.library)
    router.libraryVisit = media.beginVisit()
    media.markSeen(["old.png", "new.mp4"])
    router.selection = .search
    #expect(router.libraryVisit?.contains("new.mp4") == true)
    router.selection = .shelf(.all)
    #expect(router.libraryVisit?.contains("new.mp4") == true)
    router.selection = .go(.library)
    #expect(router.libraryVisit?.contains("new.mp4") == true)
    router.selection = .go(.generate)
    #expect(router.libraryVisit == nil)
}
