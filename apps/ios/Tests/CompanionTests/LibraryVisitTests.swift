import Foundation
import MoldClient
import Testing
@testable import MoldCompanion

@MainActor struct LibraryVisitTests {
    @Test func leavingTabCapturesOldVisitBeforeNewTabAppears() throws {
        let router = AppRouter()
        let host = MoldHost(id: UUID(), name: "Fixture", baseURL: URL(string: "http://localhost:7680")!)
        let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(#"{"filename":"new.png","timestamp":1000,"size_bytes":1,"metadata":{"seed":7,"model":"flux"}}"#.utf8))
        let entry = LibraryEntry(host: host, print: print)
        router.selection = .go(.library)
        router.libraryVisit.capture([entry])
        // Opening individual media does not change the selected tab.
        router.selection = .go(.library)
        #expect(router.completedLibraryVisit.isEmpty)
        router.selection = .go(.generate)
        #expect(router.completedLibraryVisit == [entry.id])
        #expect(router.libraryVisit.finish().isEmpty)
    }

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
