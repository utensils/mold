import AppKit
import MoldClient
import MoldStyle
import SwiftUI
import Testing
@testable import Mold

@MainActor struct LibraryFilterTests {
    @Test func titlesFollowTheQueryAndCurrentHostNames() {
        let host = MoldHost(name: "workstation", baseURL: URL(string: "http://fixture")!)
        var query = LibraryQuery()
        #expect(LibraryMachineFilter.title(for: query, hosts: [host]) == "All Machines")
        query.tokens = [.machine(id: host.id, name: "Old name")]
        #expect(LibraryMachineFilter.title(for: query, hosts: [host]) == "workstation")
        #expect(LibraryMachineFilter.title(for: query, hosts: []) == "Old name")
        query.tokens.append(.machine(id: UUID(), name: "hal9000"))
        #expect(LibraryMachineFilter.title(for: query, hosts: [host]) == "2 Machines")
        #expect(LibraryMediaTypeFilter.title(for: query) == "All Media")
        for filter in LibraryMediaFilter.allCases {
            #expect(LibraryMediaTypeFilter.title(for: filter.applying(to: query)) == filter.title)
        }
        query.tokens += [.kind(.picture), .kind(.clip)]
        #expect(LibraryMediaTypeFilter.title(for: query) == "Multiple Media Types")
    }

    /// The toolbar supplies iconOnly. The filter label must still reserve
    /// visible text, rather than exposing the selection only to VoiceOver.
    @Test(arguments: ["All Machines", "workstation", "All Items", "Not in a Collection · FLUX.1", "Photos", "3D"])
    func labelKeepsVisibleTextUnderToolbarStyle(title: String) {
        let label = NSHostingView(rootView: LibraryFilterLabel(title: title, systemImage: "server.rack")
            .labelStyle(.iconOnly))
        let text = NSHostingView(rootView: Text(title))
        #expect(label.fittingSize.width > text.fittingSize.width + 12)
        #expect(label.fittingSize.height >= text.fittingSize.height)
    }

    /// A menu draws SwiftUI images as templates, which would paint every
    /// family dot the text colour. The dot must keep its family's colour.
    @Test func familyDotsKeepTheirColourInAMenu() throws {
        let image = FamilySwatch.menuImage(rgb: 0xC24CF6)
        #expect(!image.isTemplate)
        #expect(FamilySwatch.menuImage(rgb: 0xC24CF6) === image, "one bitmap per colour")
        let bitmap = try #require(image.cgImage(forProposedRect: nil, context: nil, hints: nil)
            .map(NSBitmapImageRep.init(cgImage:)))
        let centre = try #require(bitmap.colorAt(x: bitmap.pixelsWide / 2, y: bitmap.pixelsHigh / 2)?
            .usingColorSpace(.sRGB))
        #expect(abs(centre.redComponent - 0xC2 / 255.0) < 0.02)
        #expect(abs(centre.blueComponent - 0xF6 / 255.0) < 0.02)
    }

    @Test func theFilterTitleIsTheSharedSummary() {
        var query = LibraryFilter.notInCollection.applying(to: LibraryQuery())
        query = LibraryFamilies.toggling("flux", label: "FLUX.1", in: query)
        #expect(LibraryFilter.summary(for: query) == "Not in a Collection · FLUX.1")
        #expect(LibraryMediaTypeFilter.title(for: query) == "All Media")
    }
}
