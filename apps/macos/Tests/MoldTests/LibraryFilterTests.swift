import AppKit
import MoldClient
import SwiftUI
import Testing
@testable import Mold

@MainActor struct LibraryFilterTests {
    @Test func titlesFollowTheQueryAndCurrentHostNames() {
        let host = MoldHost(name: "plato", baseURL: URL(string: "http://fixture")!)
        var query = LibraryQuery()
        #expect(LibraryMachineFilter.title(for: query, hosts: [host]) == "All Machines")
        query.tokens = [.machine(id: host.id, name: "Old name")]
        #expect(LibraryMachineFilter.title(for: query, hosts: [host]) == "plato")
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
    @Test(arguments: ["All Machines", "plato", "All Media", "Photos", "Videos", "3D"])
    func labelKeepsVisibleTextUnderToolbarStyle(title: String) {
        let label = NSHostingView(rootView: LibraryFilterLabel(title: title, systemImage: "server.rack")
            .labelStyle(.iconOnly))
        let text = NSHostingView(rootView: Text(title))
        #expect(label.fittingSize.width > text.fittingSize.width + 12)
        #expect(label.fittingSize.height >= text.fittingSize.height)
    }
}
