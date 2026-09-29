import Testing

@testable import MoldCompanion

/// The destinations keep the Mac app's words and order. The phone uses simpler
/// symbols at its smaller size, pinned here with DESIGN.md §4.
struct DestinationTests {
    @Test func titlesAreTheMacAppsWords() {
        #expect(Destination.generate.title == "Generate")
        #expect(Destination.library.title == "Library")
        #expect(Destination.queue.title == "Queue")
        #expect(Destination.models.title == "Models")
        #expect(Destination.machines.title == "Machines")
    }

    @Test func symbolsAreReadableOnThePhone() {
        #expect(Destination.generate.symbol == "wand.and.sparkles")
        #expect(Destination.library.symbol == "square.grid.2x2")
        #expect(Destination.queue.symbol == "list.bullet")
        #expect(Destination.models.symbol == "cube")
        #expect(Destination.machines.symbol == "desktopcomputer")
    }

    /// ⌘1–⌘5 follow the Mac's order, Models included -- even though an
    /// iPhone's tab bar does not show Models.
    @Test func shortcutsFollowTheMacOrder() {
        #expect(Destination.allCases == [.generate, .library, .queue, .models, .machines])
        #expect(Destination.allCases.map(\.shortcut) == ["1", "2", "3", "4", "5"])
    }

    /// On iPhone Models lives under Machines (it follows a machine, as the Mac
    /// pane does), so the bar is four destinations plus Search.
    @Test func modelsIsSidebarOnly() {
        let inBar = Destination.allCases.filter(\.showsInTabBar)
        #expect(inBar == [.generate, .library, .queue, .machines])
    }
}
