import Testing

@testable import MoldCompanion

/// The destinations are the Mac app's (`apps/macos/Sources/Mold/Shell`), in
/// its words and symbols -- DESIGN.md §3 and §4. A rename here that the Mac
/// did not make is the drift these pin.
struct DestinationTests {
    @Test func titlesAreTheMacAppsWords() {
        #expect(Destination.generate.title == "Generate")
        #expect(Destination.library.title == "Library")
        #expect(Destination.queue.title == "Queue")
        #expect(Destination.models.title == "Models")
        #expect(Destination.machines.title == "Machines")
    }

    @Test func symbolsAreTheMacAppsSymbols() {
        #expect(Destination.generate.symbol == "wand.and.sparkles")
        #expect(Destination.library.symbol == "photo.on.rectangle.angled")
        #expect(Destination.queue.symbol == "list.bullet.indent")
        #expect(Destination.models.symbol == "cube")
        #expect(Destination.machines.symbol == "server.rack")
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
