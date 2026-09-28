import Foundation
import MoldClient
import Testing

@testable import MoldCompanion

/// Small shell rules: ⌘+ / ⌘− walk the three tile sizes and stop at the
/// ends, and a sidebar machine says in words when it is not answering.
@MainActor
struct ShellBehaviourTests {
    @Test func tileSizesStepAndStopAtTheEnds() {
        #expect(TileSize.small.step(1) == .medium)
        #expect(TileSize.large.step(1) == .large)
        #expect(TileSize.small.step(-1) == .small)
        #expect(TileSize.large.step(-1) == .medium)
    }

    @Test func aSidebarMachineSaysWhenItIsNotAnswering() {
        #expect(RootView.badge(.down("Connection refused")) != nil)
        #expect(RootView.badge(.needsKey) != nil)
        #expect(RootView.badge(.unknown) == nil)
        #expect(RootView.badge(.checking) == nil)
    }
}
