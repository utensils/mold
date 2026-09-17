import Foundation
import Testing

@testable import MoldClient

/// What the Library offers about making a print bigger.
///
/// **Fails today**: `LibraryMenuPlan` has no `Make Bigger…` at all, so the
/// action exists on no surface.
@MainActor
struct LibraryMenuPlanUpscaleTests {

    private func plan(count: Int = 1, canUpscale: Bool, canReuse: Bool = true,
                      scope: LibraryScopeKind = .prints) -> LibraryMenuPlan {
        LibraryMenuPlan(scope: scope, count: count, canReuse: canReuse,
                        canUpscale: canUpscale)
    }

    private func titles(_ plan: LibraryMenuPlan) -> [String] {
        plan.items.filter { !$0.isSeparator }.map(\.title)
    }

    @Test func aMachineThatUpscalesOffersIt() {
        #expect(titles(plan(canUpscale: true)).contains("Make Bigger…"))
    }

    /// Absent, never disabled: a machine that does not advertise upscaling
    /// gets no row rather than an inert one.
    @Test func aMachineThatDoesNotIsNotOfferedIt() {
        #expect(!titles(plan(canUpscale: false)).contains("Make Bigger…"))
    }

    /// It sits with Use These Settings, the other item that makes a NEW
    /// print out of this one -- and above Copy, which acts on this one.
    @Test func itSitsWithTheOtherThingThatMakesANewPrint() throws {
        let shown = titles(plan(canUpscale: true))
        let reuse = try #require(shown.firstIndex(of: "Use These Settings"))
        let bigger = try #require(shown.firstIndex(of: "Make Bigger…"))
        let copy = try #require(shown.firstIndex(of: "Copy"))
        #expect(bigger == reuse + 1)
        #expect(bigger < copy)
    }

    /// One print at a time. The clip half is a durable job per print, and a
    /// selection of forty would queue a machine full of work from one click.
    @Test func aMultipleSelectionIsNotOfferedIt() {
        #expect(!titles(plan(count: 4, canUpscale: true)).contains("Make Bigger…"))
    }

    /// The trash offers Put Back and Delete Immediately and nothing else --
    /// making a deleted print bigger is not a thing to offer.
    @Test func aTrashedPrintIsNotOfferedIt() {
        #expect(!titles(plan(canUpscale: true, scope: .trash)).contains("Make Bigger…"))
    }

    /// A print that can be upscaled but not reused still gets the item, and
    /// the divider above it is not doubled by the one Use These Settings
    /// would have carried.
    @Test func itStandsOnItsOwnWithoutReuse() {
        let items = plan(canUpscale: true, canReuse: false).items
        #expect(items.map(\.title).contains("Make Bigger…"))
        #expect(!items.enumerated().contains { index, item in
            index > 0 && item.isSeparator && items[index - 1].isSeparator
        })
    }
}
