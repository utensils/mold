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
                      scope: LibraryScopeKind = .prints,
                      upscalers: [UpscalerChoice] = []) -> LibraryMenuPlan {
        LibraryMenuPlan(scope: scope, count: count, canReuse: canReuse,
                        canUpscale: canUpscale,
                        upscalers: UpscalePlan.options(upscalers))
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

    // MARK: - Choosing the upscaler

    private func installed(_ names: [String], downloaded: Bool = true) -> [UpscalerChoice] {
        names.map { UpscalerChoice(name: $0, isDownloaded: downloaded) }
    }

    /// Desktop lets a person choose; this app has no dialog, so the choice is
    /// where every other choice in this menu is.
    ///
    /// **Fails today**: the item is always plain and always the default.
    @Test func severalInstalledUpscalersBecomeASubmenu() throws {
        let items = plan(canUpscale: true,
                         upscalers: installed(["swinir:fp16", "real-esrgan-x4plus:fp16"])).items
        let bigger = try #require(items.first { $0.title == "Make Bigger" })
        #expect(bigger.isSubmenu)
        #expect(bigger.children.map(\.title)
            == ["real-esrgan-x4plus:fp16 (default)", "swinir:fp16"])
        #expect(bigger.children.first?.kind == .upscale(model: "real-esrgan-x4plus:fp16"))
        #expect(items.filter { $0.title == "Make Bigger…" }.isEmpty)
    }

    /// One is one item -- a submenu with a single row is a door onto a
    /// corridor -- and so is a machine whose models this app has not read,
    /// which sends no model name and lets the machine choose.
    @Test func oneOrNoneKnownStaysAPlainItem() {
        for choices in [installed(["real-esrgan-x4plus:fp16"]), []] {
            let items = plan(canUpscale: true, upscalers: choices).items
            let plain = items.filter { $0.kind == .upscale(model: nil) }
            let submenus = items.filter(\.isSubmenu)
            #expect(plain.map(\.title) == ["Make Bigger…"])
            #expect(submenus.isEmpty)
        }
    }

    /// An upscaler the machine does NOT have is not a choice, it is a
    /// download -- so it never reaches the menu.
    @Test func onlyDownloadedUpscalersAreOffered() {
        let mixed = [UpscalerChoice(name: "real-esrgan-x4plus:fp16", isDownloaded: true),
                     UpscalerChoice(name: "swinir:fp16", isDownloaded: false)]
        #expect(UpscalePlan.options(mixed).isEmpty, "one installed is not a choice")
        let items = plan(canUpscale: true, upscalers: mixed).items
        #expect(items.filter { $0.kind == .upscale(model: nil) }.count == 1)
    }
}
