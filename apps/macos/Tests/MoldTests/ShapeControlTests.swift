import Foundation
import MoldClient
import Testing

@testable import Mold

/// `ShapeControl.resolve` reads the same `ResolutionProfile` the size menu
/// already decodes; these fixtures are the real flux-schnell answer captured
/// from workstation (M8 design, decision 3).
@MainActor
struct ShapeControlTests {
    private static let fluxSchnell = """
    {"domain":"dynamic","alignment":16,"min_width":64,"min_height":64,"max_pixels":1800000,"max_axis_pixels":null,"off_bucket":null,
     "aspect_groups":[{"id":"1:1","label":"1:1","presets":[{"id":"768x768","width":768,"height":768,"tier":"recommended"},{"id":"1024x1024","width":1024,"height":1024,"tier":"recommended"}]},
                      {"id":"4:3","label":"4:3","presets":[{"id":"1024x768","width":1024,"height":768,"tier":"recommended"}]},
                      {"id":"3:4","label":"3:4","presets":[{"id":"768x1024","width":768,"height":1024,"tier":"recommended"}]},
                      {"id":"16:9","label":"16:9","presets":[{"id":"1024x576","width":1024,"height":576,"tier":"recommended"}]},
                      {"id":"9:16","label":"9:16","presets":[{"id":"576x1024","width":576,"height":1024,"tier":"recommended"}]}]}
    """

    private func profile(_ json: String) -> ResolutionProfile {
        try! MoldJSON.decoder.decode(ResolutionProfile.self, from: Data(json.utf8))
    }

    private func fluxSchnell() -> ResolutionProfile { profile(Self.fluxSchnell) }

    private func menus(_ presentation: ShapeControl.Presentation) -> ShapeControl.Shape? {
        guard case let .menus(shape) = presentation else { return nil }
        return shape
    }

    @Test func onLadderSizeReportsItsGroupAndBothPresets() {
        let shape = menus(ShapeControl.resolve(resolution: fluxSchnell(), width: 1024, height: 1024))
        #expect(shape?.aspect == "1:1")
        #expect(shape?.sizes.map(\.id) == ["768x768", "1024x1024"])
        #expect(shape?.isOnLadder == true)
    }

    @Test func offLadderSizeInAnExistingGroupAddsAnExtraLastRow() {
        let shape = menus(ShapeControl.resolve(resolution: fluxSchnell(), width: 1000, height: 1000))
        #expect(shape?.aspect == "1:1")
        #expect(shape?.sizes.map(\.id) == ["768x768", "1024x1024", "1000x1000"])
        #expect(shape?.isOnLadder == false)
    }

    @Test func aSizeWithNoOwnPresetStillFindsItsGroupByGcd() {
        let shape = menus(ShapeControl.resolve(resolution: fluxSchnell(), width: 1600, height: 900))
        #expect(shape?.aspect == "16:9")
        #expect(shape?.sizes.map(\.id) == ["1024x576", "1600x900"])
    }

    @Test func aSizeWithNoMatchingGroupAtAllShowsOnlyItself() {
        let shape = menus(ShapeControl.resolve(resolution: fluxSchnell(), width: 1000, height: 700))
        #expect(shape?.aspect == "10:7")
        #expect(shape?.sizes.map(\.id) == ["1000x700"])
    }

    @Test func sizeInGroupPicksThePresetNearestTheCurrentPixelCount() {
        let sixteenNine = fluxSchnell().aspectGroups!.first { $0.id == "16:9" }!
        let oneOne = fluxSchnell().aspectGroups!.first { $0.id == "1:1" }!

        #expect(ShapeControl.size(in: sixteenNine, nearWidth: 1024, height: 1024)?.id == "1024x576")
        // 1024x576 = 589,824px, exactly 768x768 -- nearer than 1024x1024's
        // 1,048,576.
        #expect(ShapeControl.size(in: oneOne, nearWidth: 1024, height: 576)?.id == "768x768")
    }

    @Test func sourceDrivenNeverDrawsMenus() {
        let profile = profile("""
        {"domain":"source-driven","aspect_groups":null}
        """)
        #expect(ShapeControl.resolve(resolution: profile, width: 512, height: 512) == .fromSource)
    }

    @Test func aRecipeWithNoCanvasHidesTheControlEntirely() {
        let profile = profile("""
        {"domain":"none","aspect_groups":null}
        """)
        #expect(ShapeControl.resolve(resolution: profile, width: 0, height: 0) == .hidden)
    }

    @Test func aDynamicRecipeWithNoAspectGroupsDrawsPlainDigits() {
        let profile = profile("""
        {"domain":"dynamic","aspect_groups":null}
        """)
        #expect(ShapeControl.resolve(resolution: profile, width: 512, height: 512) == .fixed("512 × 512"))
    }
}
