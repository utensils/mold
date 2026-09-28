import Testing

@testable import MoldClient

/// The Shape control both apps draw: the recipe's own ladder, and a size that
/// is off it kept visible rather than silently snapped.
struct CanvasShapeTests {
    private let square = AspectGroup(id: "1:1", label: "Square", presets: [
        SizePreset(id: "a", width: 512, height: 512), SizePreset(id: "b", width: 1024, height: 1024),
    ])
    private let wide = AspectGroup(id: "3:2", label: "3:2", presets: [
        SizePreset(id: "c", width: 1216, height: 832),
    ])

    private func profile(domain: ResolutionDomain = .buckets, groups: [AspectGroup]?) -> ResolutionProfile {
        ResolutionProfile(domain: domain, alignment: 16, minWidth: nil, minHeight: nil, maxPixels: nil,
                          maxAxisPixels: nil, offBucket: nil, aspectGroups: groups)
    }

    @Test func aSizeOnTheLadderOffersItsGroup() {
        guard case let .menus(menus) = CanvasShape.resolve(profile(groups: [square, wide]), width: 1216, height: 832)
        else { Issue.record("expected menus"); return }
        #expect(menus.aspect == "3:2")
        #expect(menus.isOnLadder)
        #expect(menus.sizes.map(\.width) == [1216])
    }

    @Test func anOffLadderSizeStaysVisibleAndChosen() {
        guard case let .menus(menus) = CanvasShape.resolve(profile(groups: [square]), width: 900, height: 900)
        else { Issue.record("expected menus"); return }
        #expect(menus.aspect == "1:1")
        #expect(!menus.isOnLadder)
        #expect(menus.sizes.last == SizePreset(id: "900x900", width: 900, height: 900))
    }

    @Test func noLadderIsAFixedSizeAndNoCanvasIsHidden() {
        #expect(CanvasShape.resolve(profile(groups: nil), width: 1024, height: 768) == .fixed("1024 × 768"))
        #expect(CanvasShape.resolve(profile(domain: .none, groups: nil), width: 1, height: 1) == .hidden)
        #expect(CanvasShape.resolve(profile(domain: .sourceDriven, groups: nil), width: 1, height: 1) == .fromSource)
    }

    @Test func switchingAspectKeepsRoughlyTheArea() {
        #expect(CanvasShape.size(in: square, near: 1216, 832)?.width == 1024)
    }
}
