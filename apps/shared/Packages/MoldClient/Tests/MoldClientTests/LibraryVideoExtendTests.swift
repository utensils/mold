import Testing
@testable import MoldClient

struct LibraryVideoExtendTests {
    @Test func ordinaryEligibleVideoOffersExtendWithoutLegacyChainIdentity() {
        let plan = LibraryMenuPlan(scope: .prints, count: 1, name: "ordinary.mp4", canExtendVideo: true)
        #expect(plan.items.contains { $0.kind == .extendVideo && $0.title == "Extend Video…" })
        #expect(!plan.items.contains { $0.kind == .continueVideo })
        let trash = LibraryMenuPlan(scope: .trash, count: 1, canExtendVideo: true)
        #expect(!trash.items.contains { $0.kind == .extendVideo })
        let multiple = LibraryMenuPlan(scope: .prints, count: 2, canExtendVideo: true)
        #expect(!multiple.items.contains { $0.kind == .extendVideo })
    }
}
