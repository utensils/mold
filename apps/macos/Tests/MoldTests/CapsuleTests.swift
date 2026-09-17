import Foundation
import MoldClient
import Testing

@testable import Mold

/// The Generate capsule's redesign (M8 design), on its PURE gates -- no view
/// needed: whether Stop is a plain button or a split menu over "Stop All
/// Queued" (decision 8), the Library picker's footer wording (decision 5),
/// and the source well's menu items (decision 5).
@MainActor
struct CapsuleTests {
    // MARK: - PromptPanel.stopControl

    @Test func noQueueIsAPlainStopButton() {
        #expect(PromptPanel.stopControl(queued: 0) == .button)
    }

    @Test func anyQueueDepthIsAMenu() {
        #expect(PromptPanel.stopControl(queued: 1) == .menu)
        #expect(PromptPanel.stopControl(queued: 5) == .menu)
    }

    // MARK: - LibraryPickerSheet.caption

    @Test func oneRowIsSingular() {
        #expect(LibraryPickerSheet.caption(count: 1) == "1 picture")
    }

    @Test func zeroAndSeveralRowsArePlural() {
        #expect(LibraryPickerSheet.caption(count: 0) == "0 pictures")
        #expect(LibraryPickerSheet.caption(count: 2) == "2 pictures")
    }

    // MARK: - The source well's menu
    //
    // ONE list now (`GenerateMenus.sourceWell`), rendered by the well's click
    // menu and by its contextual menu alike. `GenerateMenusTests` pins the
    // gating; these two keep the titles a person actually reads.

    @Test func emptyWellOffersNoRemove() {
        let items = GenerateMenus.sourceWell(
            hasPicture: false, canEditMask: true, canPaste: false)
        #expect(items.map(\.title) == ["Choose File…", "Choose from Library…"])
    }

    @Test func filledWellAddsRemove() {
        let items = GenerateMenus.sourceWell(
            hasPicture: true, canEditMask: false, canPaste: false)
        #expect(items.map(\.title) == ["Choose File…", "Choose from Library…", "Remove"])
    }
}
