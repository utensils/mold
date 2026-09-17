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

    // MARK: - SourceImageWell.menuItems

    @Test func emptyWellOffersNoRemove() {
        #expect(SourceImageWell.menuItems(hasPicture: false) == ["Choose File…", "From Library…"])
    }

    @Test func filledWellAddsRemove() {
        #expect(SourceImageWell.menuItems(hasPicture: true) == ["Choose File…", "From Library…", "Remove"])
    }
}
