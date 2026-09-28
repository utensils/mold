import Foundation
import CoreGraphics
import MoldClient
import Testing

@testable import Mold

/// The Generate capsule's redesign (M8 design), on its PURE gates -- no view
/// needed: whether Stop is a plain button or a split menu over "Stop All
/// Queued" (decision 8), the Library picker's footer wording (decision 5),
/// and the source well's menu items (decision 5).
@MainActor
struct CapsuleTests {
    @Test func promptPanelLeavesRoomForWindowInsets() {
        #expect(PromptTuckLayout.availablePanelHeight(in: 900) == 860)
        #expect(PromptTuckLayout.availablePanelHeight(in: 300) == 260)
        #expect(PromptTuckLayout.availablePanelHeight(in: 30) == 0)
    }

    @Test func clipKeepsItsPresentationRatio() {
        #expect(VideoPlaybackLayout.aspectRatio(for: CGSize(width: 960, height: 960)) == 1)
        #expect(abs(VideoPlaybackLayout.aspectRatio(for: CGSize(width: 1920, height: 1080)) - 16.0 / 9.0) < 0.000_001)
        #expect(abs(VideoPlaybackLayout.aspectRatio(for: .zero) - 16.0 / 9.0) < 0.000_001)
    }

    @Test func mutePreferenceSurvivesPlaybackInstances() throws {
        let name = "video-playback-\(UUID())"
        let defaults = try #require(UserDefaults(suiteName: name))
        defer { defaults.removePersistentDomain(forName: name) }
        #expect(VideoPlaybackPreference.isMuted(in: defaults) == false)
        VideoPlaybackPreference.setMuted(true, in: defaults)
        #expect(VideoPlaybackPreference.isMuted(in: defaults))
        VideoPlaybackPreference.setMuted(false, in: defaults)
        #expect(VideoPlaybackPreference.isMuted(in: defaults) == false)
    }

    @Test func promptHistoryCyclesAndRestoresDraft() {
        var cycler = PromptHistoryCycler()
        cycler.setEntries(["newest", "middle", "oldest"])
        #expect(cycler.previous(from: "draft") == "newest")
        #expect(cycler.previous(from: "newest") == "middle")
        #expect(cycler.next(from: "middle") == "newest")
        #expect(cycler.next(from: "newest") == "draft")
        #expect(cycler.next(from: "draft") == nil)
    }

    @Test func recalledPromptDoesNotCarryPriorTransformProvenance() {
        var draft = RenderDraft()
        draft.prompt = "expanded words"
        draft.originalPrompt = "short words"
        draft.promptTransform = PromptTransformProvenance(
            operation: .expand, rootPrompt: "short words", sourcePrompt: "short words",
            task: .textToImage)
        PromptHistoryRecall.apply("saved prompt", to: &draft)
        #expect(draft.prompt == "saved prompt")
        #expect(draft.originalPrompt == nil)
        #expect(draft.promptTransform == nil)
    }

    @Test func promptHistoryKeepsMultilineCaretMovement() {
        let text = "first\nsecond"
        #expect(PromptHistoryCaret.isOnFirstLine(text, selection: NSRange(location: 2, length: 0)))
        #expect(!PromptHistoryCaret.isOnFirstLine(text, selection: NSRange(location: 8, length: 0)))
        #expect(PromptHistoryCaret.isOnLastLine(text, selection: NSRange(location: 8, length: 0)))
        #expect(!PromptHistoryCaret.isOnLastLine(text, selection: NSRange(location: 2, length: 0)))
    }

    @Test func generateToolbarChoicesFitTheNarrowDetail() {
        let detail = CGFloat(1_080 - 220) - TrailingColumn.width
        #expect(ModelPicker.maxToolbarWidth + RecipePicker.maxToolbarWidth + 20 <= detail)
    }

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
