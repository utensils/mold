import Foundation
import Testing
@testable import Mold

@MainActor
struct GenerationSubmissionFeedbackTests {
    @Test func pendingNeverClaimsAcceptanceAndLateRepliesCannotReplaceNewPress() {
        let feedback = GenerationSubmissionFeedback()
        let first = feedback.begin("Sending first", phase: .submitting)
        #expect(feedback.isPending)
        let second = feedback.begin("Checking second")
        feedback.update("Queued first", phase: .accepted, for: first)
        #expect(feedback.message == "Checking second")
        #expect(feedback.isPreparing)
        feedback.update("Queued second", phase: .accepted, for: second)
        #expect(!feedback.isPending)
        #expect(feedback.message == "Queued second")
        feedback.dismiss()
        feedback.update("Late failure", phase: .refused, for: second)
        #expect(feedback.message == nil)
    }

    @Test func promptResizesUpwardAndNeverExceedsAvailableSpace() {
        #expect(PromptEditorHeight.dragged(from: 72, translation: -100, available: 400) == 172)
        #expect(PromptEditorHeight.dragged(from: 72, translation: 100, available: 400) == 48)
        #expect(PromptEditorHeight.resolve(560, available: 120) == 120)
        #expect(PromptEditorHeight.resolve(560, available: 24) == 24)
        #expect(PromptEditorHeight.resolve(560, available: -1) == 0)
        #expect(PromptEditorHeight.resolve(.nan, available: 400) == 72)
        #expect(PromptEditorHeight.resolve(2000, available: 1000) == 560)
    }
}
