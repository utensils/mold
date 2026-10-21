import Foundation

extension RenderDraft {
    /// The wrapper owns duration. Keep the ordinary frame value solely as the
    /// locator for opening/closing inputs, without altering the parked choice.
    public var durationGenerationDraft: Self {
        var copy = self
        copy.prefersAutomaticDuration = false
        return copy
    }
}
