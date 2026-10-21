import Foundation

extension GenerateRequest {
    /// Legacy preparation endpoints normalize prompts once. Protect only their
    /// nested wire request; the new video wrapper and editable draft stay literal.
    var protectedForLegacyPromptWire: Self {
        var wire = self
        wire.prompt = prompt.replacingOccurrences(of: "\\", with: "\\\\")
        wire.negativePrompt = negativePrompt?.replacingOccurrences(of: "\\", with: "\\\\")
        wire.originalPrompt = originalPrompt?.replacingOccurrences(of: "\\", with: "\\\\")
        if let transform = promptTransform {
            wire.promptTransform = .init(operation: transform.operation,
                rootPrompt: transform.rootPrompt?.replacingOccurrences(of: "\\", with: "\\\\"),
                sourcePrompt: transform.sourcePrompt.replacingOccurrences(of: "\\", with: "\\\\"),
                sourceKind: transform.sourceKind, task: transform.task, dimensions: transform.dimensions)
        }
        return wire
    }
}

public extension HTTPBackend {
    /// Applies only to the ordinary input-session transports preceding a
    /// positive-capability literal video parent. It never changes final JSON.
    func preparingLiteralVideoInputs() -> Self {
        var copy = self; copy.literalReferencePrompts = true; return copy
    }
}
