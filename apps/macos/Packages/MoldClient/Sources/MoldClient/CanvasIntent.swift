import Foundation

/// Why the canvas holds the size it holds.
///
/// Port of `studio/lib/outputShape.ts:60-61`, and the point of #1166: the
/// intent is RECORDED WHEN SOMEBODY ACTS and never inferred afterwards.
/// Studio's earlier rule asked "is the canvas still the value I last
/// computed", which no surface can honour -- choosing a model writes that
/// model's default width and height BEFORE any source watcher runs, so the
/// comparison always failed and the canvas stopped following the source on
/// the first model switch (`sourceResolution.ts:50-59`).
public enum CanvasIntent: String, Hashable, Sendable {
    /// Follow the attached source, on the model's own preset ladder.
    case source
    /// Follow the source at its own aligned, capped size.
    case sourceExact = "source-exact"
    /// No source authority: the model's default canvas.
    case modelDefault = "model-default"
    /// Somebody chose this canvas. Nothing may move it.
    case manual

    /// True while the canvas is still following an attached source
    /// (`outputShape.ts:63-65`).
    public var followsSource: Bool { self == .source || self == .sourceExact }
}

public extension CanvasIntent {
    /// The canvas a newly attached (or re-fitted) source should move to, or
    /// `nil` to leave it where it is. Port of
    /// `resolveSourceCanvasTransition` (`sourceResolution.ts:61-71`).
    ///
    /// `sourceExact` is the source's own aligned size; `source` is the
    /// model-authored canvas nearest it. `replaced` means the source itself
    /// was swapped for another, which re-arms the automatic choice even from
    /// `model-default` -- unless the caller asks for the replacement to be
    /// preserved.
    func canvas(
        sourceExact: (width: Int, height: Int), automatic: (width: Int, height: Int),
        replaced: Bool, preserveReplacement: Bool = false
    ) -> (width: Int, height: Int)? {
        if self == .sourceExact { return sourceExact }
        if replaced { return preserveReplacement ? nil : automatic }
        return self == .source ? automatic : nil
    }
}
