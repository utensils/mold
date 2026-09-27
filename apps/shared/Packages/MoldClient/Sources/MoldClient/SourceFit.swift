import Foundation

/// How a source picture whose shape differs from the canvas is mapped onto it.
///
/// Port of `studio/lib/sourceFit.ts:9-24`. This is a CLIENT concept: the
/// fitting happens here, before the bytes ship, and the server records the
/// policy verbatim as opaque provenance (`types.rs:3268-3273`) so a Reuse can
/// put the crop controls back exactly as they were. That is also why the
/// `upscale-then-fit` case round-trips even though this app does not yet
/// offer it: a policy this build cannot author must still survive being read
/// back rather than being silently rewritten to something else.
public enum SourceFit: Hashable, Sendable {
    /// Keep the whole picture and ask the model to paint the added borders.
    case padRepaint
    /// Keep the whole picture and add borders without repainting them.
    case padFit
    /// Keep proportions and trim the edges that do not fit.
    case cropFill(alignX: SourceFitAlignX?, alignY: SourceFitAlignY?)
    /// Resize straight onto the conditioning shape; proportions may change.
    case lanczosResize
    /// Enhance a small picture first, then fit it. Never nested inside itself.
    indirect case upscaleThenFit(upscalerModel: String, fit: SourceFit)

    /// "Intentional default for every newly attached source image on every
    /// surface" (`sourceFit.ts:26-29`).
    public static let `default` = SourceFit.cropFill(alignX: .center, alignY: .center)

    public var mode: SourceFitMode {
        switch self {
        case .padRepaint: .padRepaint
        case .padFit: .padFit
        case .cropFill: .cropFill
        case .lanczosResize: .lanczosResize
        case .upscaleThenFit: .upscaleThenFit
        }
    }

    /// The fit that actually runs -- an `upscale-then-fit` defers to its own
    /// inner policy (`sourceFit.ts:180`).
    public var effective: SourceFit {
        if case let .upscaleThenFit(_, fit) = self { return fit }
        return self
    }

    /// Rewritten for a recipe that cannot ship a repaint mask: `pad-repaint`
    /// would paint pad bands the model can never repaint, so it becomes a
    /// centred `crop-fill`, which always fills the target. Port of
    /// `coerceSourceFitForMaskless` (`sourceFit.ts:267-280`).
    public func coercedForMaskless() -> SourceFit {
        switch self {
        case .padRepaint: .default
        case let .upscaleThenFit(model, fit) where fit.mode == .padRepaint:
            .upscaleThenFit(upscalerModel: model, fit: .default)
        default: self
        }
    }
}

/// The five modes, in the order every mold surface offers them
/// (`sourceFit.ts:35-65`).
public enum SourceFitMode: String, Hashable, Sendable, CaseIterable {
    case padRepaint = "pad-repaint"
    case cropFill = "crop-fill"
    case padFit = "pad-fit"
    case lanczosResize = "lanczos-resize"
    case upscaleThenFit = "upscale-then-fit"

    /// What the row says. The server never sees these -- they are this app's
    /// half of the same sentences studio shows.
    public var label: String {
        switch self {
        case .padRepaint: "Fit + repaint borders"
        case .cropFill: "Crop to fill"
        case .padFit: "Fit with borders"
        case .lanczosResize: "Stretch to fill"
        case .upscaleThenFit: "Upscale, then crop"
        }
    }

    public var help: String {
        switch self {
        case .padRepaint: "Keeps the whole picture and asks the model to paint the added borders."
        case .cropFill: "Keeps proportions and trims the edges that do not fit."
        case .padFit: "Keeps the whole picture and adds borders without repainting them."
        case .lanczosResize: "Resizes directly to the conditioning shape; proportions may change."
        case .upscaleThenFit: "Enhances a small picture first, then keeps proportions and trims edges."
        }
    }
}

public enum SourceFitAlignX: String, Hashable, Sendable, CaseIterable {
    case left, center, right
}

public enum SourceFitAlignY: String, Hashable, Sendable, CaseIterable {
    case top, center, bottom
}
