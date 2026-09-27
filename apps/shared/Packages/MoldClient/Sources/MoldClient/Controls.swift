import Foundation

/// Whether a control may be touched, is pinned, or should not be shown.
///
/// This is the server's answer, not a hint. A recipe that fixes its step count
/// has one correct value and offering a slider would be offering a lie.
public enum ControlMode: String, OpenWireEnum {
    case adjustable
    case fixed
    case hidden
    /// A mode added after this build. Treated as `hidden`: showing a control
    /// whose rules we don't know is worse than showing nothing.
    case unknown

    /// True when the app should render an interactive control.
    public var isAdjustable: Bool { self == .adjustable }

    /// True when the control should appear at all, in any form. A `fixed`
    /// control still has something to say -- its value and its `note`.
    public var isVisible: Bool { self == .adjustable || self == .fixed }
}

/// A whole-number control: steps, frames, view counts.
public struct IntegerControl: Codable, Hashable, Sendable {
    public let `default`: Int
    public let min: Int
    public let max: Int
    public let step: Int
    public let recommended: [Int]?
    public let mode: ControlMode
    /// Prose from the server explaining a constraint. It is rendered
    /// VERBATIM -- it is the only place a recipe can explain itself, and
    /// paraphrasing it would put this app's guess in the server's mouth.
    public let note: String?

    /// A `fixed` control with nothing to say renders nothing at all.
    public var hasSomethingToShow: Bool {
        mode.isAdjustable || (mode == .fixed && note != nil)
    }
}

/// A fractional control: guidance, thresholds, weights.
public struct FloatControl: Codable, Hashable, Sendable {
    public let `default`: Double
    public let min: Double
    public let max: Double
    public let step: Double
    public let mode: ControlMode
    public let note: String?

    public var hasSomethingToShow: Bool {
        mode.isAdjustable || (mode == .fixed && note != nil)
    }
}

/// An optional input a recipe either takes, requires, or refuses.
public struct FeatureControl: Codable, Hashable, Sendable {
    public let mode: ControlMode
    public let required: Bool
    public let reason: String?

    public var isAvailable: Bool { mode.isVisible }
}
