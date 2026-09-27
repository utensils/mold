import Foundation

/// Frames per second, which some recipes pin and others let you choose.
///
/// Internally tagged on `mode`, so a fixed recipe carries a single `value`
/// while an adjustable one carries a whole range -- they are genuinely
/// different shapes, not one shape with unused fields.
public enum FpsControl: Codable, Hashable, Sendable {
    case fixed(value: Int)
    case adjustable(default: Int, min: Int, max: Int, step: Int)
    case unknown

    private enum CodingKeys: String, CodingKey {
        case mode, value, `default`, min, max, step
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        switch try container.decode(String.self, forKey: .mode) {
        case "fixed":
            self = .fixed(value: try container.decode(Int.self, forKey: .value))
        case "adjustable":
            self = .adjustable(
                default: try container.decode(Int.self, forKey: .default),
                min: try container.decode(Int.self, forKey: .min),
                max: try container.decode(Int.self, forKey: .max),
                step: try container.decodeIfPresent(Int.self, forKey: .step) ?? 1)
        default:
            self = .unknown
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case let .fixed(value):
            try container.encode("fixed", forKey: .mode)
            try container.encode(value, forKey: .value)
        case let .adjustable(def, min, max, step):
            try container.encode("adjustable", forKey: .mode)
            try container.encode(def, forKey: .default)
            try container.encode(min, forKey: .min)
            try container.encode(max, forKey: .max)
            try container.encode(step, forKey: .step)
        case .unknown:
            try container.encode("unknown", forKey: .mode)
        }
    }

    public var value: Int {
        switch self {
        case let .fixed(value): value
        case let .adjustable(def, _, _, _): def
        case .unknown: 24
        }
    }

    public var isAdjustable: Bool {
        if case .adjustable = self { return true }
        return false
    }
}

/// How long a clip is, for the families that make one.
public struct TemporalProfile: Codable, Hashable, Sendable {
    public let frames: IntegerControl
    /// Wan's grid is `4k+1`, so the frame count is an offset from a multiple
    /// of its step rather than a plain multiple.
    public let frameOffset: Int?
    public let fps: FpsControl
    public let maxDurationSeconds: Double?

    /// Seconds a given frame count comes to at this recipe's rate.
    public func duration(forFrames frames: Int) -> Double {
        Double(frames) / Double(max(fps.value, 1))
    }

    /// The nearest frame count the recipe will actually accept.
    ///
    /// Clamping matters: a family whose grid is `4k+1` refuses 120 and accepts
    /// 121, and sending the wrong one is a 422 rather than a rounded render.
    public func snap(_ requested: Int) -> Int {
        let step = max(frames.step, 1)
        let offset = frameOffset ?? 0
        let clamped = min(max(requested, frames.min), frames.max)
        guard step > 1 else { return clamped }
        let steps = ((clamped - offset) + step / 2) / step
        return min(max(steps * step + offset, frames.min), frames.max)
    }
}

/// Whether a recipe reads a still as conditioning.
public enum SourceImageCapability: String, OpenWireEnum {
    case unsupported
    case optional
    case required
    case unknown

    public var isSupported: Bool { self == .optional || self == .required }
}
