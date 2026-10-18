import Foundation

public struct ClipLengthSelection: Sendable {
    public struct Option: Hashable, Sendable {
        public let frames: Int
        public let seconds: Double
    }
    public let fps: Int
    public let frames: Int
    public let options: [Option]
}

public extension FpsControl {
    /// Repair restored/edited rates before computing duration-derived bounds.
    func resolvedRate(_ requested: Int?) -> Int {
        switch self {
        case let .fixed(value): return max(value, 1)
        case let .adjustable(fallback, minimum, maximum, step):
            let lower = max(minimum, 1), upper = max(maximum, lower), stride = max(step, 1)
            let value = requested.flatMap { $0 > 0 ? $0 : nil } ?? max(fallback, lower)
            let clamped = min(max(value, lower), upper)
            let grid = lower + ((clamped - lower + stride / 2) / stride) * stride
            return min(grid, lower + ((upper - lower) / stride) * stride)
        case .unknown: return max(requested ?? value, 1)
        }
    }
}

public extension TemporalProfile {
    /// Whole-second choices on the accepted frame grid, plus the settled current
    /// custom length. The selected rate, rather than an output upscale, owns bounds.
    func lengthSelection(fps requestedFPS: Int?, frames requestedFrames: Int?, family: String?, model: String?,
                         sourceImage: SourceImageCapability?) -> ClipLengthSelection {
        let rate = fps.resolvedRate(requestedFPS)
        let bounds = lengthBounds(fps: rate, family: family, model: model, sourceImage: sourceImage)
        func settle(_ value: Int) -> Int { min(max(snap(value), bounds.min), bounds.max) }
        let current = settle(requestedFrames ?? frames.default)
        var choices: Set<Int> = [current]
        let seconds = bounds.max / rate
        if seconds >= 1 {
            for second in 1...seconds { choices.insert(settle(second * rate)) }
        }
        return .init(fps: rate, frames: current, options: choices.sorted().map {
            .init(frames: $0, seconds: Double($0) / Double(rate))
        })
    }
}
