import Foundation

/// Scalar editor state shared by both native apps. An empty scene list is the
/// ordinary single-prompt flow; the server alone chooses clip counts and trim.
public struct VideoDurationDraft: Codable, Hashable, Sendable {
    public struct Scene: Codable, Hashable, Sendable, Identifiable {
        public var id: UUID
        public var prompt: String
        public var seconds: String
        public init(id: UUID = UUID(), prompt: String, seconds: String) {
            self.id = id; self.prompt = prompt; self.seconds = seconds
        }
    }
    public var seconds: String
    public var scenes: [Scene]
    public var scenesEnabled: Bool
    public var source: VideoJobSource?

    public init(seconds: String, scenes: [Scene] = [], source: VideoJobSource? = nil, scenesEnabled: Bool = true) {
        self.seconds = seconds; self.scenes = scenes; self.source = source; self.scenesEnabled = scenesEnabled
    }

    /// Decimal arithmetic never rounds long requested durations through Double.
    public static func milliseconds(_ seconds: String) throws -> UInt64 {
        let text = seconds.trimmingCharacters(in: .whitespacesAndNewlines)
        let parts = text.split(separator: ".", omittingEmptySubsequences: false)
        guard (1...2).contains(parts.count), let whole = parts.first, !whole.isEmpty,
              whole.utf8.allSatisfy({ $0 >= 48 && $0 <= 57 }), let integer = UInt64(whole) else {
            throw invalidDuration()
        }
        let fraction = parts.count == 2 ? String(parts[1]) : ""
        guard fraction.count <= 3, parts.count == 1 || !fraction.isEmpty,
              fraction.utf8.allSatisfy({ $0 >= 48 && $0 <= 57 }) else { throw invalidDuration() }
        let (base, overflow) = integer.multipliedReportingOverflow(by: 1_000)
        let milli = UInt64(fraction.padding(toLength: 3, withPad: "0", startingAt: 0)) ?? 0
        let (result, additionOverflow) = base.addingReportingOverflow(milli)
        guard !overflow, !additionOverflow, result > 0 else { throw invalidDuration() }
        return result
    }

    public func request(generation: GenerateRequest) throws -> VideoJobRequest {
        let duration = try Self.milliseconds(seconds)
        let scenes = scenesEnabled ? scenes : []
        guard scenes.count <= 4_096 else { throw SequenceDraftError.invalid("Use at most 4,096 scenes in one video.") }
        var inputs: [VideoJobScene] = [], sum: UInt64 = 0
        for (index, scene) in scenes.enumerated() {
            guard !scene.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw SequenceDraftError.invalid("Describe scene \(index + 1).")
            }
            guard scene.prompt.utf8.count <= 77_000 else {
                throw SequenceDraftError.invalid("Scene \(index + 1)'s prompt is too long. Shorten it.")
            }
            let value = try Self.milliseconds(scene.seconds)
            let (next, overflow) = sum.addingReportingOverflow(value)
            guard !overflow else { throw Self.invalidDuration() }
            sum = next; inputs.append(.init(prompt: scene.prompt, durationMs: value))
        }
        guard inputs.isEmpty || sum == duration else {
            throw SequenceDraftError.invalid("Scene durations must add up to the video's requested duration.")
        }
        return .init(generation: generation, durationMs: duration, source: source, scenes: inputs.isEmpty ? nil : inputs)
    }

    /// Whole-frame recipe defaults use integer arithmetic even for malformed
    /// saved values, so restoring a draft cannot overflow or round via Double.
    public static func initialSeconds(frames: Int, fps: Int) throws -> String {
        guard frames > 0, fps > 0 else { throw invalidDuration() }
        let product = UInt64(frames).multipliedFullWidth(by: 1_000)
        let divisor = UInt64(fps)
        guard product.high < divisor else { throw invalidDuration() }
        let milliseconds = divisor.dividingFullWidth(product).quotient
        return seconds(milliseconds: max(1, milliseconds))
    }

    public static func seconds(milliseconds: UInt64) -> String {
        let whole = milliseconds / 1_000, fraction = milliseconds % 1_000
        guard fraction != 0 else { return String(whole) }
        var digits = String(1_000 + fraction).dropFirst()
        while digits.last == "0" { digits = digits.dropLast() }
        return "\(whole).\(digits)"
    }

    /// Scene mode displays its summed total. Invalid in-progress input leaves
    /// the prior total intact until the scene duration can be represented.
    @discardableResult public mutating func updateDurationFromScenes() -> Bool {
        guard scenesEnabled, !scenes.isEmpty else { return false }
        var total: UInt64 = 0
        for scene in scenes {
            guard let value = try? Self.milliseconds(scene.seconds) else { return false }
            let (next, overflow) = total.addingReportingOverflow(value)
            guard !overflow else { return false }
            total = next
        }
        seconds = Self.seconds(milliseconds: total)
        return true
    }

    private static func invalidDuration() -> SequenceDraftError {
        .invalid("Enter a positive duration in seconds, with up to three decimal places.")
    }
}
