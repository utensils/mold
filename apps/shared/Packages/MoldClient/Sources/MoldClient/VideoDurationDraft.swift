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
    public var source: VideoJobSource?

    public init(seconds: String = "30", scenes: [Scene] = [], source: VideoJobSource? = nil) {
        self.seconds = seconds; self.scenes = scenes; self.source = source
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
        guard scenes.count <= 4_096 else { throw SequenceDraftError.invalid("Use at most 4,096 scenes in one video.") }
        var inputs: [VideoJobScene] = [], sum: UInt64 = 0
        for scene in scenes {
            guard !scene.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                  scene.prompt.utf8.count <= 77_000 else {
                throw SequenceDraftError.invalid("Every scene needs a prompt of at most 77,000 bytes.")
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

    private static func invalidDuration() -> SequenceDraftError {
        .invalid("Enter a positive duration in seconds, with up to three decimal places.")
    }
}
