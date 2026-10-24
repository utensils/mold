import Foundation

/// How a durable duration parent -- or an authored multi-clip sequence --
/// reads as an ordinary Queue row.
///
/// One wording on every surface: `docs/contracts/video-job-rows.json` is the
/// cross-language fixture, `crates/mold-core/src/video_job_row.rs` the Rust
/// reference and `studio/lib/videoJobRow.ts` the web twin.
public struct VideoJobRow: Hashable, Sendable {
    public enum Symbol: String, Codable, Hashable, Sendable {
        case waiting, rendering, joining, stopping, paused, interrupted, failed, stopped, done
    }

    /// The facts a row is drawn from. `current` is false when the client can
    /// only show its last cached snapshot (its owner check refused controls).
    public struct Input: Hashable, Sendable {
        public var state: VideoJobState
        public var progressPhase: VideoJobProgressPhase?
        public var completedClips: UInt64
        public var clipCount: UInt64
        /// `nil` when the length is not known (an authored sequence); the
        /// detail then names only the clips.
        public var totalDurationMs: UInt64?
        public var canCancel: Bool
        public var current: Bool

        public init(state: VideoJobState, progressPhase: VideoJobProgressPhase?, completedClips: UInt64,
                    clipCount: UInt64, totalDurationMs: UInt64?, canCancel: Bool, current: Bool) {
            self.state = state; self.progressPhase = progressPhase; self.completedClips = completedClips
            self.clipCount = clipCount; self.totalDurationMs = totalDurationMs
            self.canCancel = canCancel; self.current = current
        }
    }

    public let caption: String
    public let detail: String
    public let symbol: Symbol
    /// Completed fraction for a determinate meter; only multi-clip work in
    /// progress has one.
    public let progress: Double?

    public init(caption: String, detail: String, symbol: Symbol, progress: Double?) {
        self.caption = caption; self.detail = detail; self.symbol = symbol; self.progress = progress
    }

    public init(_ input: Input) {
        let clips = max(input.clipCount, 1)
        let done = min(input.completedClips, clips)
        let multi = clips > 1
        func tally(_ verb: String) -> String { multi ? "\(verb) · \(done) of \(clips) clips done" : verb }
        let meter: Double? = multi ? Double(done) / Double(clips) : nil
        let caption: String, symbol: Symbol, progress: Double?
        switch input.state {
        case .queued: (caption, symbol, progress) = ("Waiting", .waiting, nil)
        case .running where !input.canCancel: (caption, symbol, progress) = ("Stopping", .stopping, nil)
        case .running:
            switch input.progressPhase {
            case .waiting? where done == 0: (caption, symbol, progress) = ("Waiting", .waiting, nil)
            case .joining?: (caption, symbol, progress) = (multi ? "Joining clips" : "Finishing", .joining, meter)
            // Rendering, a waiting later clip, an unknown future phase or an
            // older server's bare `running` all read as the movie rendering.
            default:
                (caption, symbol, progress) = multi
                    ? ("Rendering clip \(min(done + 1, clips)) of \(clips)", .rendering, meter)
                    : ("Rendering", .rendering, nil)
            }
        case .paused: (caption, symbol, progress) = (tally("Paused"), .paused, nil)
        case .interrupted: (caption, symbol, progress) = (tally("Interrupted"), .interrupted, nil)
        case .failed: (caption, symbol, progress) = ("Failed", .failed, nil)
        // A state this build has never heard of stays read-only and honest.
        case .unknown: (caption, symbol, progress) = ("Status unknown", .interrupted, nil)
        case .cancelled: (caption, symbol, progress) = ("Stopped", .stopped, nil)
        case .completed: (caption, symbol, progress) = ("Done", .done, nil)
        }
        self.caption = input.current ? caption : "Last known: \(caption)"
        let clipWords = Plural.count(Int(clamping: clips), of: "clip")
        if let total = input.totalDurationMs {
            detail = multi ? "\(Self.seconds(total)) video · \(clipWords)" : "\(Self.seconds(total)) video"
        } else {
            detail = clipWords
        }
        self.symbol = symbol
        self.progress = progress
    }

    /// A duration parent's row.
    public init(_ job: VideoJob, current: Bool) {
        self.init(Input(state: job.state, progressPhase: job.progressPhase, completedClips: job.completedClips,
                        clipCount: job.plan.clipCount, totalDurationMs: job.plan.totalDurationMs,
                        canCancel: job.canCancel, current: current))
    }

    /// An authored sequence's row, in the same words. `currentStage` is the
    /// zero-based clip being worked on, so it is also the count finished.
    public init(chain state: ChainJobState, executionPhase: String?, currentStage: Int, stageCount: Int,
                cancelling: Bool, current: Bool) {
        let mapped: VideoJobState = switch state {
        case .queued: .queued
        case .running: .running
        case .paused: .paused
        case .interrupted: .interrupted
        case .completed: .completed
        case .failed: .failed
        case .cancelled: .cancelled
        case .unknown: .unknown
        }
        let phase: VideoJobProgressPhase? = switch executionPhase {
        case "queued": .waiting
        case "finalizing": .joining
        case nil: nil
        default: .rendering
        }
        let count = UInt64(max(stageCount, 1))
        let done = state == .completed ? count : UInt64(max(currentStage, 0))
        self.init(Input(state: mapped, progressPhase: phase, completedClips: done, clipCount: count,
                        totalDurationMs: nil, canCancel: !cancelling && ChainJobRecoveryPolicy.canCancel(state: state),
                        current: current))
    }

    /// `14000` → `14 s`, `14375` → `14.4 s` (one decimal, trailing zero dropped).
    public static func seconds(_ milliseconds: UInt64) -> String {
        let tenths = milliseconds / 100 + (milliseconds % 100 >= 50 ? 1 : 0)
        return tenths % 10 == 0 ? "\(tenths / 10) s" : "\(tenths / 10).\(tenths % 10) s"
    }
}

/// "1 clip", "3 clips". Never "1 clips".
public enum Plural {
    public static func count(_ value: Int, of noun: String, plural: String? = nil) -> String {
        "\(value) \(value == 1 ? noun : plural ?? noun + "s")"
    }
}
