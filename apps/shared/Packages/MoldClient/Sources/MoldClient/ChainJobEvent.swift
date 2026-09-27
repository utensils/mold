import Foundation

/// One frame of `GET /api/chain-jobs/{id}/events`.
///
/// Internally tagged on `type`, mirroring `ChainJobEvent`
/// (`studio/lib/api/chainTypes.ts:233-252`). An unrecognised `type` is
/// `.other` rather than a decode failure: the host is versioned independently
/// of this app and one unknown frame must not end a follow.
public enum ChainJobEvent: Decodable, Hashable, Sendable {
    case snapshot(ChainJobDetail)
    case stageStart(stage: Int)
    case denoiseStep(stage: Int, step: Int, total: Int)
    case stageDone(stage: Int)
    case finalizing(totalFrames: Int?)
    /// The stitched print. `galleryFilename` is what a client attaching after
    /// settlement fetches; `output` is a job-relative artifact and is NEVER
    /// fetchable (`chainTypes.ts:180-183`).
    case finalized(galleryFilename: String?)
    case stateChanged(state: ChainJobState, error: String?)
    case other

    private enum CodingKeys: String, CodingKey {
        case type, job, stageIdx, step, total, totalFrames, galleryFilename, state, error
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        switch try container.decode(String.self, forKey: .type) {
        case "snapshot":
            self = .snapshot(try container.decode(ChainJobDetail.self, forKey: .job))
        case "stage_start":
            self = .stageStart(stage: try container.decode(Int.self, forKey: .stageIdx))
        case "denoise_step":
            self = .denoiseStep(
                stage: try container.decode(Int.self, forKey: .stageIdx),
                step: try container.decode(Int.self, forKey: .step),
                total: try container.decode(Int.self, forKey: .total))
        case "stage_done":
            self = .stageDone(stage: try container.decode(Int.self, forKey: .stageIdx))
        case "finalizing":
            self = .finalizing(
                totalFrames: try container.decodeIfPresent(Int.self, forKey: .totalFrames))
        case "finalized":
            self = .finalized(
                galleryFilename: try container.decodeIfPresent(
                    String.self, forKey: .galleryFilename))
        case "state_changed":
            self = .stateChanged(
                state: try container.decode(ChainJobState.self, forKey: .state),
                error: try container.decodeIfPresent(String.self, forKey: .error))
        default:
            self = .other
        }
    }
}

public enum ChainJobState: String, OpenWireEnum {
    case queued
    case running
    case paused
    case completed
    case failed
    case cancelled
    case unknown

    /// Nothing further will arrive on this job's stream.
    public var isTerminal: Bool {
        self == .completed || self == .failed || self == .cancelled
    }
}

/// What `POST /api/chain-jobs` answers with.
public struct CreateChainJobResponse: Decodable, Hashable, Sendable {
    public let jobId: String
}

/// The job itself, as a `snapshot` frame carries it. Only the fields this app
/// reads -- the rest of `ChainJobDetail` is authored-sequence machinery no
/// GUI touches any more.
public struct ChainJobDetail: Decodable, Hashable, Sendable {
    public let id: String
    public let state: ChainJobState
    public let model: String
    public let stageCount: Int
    public let currentStage: Int
    public let error: String?
    public let finalizes: [ChainJobFinalize]?

    /// The gallery file the LAST finalize named -- what a client attaching
    /// after settlement fetches (`chainTypes.ts:189-192`).
    public var galleryFilename: String? {
        finalizes?.last(where: { $0.galleryFilename != nil })?.galleryFilename
    }
}

public struct ChainJobFinalize: Decodable, Hashable, Sendable {
    public let galleryFilename: String?
}
