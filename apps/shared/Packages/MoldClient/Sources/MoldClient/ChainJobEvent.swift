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
    case interrupted
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
    public let requestWarnings: [String]?
}

/// Full durable detail. Ordinary reads omit authored bytes; explicitly request
/// source media only for editing/export and never persist it in an activity cache.
public struct ChainJobDetail: Decodable, Hashable, Sendable {
    public let id: String
    public let state: ChainJobState
    public let model: String
    public let stageCount: Int
    public let currentStage: Int
    public let error: String?
    public let finalizes: [ChainJobFinalize]?
    public let stages: [ChainJobStageDetail]?
    public let script: ChainScript?
    public let retakes: [ChainRetakeRecord]?
    public let amends: [ChainAmendRecord]?
    public let ephemeral: Bool?
    public let executionPhase: String?
    public let cancelling: Bool?
    public let createdAtUnixMs: UInt64?
    public let updatedAtUnixMs: UInt64?
    public var latestFinalize: ChainJobFinalize? {
        guard var latest = finalizes?.last else { return nil }
        // Rust FinalizeRecord has no take field. The append-only array's
        // one-based index is the continuation take authority.
        latest.take = finalizes?.count
        return latest
    }
    public var galleryFilename: String? { latestFinalize?.galleryFilename }
}

public struct ChainJobFinalize: Decodable, Hashable, Sendable {
    public let output: String?
    public let galleryFilename: String?
    public let atUnixMs: UInt64?
    public var take: Int?
    public let stageSeeds: [String]?
}
public struct ChainJobStageDetail: Decodable, Hashable, Sendable {
    public let idx: Int
    public let state: String
    public let seed: UInt64?
    public let framesEmitted: Int?
    public let generationTimeMs: UInt64?
    public let hasPreview: Bool?
    public let hasMedia: Bool?
    public let cacheReady: Bool?
    public let error: String?

    private enum CodingKeys: String, CodingKey {
        case idx, state, seed, framesEmitted, generationTimeMs, hasPreview, hasMedia, cacheReady, error
    }
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        idx = try values.decode(Int.self, forKey: .idx)
        state = try values.decode(String.self, forKey: .state)
        // Deployed hosts use decimal strings; newer hosts may emit a JSON u64.
        // Decode directly to UInt64 so neither format passes through Double.
        if let text = try? values.decode(String.self, forKey: .seed) {
            guard !text.isEmpty, text.utf8.allSatisfy({ $0 >= 48 && $0 <= 57 }), let parsed = UInt64(text) else {
                throw DecodingError.dataCorruptedError(forKey: .seed, in: values, debugDescription: "Invalid unsigned seed")
            }
            seed = parsed
        } else {
            seed = try values.decodeIfPresent(UInt64.self, forKey: .seed)
        }
        framesEmitted = try values.decodeIfPresent(Int.self, forKey: .framesEmitted)
        generationTimeMs = try values.decodeIfPresent(UInt64.self, forKey: .generationTimeMs)
        hasPreview = try values.decodeIfPresent(Bool.self, forKey: .hasPreview)
        hasMedia = try values.decodeIfPresent(Bool.self, forKey: .hasMedia)
        cacheReady = try values.decodeIfPresent(Bool.self, forKey: .cacheReady)
        error = try values.decodeIfPresent(String.self, forKey: .error)
    }
}
public struct ChainJobListing: Decodable, Hashable, Sendable {
    public let jobs: [ChainJobDetail]
}

public struct ChainRetakeRecord: Decodable, Hashable, Sendable {
    public let stageIdx: Int
    public let mode: String
    public let oldSeed: String
    public let newSeed: String
    public let oldPrompt: String?
    public let newPrompt: String?
    public let atUnixMs: UInt64
}
public struct ChainAmendRecord: Decodable, Hashable, Sendable {
    public let atUnixMs: UInt64
    public let previousRequestJson: String
    public let preservedStages: Int
}
