import Foundation

/// Duration parents own clip planning, one final Library print and recovery.
public protocol MoldVideoBackend: Sendable {
    func prepareVideoJob(_ request: VideoJobRequest) async throws -> PreparedVideoJob
    func planVideoJob(_ request: VideoJobRequest) async throws -> VideoJobPlan
    func validateVideoJob(_ request: VideoJobRequest) async throws -> VideoJobPlan
    func createVideoJob(_ request: VideoJobRequest, operationId: String, destinationInstanceId: String) async throws -> VideoJob
    func videoJob(id: String) async throws -> VideoJob
    func listVideoJobs(limit: Int, cursor: String?) async throws -> VideoJobListing
    func cancelVideoJob(id: String, destinationInstanceId: String) async throws
    func resumeVideoJob(id: String, destinationInstanceId: String) async throws -> VideoJob
    func deleteVideoJob(id: String, destinationInstanceId: String) async throws
    func videoJobEvents(id: String) -> AsyncThrowingStream<VideoJob, Error>
}

public extension MoldVideoBackend {
    func prepareVideoJob(_ request: VideoJobRequest) async throws -> PreparedVideoJob {
        guard request.generation.pendingReferenceInputs == nil else { throw unsupportedVideoJob() }
        return .init(request: request)
    }
    func planVideoJob(_ request: VideoJobRequest) async throws -> VideoJobPlan { throw unsupportedVideoJob() }
    func validateVideoJob(_ request: VideoJobRequest) async throws -> VideoJobPlan { throw unsupportedVideoJob() }
    func createVideoJob(_ request: VideoJobRequest, operationId: String, destinationInstanceId: String) async throws -> VideoJob {
        throw unsupportedVideoJob()
    }
    func videoJob(id: String) async throws -> VideoJob { throw unsupportedVideoJob() }
    func listVideoJobs(limit: Int, cursor: String?) async throws -> VideoJobListing { throw unsupportedVideoJob() }
    func cancelVideoJob(id: String, destinationInstanceId: String) async throws { throw unsupportedVideoJob() }
    func resumeVideoJob(id: String, destinationInstanceId: String) async throws -> VideoJob { throw unsupportedVideoJob() }
    func deleteVideoJob(id: String, destinationInstanceId: String) async throws { throw unsupportedVideoJob() }
    func videoJobEvents(id: String) -> AsyncThrowingStream<VideoJob, Error> {
        AsyncThrowingStream(bufferingPolicy: .bufferingNewest(StreamBuffering.latestOnly)) { $0.finish(throwing: unsupportedVideoJob()) }
    }
    private func unsupportedVideoJob() -> SequenceDraftError {
        .invalid("Update this machine before creating or recovering duration-based videos.")
    }
}

public struct VideoJobListing: Decodable, Hashable, Sendable {
    public let jobs: [VideoJob]
    public let hasMore: Bool
    public let nextCursor: String?
    private enum CodingKeys: String, CodingKey { case jobs, hasMore, nextCursor }
    public init(from decoder: any Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        jobs = try values.decode([VideoJob].self, forKey: .jobs)
        hasMore = try values.decode(Bool.self, forKey: .hasMore)
        nextCursor = try values.decodeIfPresent(String.self, forKey: .nextCursor)
        guard !hasMore || nextCursor?.isEmpty == false else {
            throw DecodingError.dataCorruptedError(forKey: .nextCursor, in: values, debugDescription: "More video jobs require a continuation cursor")
        }
    }
}
