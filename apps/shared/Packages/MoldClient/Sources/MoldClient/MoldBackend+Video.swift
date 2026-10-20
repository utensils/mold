import Foundation

/// Duration parents own clip planning, one final Library print and recovery.
public protocol MoldVideoBackend: Sendable {
    func planVideoJob(_ request: VideoJobRequest) async throws -> VideoJobPlan
    func validateVideoJob(_ request: VideoJobRequest) async throws -> VideoJobPlan
    func createVideoJob(_ request: VideoJobRequest, operationId: String, destinationInstanceId: String) async throws -> VideoJob
    func videoJob(id: String) async throws -> VideoJob
    func listVideoJobs(limit: Int, offset: Int) async throws -> VideoJobListing
    func cancelVideoJob(id: String, destinationInstanceId: String) async throws
    func resumeVideoJob(id: String, destinationInstanceId: String) async throws -> VideoJob
    func deleteVideoJob(id: String, destinationInstanceId: String) async throws
    func videoJobEvents(id: String) -> AsyncThrowingStream<VideoJob, Error>
}

public extension MoldVideoBackend {
    func planVideoJob(_ request: VideoJobRequest) async throws -> VideoJobPlan { throw unsupportedVideoJob() }
    func validateVideoJob(_ request: VideoJobRequest) async throws -> VideoJobPlan { throw unsupportedVideoJob() }
    func createVideoJob(_ request: VideoJobRequest, operationId: String, destinationInstanceId: String) async throws -> VideoJob {
        throw unsupportedVideoJob()
    }
    func videoJob(id: String) async throws -> VideoJob { throw unsupportedVideoJob() }
    func listVideoJobs(limit: Int, offset: Int) async throws -> VideoJobListing { throw unsupportedVideoJob() }
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

public struct VideoJobListing: Codable, Hashable, Sendable { public let jobs: [VideoJob] }
