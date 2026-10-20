import Foundation
import MoldClient

public extension FakeBackend {
    func prepareVideoJob(_ request: VideoJobRequest) async throws -> PreparedVideoJob {
        try await respond("prepareVideoJob(_:)", [request])
    }
    func planVideoJob(_ request: VideoJobRequest) async throws -> VideoJobPlan {
        try await respond("planVideoJob(_:)", [request])
    }

    func validateVideoJob(_ request: VideoJobRequest) async throws -> VideoJobPlan {
        try await respond("validateVideoJob(_:)", [request])
    }
    func createVideoJob(_ request: VideoJobRequest, operationId: String, destinationInstanceId: String) async throws -> VideoJob {
        try await respond("createVideoJob(_:operationId:destinationInstanceId:)", [request, operationId, destinationInstanceId])
    }
    func videoJob(id: String) async throws -> VideoJob { try await respond("videoJob(id:)", [id]) }
    func listVideoJobs(limit: Int, cursor: String?) async throws -> VideoJobListing {
        try await respond("listVideoJobs(limit:cursor:)", [limit, cursor])
    }
    func cancelVideoJob(id: String, destinationInstanceId: String) async throws {
        let _: Void = try await respond("cancelVideoJob(id:destinationInstanceId:)", [id, destinationInstanceId])
    }
    func resumeVideoJob(id: String, destinationInstanceId: String) async throws -> VideoJob {
        try await respond("resumeVideoJob(id:destinationInstanceId:)", [id, destinationInstanceId])
    }
    func deleteVideoJob(id: String, destinationInstanceId: String) async throws {
        let _: Void = try await respond("deleteVideoJob(id:destinationInstanceId:)", [id, destinationInstanceId])
    }
    func videoJobEvents(id: String) -> AsyncThrowingStream<VideoJob, Error> { stream("videoJobEvents(id:)", [id]) }
}
