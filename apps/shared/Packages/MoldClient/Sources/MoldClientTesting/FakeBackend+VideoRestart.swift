import Foundation
import MoldClient

extension FakeBackend {
    public func videoRestartOptions(id: String, sourceInstanceId: String) async throws -> ChainRestartOptions {
        try await respond("videoRestartOptions(id:sourceInstanceId:)", [id, sourceInstanceId])
    }
    public func exportVideoRestart(id: String, request: ChainRestartExportRequest) async throws
        -> VideoParentRestartExport
    {
        try await respond("exportVideoRestart(id:request:)", [id, request])
    }
    public func videoRestartSource(id: String, offsetBytes: UInt64, sourceInstanceId: String, sourceRef: String)
        async throws -> Data
    {
        try await respond(
            "videoRestartSource(id:offsetBytes:sourceInstanceId:sourceRef:)",
            [id, offsetBytes, sourceInstanceId, sourceRef])
    }
    public func beginVideoSourceUpload(id: String, specJSON: String, destinationInstanceId: String) async throws
        -> VideoSourceUploadStatus
    {
        try await respond(
            "beginVideoSourceUpload(id:specJSON:destinationInstanceId:)", [id, specJSON, destinationInstanceId])
    }
    public func appendVideoSourceUpload(id: String, offsetBytes: UInt64, bytes: Data, destinationInstanceId: String)
        async throws -> VideoSourceUploadStatus
    {
        try await respond(
            "appendVideoSourceUpload(id:offsetBytes:bytes:destinationInstanceId:)",
            [id, offsetBytes, bytes, destinationInstanceId])
    }
    public func completeVideoSourceUpload(id: String, destinationInstanceId: String) async throws
        -> VideoSourceUploadRef
    {
        try await respond("completeVideoSourceUpload(id:destinationInstanceId:)", [id, destinationInstanceId])
    }
    public func validateVideoRestart(
        requestJSON: String, destinationInstanceId: String, sourceUpload: VideoSourceUploadRef?
    ) async throws -> VideoJobPlan {
        try await respond(
            "validateVideoRestart(requestJSON:destinationInstanceId:sourceUpload:)",
            [requestJSON, destinationInstanceId, sourceUpload])
    }
    public func createVideoJob(
        requestJSON: String, validationRevision: String, operationId: String, destinationInstanceId: String,
        sourceUpload: VideoSourceUploadRef?
    ) async throws -> VideoJob {
        try await respond(
            "createVideoJob(requestJSON:validationRevision:operationId:destinationInstanceId:sourceUpload:)",
            [requestJSON, validationRevision, operationId, destinationInstanceId, sourceUpload])
    }
}
