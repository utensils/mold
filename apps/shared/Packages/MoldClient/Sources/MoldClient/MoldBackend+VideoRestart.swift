import Foundation

public protocol MoldParentRestartBackend: Sendable {
    func videoRestartOptions(id: String, sourceInstanceId: String) async throws -> ChainRestartOptions
    func exportVideoRestart(id: String, request: ChainRestartExportRequest) async throws -> VideoParentRestartExport
    func videoRestartSource(id: String, offsetBytes: UInt64, sourceInstanceId: String, sourceRef: String) async throws
        -> Data
    func beginVideoSourceUpload(id: String, specJSON: String, destinationInstanceId: String) async throws
        -> VideoSourceUploadStatus
    func appendVideoSourceUpload(id: String, offsetBytes: UInt64, bytes: Data, destinationInstanceId: String)
        async throws -> VideoSourceUploadStatus
    func completeVideoSourceUpload(id: String, destinationInstanceId: String) async throws -> VideoSourceUploadRef
    func validateVideoRestart(requestJSON: String, destinationInstanceId: String, sourceUpload: VideoSourceUploadRef?)
        async throws -> VideoJobPlan
    func createVideoJob(
        requestJSON: String, validationRevision: String, operationId: String, destinationInstanceId: String,
        sourceUpload: VideoSourceUploadRef?
    ) async throws -> VideoJob
}

extension MoldParentRestartBackend {
    public func videoRestartOptions(id: String, sourceInstanceId: String) async throws -> ChainRestartOptions {
        throw unsupportedRestart()
    }
    public func exportVideoRestart(id: String, request: ChainRestartExportRequest) async throws
        -> VideoParentRestartExport
    { throw unsupportedRestart() }
    public func videoRestartSource(id: String, offsetBytes: UInt64, sourceInstanceId: String, sourceRef: String)
        async throws -> Data
    { throw unsupportedRestart() }
    public func beginVideoSourceUpload(id: String, specJSON: String, destinationInstanceId: String) async throws
        -> VideoSourceUploadStatus
    { throw unsupportedRestart() }
    public func appendVideoSourceUpload(id: String, offsetBytes: UInt64, bytes: Data, destinationInstanceId: String)
        async throws -> VideoSourceUploadStatus
    { throw unsupportedRestart() }
    public func completeVideoSourceUpload(id: String, destinationInstanceId: String) async throws
        -> VideoSourceUploadRef
    { throw unsupportedRestart() }
    public func validateVideoRestart(
        requestJSON: String, destinationInstanceId: String, sourceUpload: VideoSourceUploadRef?
    ) async throws -> VideoJobPlan { throw unsupportedRestart() }
    public func createVideoJob(
        requestJSON: String, validationRevision: String, operationId: String, destinationInstanceId: String,
        sourceUpload: VideoSourceUploadRef?
    ) async throws -> VideoJob { throw unsupportedRestart() }
    private func unsupportedRestart() -> SequenceDraftError {
        .invalid("Update this machine before starting this saved video again.")
    }
}
