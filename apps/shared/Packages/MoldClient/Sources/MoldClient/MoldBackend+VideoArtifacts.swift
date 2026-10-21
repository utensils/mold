import Foundation

public protocol MoldVideoArtifactsBackend: Sendable {
    func videoArtifactBytes(filename: String, role: VideoArtifactRole, offset: UInt64,
                            sourceInstance: String, archive: String) async throws -> Data
    func beginVideoArtifacts(filename: String, artifacts: VideoJobArtifacts,
                             destinationInstance: String, archive: String) async throws -> VideoArtifactTransferStatus
    func videoArtifactsStatus(filename: String, handle: String, destinationInstance: String,
                              archive: String) async throws -> VideoArtifactTransferStatus
    func appendVideoArtifact(filename: String, handle: String, role: VideoArtifactRole, offset: UInt64, bytes: Data,
                             destinationInstance: String, archive: String) async throws -> VideoArtifactTransferStatus
    func completeVideoArtifacts(filename: String, handle: String, destinationInstance: String,
                                archive: String) async throws -> VideoArtifactTransferStatus
}

public extension MoldVideoArtifactsBackend {
    func videoArtifactBytes(filename: String, role: VideoArtifactRole, offset: UInt64,
                            sourceInstance: String, archive: String) async throws -> Data { throw unavailableArtifacts() }
    func beginVideoArtifacts(filename: String, artifacts: VideoJobArtifacts,
                             destinationInstance: String, archive: String) async throws -> VideoArtifactTransferStatus { throw unavailableArtifacts() }
    func videoArtifactsStatus(filename: String, handle: String, destinationInstance: String,
                              archive: String) async throws -> VideoArtifactTransferStatus { throw unavailableArtifacts() }
    func appendVideoArtifact(filename: String, handle: String, role: VideoArtifactRole, offset: UInt64, bytes: Data,
                             destinationInstance: String, archive: String) async throws -> VideoArtifactTransferStatus { throw unavailableArtifacts() }
    func completeVideoArtifacts(filename: String, handle: String, destinationInstance: String,
                                archive: String) async throws -> VideoArtifactTransferStatus { throw unavailableArtifacts() }
    private func unavailableArtifacts() -> MoldClientError {
        .http(status: 409, code: "VIDEO_ARTIFACT_TRANSFER_UNAVAILABLE", message: "Update this machine to copy all saved video data.")
    }
}
