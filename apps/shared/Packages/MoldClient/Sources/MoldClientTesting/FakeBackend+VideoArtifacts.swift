import Foundation
import MoldClient

public extension FakeBackend {
    func videoArtifactBytes(filename: String, role: VideoArtifactRole, offset: UInt64,
                            sourceInstance: String, archive: String) async throws -> Data {
        try await respond("videoArtifactBytes", [filename, role, offset, sourceInstance, archive])
    }
    func beginVideoArtifacts(filename: String, artifacts: VideoJobArtifacts,
                             destinationInstance: String, archive: String) async throws -> VideoArtifactTransferStatus {
        try await respond("beginVideoArtifacts", [filename, artifacts, destinationInstance, archive])
    }
    func videoArtifactsStatus(filename: String, handle: String, destinationInstance: String,
                              archive: String) async throws -> VideoArtifactTransferStatus {
        try await respond("videoArtifactsStatus", [filename, handle, destinationInstance, archive])
    }
    func appendVideoArtifact(filename: String, handle: String, role: VideoArtifactRole, offset: UInt64, bytes: Data,
                             destinationInstance: String, archive: String) async throws -> VideoArtifactTransferStatus {
        try await respond("appendVideoArtifact", [filename, handle, role, offset, bytes, destinationInstance, archive])
    }
    func completeVideoArtifacts(filename: String, handle: String, destinationInstance: String,
                                archive: String) async throws -> VideoArtifactTransferStatus {
        try await respond("completeVideoArtifacts", [filename, handle, destinationInstance, archive])
    }
}
