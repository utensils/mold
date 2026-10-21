import Foundation

extension VideoRestartController {
    func prepareParentRestart(
        source: VideoRestartTarget, sourceBackend: any MoldBackend, jobId: String,
        destination: VideoRestartTarget, backend: any MoldBackend,
        exportRequest: ChainRestartExportRequest,
        current: @escaping @MainActor () -> Bool
    ) async throws -> (raw: String, proof: String?, prefix: VideoSourceUploadRef?) {
        let exported = try await sourceBackend.exportVideoRestart(id: jobId, request: exportRequest)
        guard current(), !Task.isCancelled else { throw changed() }
        let raw = try exported.validated(for: exportRequest)
        let prefix = try await transferPrefix(
            exported, source: source, sourceBackend: sourceBackend,
            sourceJobId: jobId, destination: destination, backend: backend, current: current)
        guard current(), !Task.isCancelled else { throw changed() }
        progress = "Checking saved inputs…"
        let plan = try await backend.validateVideoRestart(
            requestJSON: raw,
            destinationInstanceId: destination.instanceId, sourceUpload: prefix)
        guard current(), !Task.isCancelled, !plan.validationRevision.isEmpty else { throw changed() }
        return (raw, plan.validationRevision, prefix)
    }

    /// At most one 4 MiB chunk is retained. Same-installation exports can
    /// already own an immutable private prefix and require no client relay.
    func transferPrefix(
        _ exported: VideoParentRestartExport, source: VideoRestartTarget,
        sourceBackend: any MoldBackend, sourceJobId: String,
        destination: VideoRestartTarget, backend: any MoldBackend,
        current: @escaping @MainActor () -> Bool
    ) async throws -> VideoSourceUploadRef? {
        guard let prefix = exported.prefix else { return nil }
        let sourceCaps = try await sourceBackend.capabilities()
        guard current(), !Task.isCancelled else { throw changed() }
        let destinationCaps = try await backend.capabilities()
        guard current(), !Task.isCancelled else { throw changed() }
        guard sourceCaps.canRestartVideoJob, destinationCaps.canRestartVideoJob else {
            throw SequenceDraftError.invalid("Update both machines before restarting this video.")
        }
        if let local = try exported.localPrefixUpload(destinationInstanceId: destination.instanceId) { return local }
        guard sourceCaps.canTransferVideoSource, destinationCaps.canTransferVideoSource else {
            throw SequenceDraftError.invalid("Update both machines before transferring a video's original prefix.")
        }
        let uploadID = UUID().uuidString.lowercased()
        progress = "Copying original video…"
        var staged = try await backend.beginVideoSourceUpload(
            id: uploadID, specJSON: prefix.uploadSpecJSON,
            destinationInstanceId: destination.instanceId)
        guard current(), !Task.isCancelled, staged.handle == uploadID, staged.offsetBytes <= prefix.sizeBytes else {
            throw changed()
        }
        while staged.offsetBytes < prefix.sizeBytes {
            guard current(), !Task.isCancelled, !staged.completed else { throw changed() }
            let offset = staged.offsetBytes
            let expected = Int(min(UInt64(4 * 1024 * 1024), prefix.sizeBytes - offset))
            let chunk = try await sourceBackend.videoRestartSource(
                id: sourceJobId, offsetBytes: offset,
                sourceInstanceId: source.instanceId, sourceRef: prefix.sourceRef)
            guard current(), !Task.isCancelled, chunk.count == expected else { throw changed() }
            staged = try await backend.appendVideoSourceUpload(
                id: uploadID, offsetBytes: offset, bytes: chunk,
                destinationInstanceId: destination.instanceId)
            guard current(), !Task.isCancelled, staged.handle == uploadID,
                staged.offsetBytes == offset + UInt64(expected)
            else { throw changed() }
            progress = "Copying original video: \(staged.offsetBytes) of \(prefix.sizeBytes) bytes"
        }
        let sealed = try await backend.completeVideoSourceUpload(
            id: uploadID, destinationInstanceId: destination.instanceId)
        guard current(), !Task.isCancelled, sealed.handle == uploadID,
            sealed.sha256 == prefix.sha256, sealed.sizeBytes == prefix.sizeBytes
        else { throw changed() }
        return sealed
    }
}
