import Foundation

extension RetainedSourceMedia {
    struct ArtifactTransferAuthority: Hashable, Sendable {
        let sourceInstance: String
        let destinationInstance: String
    }
    static func prepareArtifacts(_ offer: TransferOffer, origin: any MoldBackend,
                                 target: any MoldBackend) async throws -> ArtifactTransferAuthority? {
        guard let artifacts = offer.metadata?.videoJob?.artifacts else { return nil }
        try artifacts.validate()
        guard offer.videoArtifactsComplete == true else {
            throw transferIncomplete("The original video's saved audio or clip history is unavailable or incomplete.")
        }
        let sourceCaps = try await origin.capabilities()
        let targetCaps = try await target.capabilities()
        guard sourceCaps.retainedMediaTransfer?.protocolVersion == 1,
              targetCaps.retainedMediaTransfer?.protocolVersion == 1,
              sourceCaps.retainedMediaTransfer?.videoArtifacts == true,
              targetCaps.retainedMediaTransfer?.videoArtifacts == true else {
            throw transferIncomplete("Update both machines to copy all saved video data.")
        }
        let source = try await origin.status()
        let destination = try await target.status()
        guard let sourceID = source.instanceId, UUID(uuidString: sourceID) != nil,
              let destinationID = destination.instanceId, UUID(uuidString: destinationID) != nil else {
            throw transferIncomplete("Reconnect both machines before copying all saved video data.")
        }
        return .init(sourceInstance: sourceID, destinationInstance: destinationID)
    }

    static func mirrorArtifacts(source: TransferOffer, destination: TransferOffer,
                                sourceFilename: String, targetFilename: String,
                                origin: any MoldBackend, target: any MoldBackend,
                                captured: ArtifactTransferAuthority?,
                                isCurrent: @Sendable () async -> Bool) async throws {
        guard let artifacts = source.metadata?.videoJob?.artifacts else { return }
        let authority: ArtifactTransferAuthority?
        if let captured { authority = captured }
        else { authority = try await prepareArtifacts(source, origin: origin, target: target) }
        guard let authority else { throw MoldClientError.malformedResponse }
        try artifacts.validate()
        func current() async throws {
            try Task.checkCancellation()
            guard await isCurrent() else { throw CancellationError() }
        }
        try await current()
        var status = try await target.beginVideoArtifacts(filename: targetFilename, artifacts: artifacts,
            destinationInstance: authority.destinationInstance, archive: destination.archiveIdentitySha256)
        let handle = status.handle
        func checked(_ value: VideoArtifactTransferStatus) throws {
            guard validTransferDigest(handle), value.handle == handle, value.archiveIdentitySha256 == destination.archiveIdentitySha256,
                  value.offsets.provenance <= artifacts.provenance.sizeBytes,
                  (artifacts.pcm == nil ? value.offsets.pcm == nil : value.offsets.pcm != nil),
                  value.offsets.pcm ?? 0 <= artifacts.pcm?.file.sizeBytes ?? 0 else {
                throw MoldClientError.malformedResponse
            }
            if value.complete {
                guard value.offsets.provenance == artifacts.provenance.sizeBytes,
                      value.offsets.pcm == artifacts.pcm?.file.sizeBytes else { throw MoldClientError.malformedResponse }
            }
        }
        try await current(); try checked(status)
        if status.complete { return }
        let files: [(VideoArtifactRole, VideoArtifactFile)] = [(.provenance, artifacts.provenance)]
            + (artifacts.pcm.map { [(.pcm, $0.file)] } ?? [])
        for (role, file) in files {
            var offset = role == .provenance ? status.offsets.provenance : status.offsets.pcm ?? 0
            while offset < file.sizeBytes {
                try await current()
                let bytes = try await origin.videoArtifactBytes(filename: sourceFilename, role: role, offset: offset,
                    sourceInstance: authority.sourceInstance, archive: source.archiveIdentitySha256)
                try await current()
                guard !bytes.isEmpty, bytes.count <= 4 * 1024 * 1024,
                      UInt64(bytes.count) <= file.sizeBytes - offset else { throw MoldClientError.malformedResponse }
                let previous = status
                status = try await target.appendVideoArtifact(filename: targetFilename, handle: handle, role: role,
                    offset: offset, bytes: bytes, destinationInstance: authority.destinationInstance,
                    archive: destination.archiveIdentitySha256)
                try await current(); try checked(status)
                let next = offset + UInt64(bytes.count)
                guard role == .provenance ? (status.offsets.provenance == next && status.offsets.pcm == previous.offsets.pcm)
                    : (status.offsets.pcm == next && status.offsets.provenance == previous.offsets.provenance) else {
                    throw MoldClientError.malformedResponse
                }
                offset = next
            }
        }
        try await current()
        status = try await target.completeVideoArtifacts(filename: targetFilename, handle: handle,
            destinationInstance: authority.destinationInstance, archive: destination.archiveIdentitySha256)
        try await current(); try checked(status)
        guard status.complete else { throw MoldClientError.malformedResponse }
    }
}
