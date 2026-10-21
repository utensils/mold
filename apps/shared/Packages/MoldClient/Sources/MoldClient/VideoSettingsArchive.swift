import Foundation

/// Read-only authority for one clicked physical Library copy. No output movie,
/// credential or reusable transport grant is downloaded or persisted here.
public struct VideoSettingsArchive: Sendable {
    public let filename: String
    public let owner: VideoJobOwner
    public let metadata: OutputMetadata
    public let inventory: RetainedSourceMedia.Inventory

    @MainActor public static func inspect(filename: String, metadata: OutputMetadata, owner: VideoJobOwner,
        backend: any MoldBackend, current: @escaping @MainActor () -> Bool) async throws -> Self {
        func check() throws {
            guard !Task.isCancelled, current(), backend.host == owner.host else { throw changed() }
        }
        try check()
        let initial = try await backend.status()
        try check(); guard initial.instanceId == owner.instanceId else { throw changed() }
        guard let print = try await backend.galleryPrint(filename), print.filename == filename,
              print.isVideo, print.trashedAt == nil,
              RetainedSourceMedia.mirrorMetadataMatches(print.metadata, metadata) else { throw changed() }
        try check()
        let inventory = try await backend.retainedSourceMedia(for: filename)
        try check()
        guard inventory.members.count <= RetainedSourceMedia.maxSessionMembers else { throw MoldClientError.malformedResponse }
        let canonical: OutputMetadata
        switch inventory.availability {
        case .available:
            let offer = try await backend.retainedMediaTransferOffer(for: filename)
            try check()
            guard RetainedSourceMedia.validTransferDigest(offer.archiveIdentitySha256),
                  RetainedSourceMedia.validTransferDigest(offer.outputSha256), offer.outputSizeBytes > 0,
                  let recipe = offer.metadata, RetainedSourceMedia.mirrorMetadataMatches(recipe, print.metadata) else { throw changed() }
            try RetainedSourceMedia.validateTransferMembers(offer.members)
            let identifiers = offer.members.compactMap(\.memberId)
            guard identifiers.count == offer.members.count, Set(identifiers).count == identifiers.count,
                  identifiers.count == inventory.members.count,
                  Set(identifiers) == Set(inventory.members.map(\.memberId)),
                  offer.members.allSatisfy({ member in inventory.members.contains {
                      $0.memberId == member.memberId && $0.role == member.role && $0.sizeBytes == member.sizeBytes
                  } }) else { throw changed() }
            canonical = recipe
        case .unavailableLegacy:
            canonical = print.metadata
        case .unavailableAuth:
            throw SequenceDraftError.invalid("Reconnect with access to this video's original input files before using its settings.")
        case .unavailableMissingOrCorrupt:
            throw SequenceDraftError.invalid("This video's original inputs are missing or corrupt. Restore its archive before using its settings.")
        case .unknown:
            throw SequenceDraftError.invalid("Update this machine before restoring its unknown input archive state.")
        }
        let final = try await backend.status()
        try check(); guard final.instanceId == owner.instanceId else { throw changed() }
        return .init(filename: filename, owner: owner, metadata: canonical, inventory: inventory)
    }

    @MainActor public func legacySequence(backend: any MoldBackend, current: @escaping @MainActor () -> Bool) async throws -> LegacySequenceSettingsReuse {
        func check() throws {
            guard !Task.isCancelled, current(), backend.host == owner.host else { throw Self.changed() }
        }
        let original = try LegacySequenceSettingsReuse(metadata: metadata)
        try check()
        guard inventory.members.count <= AuthoredChainRequest.retainedMediaMemberLimit else { throw MoldClientError.malformedResponse }
        var rawBytes = 0
        for member in inventory.members {
            guard member.sizeBytes >= 0, member.sizeBytes <= AuthoredChainRequest.inlineMediaLimitBytes - rawBytes else {
                throw SequenceDraftError.invalid("The original clip inputs exceed the authored-part media limit. Use smaller references or the original retained request.")
            }
            rawBytes += member.sizeBytes
        }
        var downloaded: [(member: RetainedSourceMedia.Member, bytes: Data)] = []
        for member in inventory.members {
            let bytes = try await backend.retainedSourceMediaBytes(for: filename, member: member.memberId)
            try check()
            guard bytes.count == member.sizeBytes else { throw MoldClientError.malformedResponse }
            downloaded.append((member, bytes))
        }
        let result = try original.materialized(downloaded)
        let status = try await backend.status()
        try check(); guard status.instanceId == owner.instanceId else { throw Self.changed() }
        return result
    }

    private static func changed() -> SequenceDraftError {
        .invalid("The video, draft or machine changed. Open Use Settings again.")
    }
}
