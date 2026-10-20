import Foundation

/// Read-only metadata and input descriptors captured from one physical print.
/// No original media bytes or transport grants are downloaded or persisted here.
public struct VideoLibraryExtensionSource: Hashable, Sendable {
    public let source: VideoJobSource
    public let owner: VideoJobOwner
    public let title: String
    public let metadata: OutputMetadata
    public let inventory: RetainedSourceMedia.Inventory

    public var originalReferences: [VideoOriginalReference] { VideoOriginalReference.ordered(metadata: metadata, inventory: inventory) }

    public var prefilledDraft: RenderDraft {
        var draft = RenderDraft(reusing: metadata)
        let loras = draft.media.loras, fit = draft.media.sourceFit
        // Historical boundary images must not become the NEW movie's end.
        // Required original typed references are hydrated by source authority.
        draft.media = DraftMedia()
        draft.media.loras = loras; draft.media.sourceFit = fit
        draft.batchSize = 1
        return draft
    }

    public func authoring(seconds: String) -> VideoDurationAuthoring {
        .init(draft: .init(seconds: seconds, source: source), sourceOwner: .init(owner: owner, title: title))
    }

    /// Reload descriptors after launch without changing the user's new prompt,
    /// duration or replacement inputs. Captured archive identity stays pinned.
    @MainActor public static func restore(_ authoring: VideoDurationAuthoring, owner: VideoJobOwner,
                                         backend: any MoldBackend, current: @escaping @MainActor () -> Bool) async throws -> Self {
        guard let source = authoring.draft.source, authoring.sourceOwner?.matches(owner) == true,
              current(), !Task.isCancelled, backend.host == owner.host else { throw changed() }
        guard let print = try await backend.galleryPrint(source.filename), print.filename == source.filename, print.isVideo, print.trashedAt == nil else {
            throw SequenceDraftError.invalid("This video's original Library copy is unavailable. Restore it or start a new video.")
        }
        guard current(), !Task.isCancelled else { throw changed() }
        let inspected = try await inspect(filename: source.filename, title: authoring.sourceOwner?.title ?? print.displayName,
            metadata: print.metadata, owner: owner, backend: backend, current: current)
        if let expected = source.expectedArchiveIdentitySha256,
           inspected.source.expectedArchiveIdentitySha256 != expected { throw changed() }
        return inspected
    }

    @MainActor public static func inspect(filename: String, title: String, metadata: OutputMetadata,
                                         owner: VideoJobOwner, backend: any MoldBackend,
                                         current: @escaping @MainActor () -> Bool) async throws -> Self {
        func check() throws {
            guard !Task.isCancelled, current(), backend.host == owner.host else {
                throw SequenceDraftError.invalid("The video or machine changed. Open Extend Video again.")
            }
        }
        @MainActor func verifyInstallation() async throws {
            let status = try await backend.status()
            try check()
            guard status.instanceId == owner.instanceId else { throw changed() }
        }
        try check()
        let status = try await backend.status()
        try check()
        guard status.instanceId == owner.instanceId else { throw changed() }
        let capabilities = try await backend.capabilities()
        try check()
        guard capabilities.canExtendLibraryVideo else {
            throw SequenceDraftError.invalid("Update this machine before extending Library videos.")
        }
        let inventory = try await backend.retainedSourceMedia(for: filename)
        try check()
        guard inventory.members.count <= 64 else { throw MoldClientError.malformedResponse }
        switch inventory.availability {
        case .available:
            let offer = try await backend.retainedMediaTransferOffer(for: filename)
            try check()
            guard RetainedSourceMedia.validTransferDigest(offer.archiveIdentitySha256),
                  RetainedSourceMedia.validTransferDigest(offer.outputSha256), offer.outputSizeBytes > 0,
                  let recipe = offer.metadata, RetainedSourceMedia.mirrorMetadataMatches(recipe, metadata) else { throw changed() }
            try RetainedSourceMedia.validateTransferMembers(offer.members)
            let offeredIDs = offer.members.compactMap(\.memberId)
            guard offeredIDs.count == offer.members.count, Set(offeredIDs).count == offeredIDs.count,
                  Set(offeredIDs) == Set(inventory.members.map(\.memberId)),
                  offer.members.count == inventory.members.count,
                  offer.members.allSatisfy({ member in
                      // Inventory content hashes identify decoded typed media;
                      // transfer hashes identify its retained envelope bytes.
                      inventory.members.contains { $0.memberId == member.memberId
                          && $0.sizeBytes == member.sizeBytes && $0.role == member.role }
                  }) else { throw changed() }
            try await verifyInstallation()
            return .init(source: .init(filename: filename, expectedArchiveIdentitySha256: offer.archiveIdentitySha256),
                         owner: owner, title: title, metadata: recipe, inventory: inventory)
        case .unavailableLegacy:
            // An affirmative legacy inventory can open an editor. It does not
            // promise prefix eligibility; full server validation decides it.
            try await verifyInstallation()
            return .init(source: .init(filename: filename), owner: owner, title: title, metadata: metadata, inventory: inventory)
        case .unavailableAuth:
            throw SequenceDraftError.invalid("Reconnect with access to this video's original inputs before extending it.")
        case .unavailableMissingOrCorrupt:
            throw SequenceDraftError.invalid("This video's original input archive is missing or corrupt. Restore it before extending the video.")
        case .unknown:
            throw SequenceDraftError.invalid("This machine returned an unknown source state. Update or reconnect before extending the video.")
        }
    }

    private static func changed() -> SequenceDraftError {
        .invalid("This video's saved source changed. Refresh the Library and open Extend Video again.")
    }
}
