import Foundation
import CryptoKit
import MoldClient
@testable import Mold

// What a machine says it kept for a print, and the handle it mints for it.
// Same rules as every other route on this fake: unplanted THROWS, so a store
// that reaches for an inventory a test did not plant fails the test.
extension FakeBackend {
    func retainedMediaTransferOffer(for filename: String) async throws -> RetainedSourceMedia.TransferOffer {
        try record("retainedMediaTransferOffer")
        if let offer = retainedTransferOffers[filename] { return offer }
        if noRetainedMedia {
            let metadata = prints.first(where: { $0.filename == filename })?.metadata
                ?? zip(importedNames, importedItems).first(where: { $0.0 == filename })?.1.originalMetadata
            let bytes = zip(importedNames, importedMedia).first(where: { $0.0 == filename })?.1
                ?? mediaAnswers[filename] ?? mediaAnswer ?? Data()
            return RetainedSourceMedia.TransferOffer(archiveIdentitySha256: String(repeating: "a", count: 64), members: [],
                outputSha256: SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined(),
                outputSizeBytes: bytes.count, metadata: metadata)
        }
        throw notPlanted()
    }

    func importRetainedMedia(_ transfer: RetainedSourceMedia.Transfer, for filename: String) async throws {
        try record("importRetainedMedia")
        let offer = try await retainedMediaTransferOffer(for: filename)
        retainedTransfers.append((filename, transfer))
        retainedTransferOffers[filename] = RetainedSourceMedia.TransferOffer(
            archiveIdentitySha256: transfer.archiveIdentitySha256, members: transfer.members,
            outputSha256: offer.outputSha256, outputSizeBytes: offer.outputSizeBytes, metadata: offer.metadata)
    }

    func retainedSourceMedia(for filename: String) async throws
        -> RetainedSourceMedia.Inventory {
        try record("retainedSourceMedia")
        retainedInventoryRequests.append(filename)
        if let retainedInventoryResponder { return try await retainedInventoryResponder(filename) }
        guard let planted = retainedInventories[filename] else { throw notPlanted() }
        return planted
    }

    func retainedSourceMediaBytes(for filename: String, member memberId: String) async throws
        -> Data {
        try record("retainedSourceMediaBytes")
        retainedMemberRequests.append(memberId)
        guard let planted = retainedMemberBytes[memberId] else { throw notPlanted() }
        return planted
    }

    func retainedMediaReuseSession(
        for filename: String, members memberIds: [String], target: GenerateRequest
    ) async throws -> RetainedSourceMedia.ReuseSession {
        try record("retainedMediaReuseSession")
        // The REQUEST is recorded beside the handle, because the host hashes
        // it: a test's real question is whether the session was minted
        // against the request that was then submitted.
        retainedSessionRequests.append((filename, memberIds, target))
        // Planted per ATTEMPT, in order, so a test can pin what the SECOND
        // mint does -- `refuses` and `plantedErrors` are per route and cannot
        // tell one attempt from the next.
        if !retainedSessionFailures.isEmpty { throw retainedSessionFailures.removeFirst() }
        guard let planted = retainedSession else { throw notPlanted() }
        return planted
    }
}
