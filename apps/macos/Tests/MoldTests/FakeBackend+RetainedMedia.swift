import Foundation
import MoldClient
@testable import Mold

// What a machine says it kept for a print, and the handle it mints for it.
// Same rules as every other route on this fake: unplanted THROWS, so a store
// that reaches for an inventory a test did not plant fails the test.
extension FakeBackend {

    func retainedSourceMedia(for filename: String) async throws
        -> RetainedSourceMedia.Inventory {
        try record("retainedSourceMedia")
        retainedInventoryRequests.append(filename)
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
        guard let planted = retainedSession else { throw notPlanted() }
        return planted
    }
}
