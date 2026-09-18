import Foundation

// Reading a print's retained private conditioning media, and minting the
// one-use handle that lets the host hydrate a request from it.
public extension HTTPBackend {

    /// What this host retained for one print.
    ///
    /// A keyed host rejects the probe in MIDDLEWARE, before the handler can
    /// answer -- that IS the auth state, and the one case this wording is
    /// about, so it is reported rather than thrown for a caller's `catch` to
    /// swallow into silence (`gallerySourceMedia.ts:54-64`). Everything else
    /// still throws, so the server stays the only authority on the other
    /// three states.
    func retainedSourceMedia(
        for filename: String
    ) async throws -> RetainedSourceMedia.Inventory {
        do {
            return try await get(retainedSourceMediaPath(filename))
        } catch MoldClientError.unauthorized {
            return RetainedSourceMedia.Inventory(availability: .unavailableAuth)
        }
    }

    /// One retained file's original bytes. Bounded: the host serves at most
    /// 512 MiB for a member and a clip can be most of that.
    func retainedSourceMediaBytes(
        for filename: String, member memberId: String
    ) async throws -> Data {
        let data = try await bytes(for: request(
            retainedSourceMediaPath(filename) + "/\(escaped(memberId))"))
        return try ResponseCeiling.checked(
            data, ceiling: ResponseCeiling.media, what: "retained source media")
    }

    /// Mints the handle for a SAME-HOST reuse.
    ///
    /// The host hashes `target_request`, so the handle is bound to the exact
    /// request about to be submitted: any later edit to a hydrated role makes
    /// it a `RETAINED_MEDIA_REUSE_SCOPE_MISMATCH` and it has to be minted
    /// again. Never logged, never persisted, never put in a URL.
    func retainedMediaReuseSession(
        for filename: String, members memberIds: [String], target: GenerateRequest
    ) async throws -> RetainedSourceMedia.ReuseSession {
        try await post(retainedSourceMediaPath(filename) + "/reuse-sessions",
                       body: ReuseSessionBody(targetRequest: target, memberIds: memberIds))
    }
}

/// The body of `POST …/reuse-sessions` (`CreateReuseSessionRequest`).
struct ReuseSessionBody: Encodable {
    let targetRequest: GenerateRequest
    let memberIds: [String]
}

/// Path construction, split out so a test can pin it without a network call
/// -- the `historyPath` / `transferExportPath` precedent.
extension HTTPBackend {
    func retainedSourceMediaPath(_ filename: String) -> String {
        "/api/gallery/source-media/\(escaped(filename))"
    }
}
