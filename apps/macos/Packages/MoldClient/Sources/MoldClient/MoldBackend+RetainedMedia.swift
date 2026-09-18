import Foundation

/// A print's retained private conditioning media, and the handle that lets
/// the machine holding it hydrate a request from it.
///
/// Its own protocol rather than three more requirements on
/// `MoldGenerationBackend`: submitting a batch and reading a print's archive
/// are different concerns, and the reuse session is the only route in the app
/// that mints a credential.
public protocol MoldRetainedMediaBackend: Sendable {
    /// Never throws for an unauthorised caller -- a keyed host refuses the
    /// probe in middleware, and that IS the answer `unavailable_auth`.
    func retainedSourceMedia(for filename: String) async throws
        -> RetainedSourceMedia.Inventory
    func retainedSourceMediaBytes(for filename: String, member memberId: String) async throws
        -> Data
    /// Bound by the host to this exact request; a later edit invalidates it.
    func retainedMediaReuseSession(
        for filename: String, members memberIds: [String], target: GenerateRequest
    ) async throws -> RetainedSourceMedia.ReuseSession
}
