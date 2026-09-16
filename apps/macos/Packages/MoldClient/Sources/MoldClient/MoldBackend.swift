import Foundation

/// The one seam between the UI and wherever generation actually happens.
///
/// There will be three conformances and only the composition root knows which
/// one is in play:
///   - `HTTPBackend`     a remote `mold serve` over the network
///   - `EmbeddedBackend` mold's own Rust engine running in-process on loopback
///   - `MockBackend`     previews and tests
///
/// The first two are the SAME transport. `mold-server` speaks one wire
/// contract whether it is bound to a Tailscale address or to `127.0.0.1`, so
/// embedding the engine later changes a URL, not this protocol.
public protocol MoldBackend: Sendable {
    var host: MoldHost { get }
    func status() async throws -> ServerStatus
    func capabilities() async throws -> Capabilities
    func models() async throws -> [Model]
    /// Pass the previous `etag` to let the host answer `.notModified`.
    func gallery(etag: String?) async throws -> Fetched<[GalleryPrint]>
    /// Read-only: reserves nothing, queues nothing.
    func placementPreview(_ request: GenerateRequest, copies: Int) async throws -> PlacementPreview
    func queue() async throws -> QueueListing

    /// Admits a batch. Idempotent on `clientBatchId`: re-sending the same one
    /// returns the work already held rather than starting it twice.
    func submit(_ admission: BatchAdmission) async throws -> BatchStatus
    /// Whole-snapshot frames; safe to reconnect at any point.
    func batchEvents(id: String) -> AsyncThrowingStream<BatchStatus, Error>
    func batchStatus(id: String) async throws -> BatchStatus
    /// Recovers a batch whose admission response was lost, by the id this
    /// client minted for it.
    func batchStatus(clientBatchId: String) async throws -> BatchStatus
    /// Step progress and the denoise preview for one running job.
    func jobPreview(jobId: String) async throws -> JobProgress?
    func cancelBatch(id: String) async throws
}

public enum MoldClientError: Error, Sendable, LocalizedError {
    case unreachable(String)
    case unauthorized
    case http(status: Int, code: String?, message: String?)
    case malformedResponse

    /// Whether sending the same request again could plausibly work.
    ///
    /// Retrying something that cannot succeed is not resilience, it is a
    /// spinner that never stops -- so this is deliberately narrow: the link
    /// being down, the machine being too busy, and the machine having a bad
    /// minute. A missing key does not appear by waiting, a refused request
    /// stays refused, and a reply this build cannot parse will not parse on
    /// the next attempt either.
    public var isTransient: Bool {
        switch self {
        case .unreachable: true
        case .unauthorized: false
        case let .http(status, _, _): status >= 500 || status == 429
        case .malformedResponse: false
        }
    }

    public var errorDescription: String? {
        switch self {
        case let .unreachable(reason):
            "Couldn't reach this machine. \(reason)"
        case .unauthorized:
            "This machine needs an API key. Add one in Settings."
        case let .http(status, _, message):
            message ?? "The machine answered with an error (\(status))."
        case .malformedResponse:
            "The machine sent something this version of Mold can't read."
        }
    }
}
