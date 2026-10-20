import Foundation

/// Durable authored sequences and automatic long-video jobs share the same
/// lifecycle. Authored submission preserves history; AutoChainRequest keeps
/// the existing hidden one-shot behavior. Job ids always use these routes,
/// never the ordinary batch queue.
public protocol MoldChainBackend: Sendable {
    /// `POST /api/chain-jobs` with `ephemeral: true`.
    ///
    /// `operationId` is the idempotency fence, exactly as `clientBatchId` is
    /// for a batch: the id is minted and persisted BEFORE the request goes
    /// out, so an answer lost to a crash is recovered by asking rather than by
    /// submitting again and rendering twice.
    func createChainJob(_ request: AutoChainRequest, operationId: String) async throws
        -> CreateChainJobResponse
    func createChainJob(_ request: AuthoredChainRequest, operationId: String) async throws
        -> CreateChainJobResponse
    func validateChain(_ request: AuthoredChainRequest) async throws -> ChainValidationResponse
    func listChainJobs() async throws -> ChainJobListing
    func chainJob(id: String, includeSourceMedia: Bool) async throws -> ChainJobDetail
    func chainJobEvents(id: String) -> AsyncThrowingStream<ChainJobEvent, Error>
    func chainJob(id: String) async throws -> ChainJobDetail
    func cancelChainOperation(id: String, operationId: String) async throws
    func cancelChainJob(id: String) async throws
    /// Restarts a job a host restart PARKED. An ephemeral chain keeps its
    /// manifest, its source media, its finished clips and its tail cache
    /// across a graceful shutdown, so resuming is continuing -- not a
    /// re-render (CLAUDE.md, "Scripted sequences").
    func resumeChainJob(id: String) async throws
    /// What this machine will chain for one model -- the HOST's own limits,
    /// which outrank every constant this app carries.
    func chainLimits(model: String, fps: Int?) async throws -> ChainLimits
}

/// Older/fake backends fail closed instead of silently submitting ephemeral work.
public extension MoldChainBackend {
    func cancelChainOperation(id: String, operationId: String) async throws {
        throw SequenceDraftError.invalid("This connection cannot cancel a pending video request.")
    }
    func createChainJob(_ request: AuthoredChainRequest, operationId: String) async throws -> CreateChainJobResponse {
        throw SequenceDraftError.invalid("This connection does not support authored clips.")
    }
    func validateChain(_ request: AuthoredChainRequest) async throws -> ChainValidationResponse {
        throw SequenceDraftError.invalid("This connection does not support authored clips.")
    }
    func listChainJobs() async throws -> ChainJobListing {
        throw SequenceDraftError.invalid("This connection does not support video history.")
    }
    func chainJob(id: String, includeSourceMedia: Bool) async throws -> ChainJobDetail {
        guard !includeSourceMedia else {
            throw SequenceDraftError.invalid("This connection cannot restore video inputs.")
        }
        return try await chainJob(id: id)
    }
}
