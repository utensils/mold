import Foundation

/// The durable chain job an over-long clip becomes.
///
/// Three verbs and nothing else: this app AUTHORS no sequences, so it never
/// lists, resumes, retakes or amends one. It creates the ephemeral job a
/// length past the checkpoint's clip size requires, follows it, and cancels
/// it -- through the job's OWN route, never the queue, because a chain id is
/// not a batch id.
public protocol MoldChainBackend: Sendable {
    /// `POST /api/chain-jobs` with `ephemeral: true`.
    ///
    /// `operationId` is the idempotency fence, exactly as `clientBatchId` is
    /// for a batch: the id is minted and persisted BEFORE the request goes
    /// out, so an answer lost to a crash is recovered by asking rather than by
    /// submitting again and rendering twice.
    func createChainJob(_ request: AutoChainRequest, operationId: String) async throws
        -> CreateChainJobResponse
    func chainJobEvents(id: String) -> AsyncThrowingStream<ChainJobEvent, Error>
    func chainJob(id: String) async throws -> ChainJobDetail
    func cancelChainJob(id: String) async throws
    /// What this machine will chain for one model -- the HOST's own limits,
    /// which outrank every constant this app carries.
    func chainLimits(model: String, fps: Int?) async throws -> ChainLimits
}
