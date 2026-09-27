import Foundation

/// Admitting and watching one batch of generation work.
public protocol MoldGenerationBackend: Sendable {
    /// Read-only: reserves nothing, queues nothing.
    func placementPreview(_ request: GenerateRequest, copies: Int) async throws -> PlacementPreview
    /// Admits a batch. Idempotent on `clientBatchId`: re-sending the same one
    /// returns the work already held rather than starting it twice.
    func submit(_ admission: BatchAdmission) async throws -> BatchStatus
    func batchStatus(id: String) async throws -> BatchStatus
    /// Recovers a batch whose admission response was lost, by the id this
    /// client minted for it.
    func batchStatus(clientBatchId: String) async throws -> BatchStatus
    /// Step progress and the denoise preview for one running job.
    func jobPreview(jobId: String) async throws -> JobProgress?
    func cancelBatch(id: String) async throws
}
