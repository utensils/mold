import Foundation

/// How far through its clips an automatically chained render is.
///
/// A long clip is ONE print and one canvas result. This is the only thing on
/// screen that says it was made in pieces -- there is no scenes UI, no
/// timeline, and nothing about it is an authored sequence.
struct ChainProgress: Equatable {
    /// The durable job's own id. A chain is cancelled through ITS route,
    /// never the queue: a chain id is not a batch id.
    let jobId: String
    var stageCount: Int
    /// 1-based, as it reads. The wire's `stage_idx` is 0-based.
    var currentStage: Int = 1
    var step: Int?
    var total: Int?

    /// What the capsule says under the spinner.
    var label: String { "Clip \(currentStage) of \(stageCount)" }
}
