import Foundation
import MoldClient

/// One admitted piece of work waiting for the canvas.
///
/// M8 decision 8 is about ADMISSION, not about which door the work went
/// through: the host's queue is durable, so a press while something is on
/// screen still reaches the machine and then waits its turn. A chain is a
/// different id on a different route, but it waits in the SAME line -- a
/// separate one would reorder the two kinds against each other, and a press
/// that preempted the canvas would drop whatever was on it.
enum QueuedRun: Equatable {
    case batch(GenerateController.ActiveBatch)
    /// A chain job the host has already minted and nobody is watching yet.
    case chain(AdmittedChain)

    var host: MoldHost.ID {
        switch self {
        case let .batch(batch): batch.host
        case let .chain(chain): chain.host
        }
    }
}

/// A created-but-unwatched chain job. The stage count travels with it because
/// nothing else knows it once the routing decision is out of scope.
struct AdmittedChain: Equatable {
    let jobId: String
    let stageCount: Int
    let host: MoldHost.ID
}
