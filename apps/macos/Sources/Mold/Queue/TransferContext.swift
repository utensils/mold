import Foundation
import MoldClient

/// What every step of one queue transfer needs, bundled so `TransferStore`
/// passes one value instead of eight. Not a property of `TransferStore`
/// itself -- it is rebuilt fresh per call and carries no state of its own,
/// only the two backends and names one transfer is running against.
struct TransferContext {
    let source: MoldHost.ID
    let sourceClient: any MoldBackend
    let sourceName: String
    let destClient: any MoldBackend
    let destName: String
    let expectedDest: String
    let clientBatchId: String
    let verb: String

    func next(_ step: TransferPlan.Step, _ outcome: TransferPlan.Outcome) -> TransferPlan.Result {
        TransferPlan.next(
            after: step, outcome: outcome, clientBatchId: clientBatchId,
            destinationInstance: expectedDest, destinationLabel: destName, sourceLabel: sourceName)
    }
}

/// The machine's own sentence, already produced by `TransferPlan`, surfaced
/// through the one failure funnel every other refusal uses.
struct TransferRefusal: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}
