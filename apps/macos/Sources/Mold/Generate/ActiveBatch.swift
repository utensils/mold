import Foundation
import MoldClient

extension GenerateController {
    /// One admitted batch: what `submit(on:backend:)` got back, kept so
    /// `followNext` can start following it without a second read.
    ///
    /// In its own file rather than beside the queue it waits in: a nested
    /// type is only a shape, and the controller has no room for lines that
    /// are not behaviour.
    struct ActiveBatch: Equatable {
        let id: String
        let clientBatchId: String
        let host: MoldHost.ID
        /// The 202 answer `submit(on:backend:)` got back for this batch.
        let admitted: BatchStatus
    }
}
