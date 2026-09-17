import Foundation
import MoldClient

/// Finding a render whose admission response never made it back -- a crash, a
/// force-quit, a dropped connection while a batch was in flight.
///
/// mold's queue is idempotent on the client batch id, so recovery ASKS the
/// host what happened rather than submitting again, which would render the
/// same thing twice and bill the GPU for both. A type of its own because the
/// asking is a whole concern and `GenerateController` is over its budget: the
/// controller is left with the one thing only it can do, which is to follow
/// what comes back.
@MainActor
enum PendingRecovery {
    /// A batch the host still has live, and the id this Mac knows it by.
    struct Resumable {
        let status: BatchStatus
        let host: MoldHost.ID
        let clientBatchId: String
    }

    /// Asks every machine about the ids this Mac never got an answer for, and
    /// FORGETS the ones nothing can be done about: a machine that is gone, a
    /// batch already at rest (settled, or HELD -- a hold is the Queue's to
    /// show, and re-attaching to one made every launch look as though the app
    /// had started generating on its own), and an id the host refuses outright.
    /// A bad minute on the network is left for the next launch rather than
    /// guessed to be gone.
    static func resolve(hosts: HostStore) async -> [Resumable] {
        var resumable: [Resumable] = []
        for (clientBatchId, hostID) in PendingBatch.all() {
            guard let uuid = UUID(uuidString: hostID), let host = hosts.host(uuid) else {
                PendingBatch.forget(clientBatchId)
                continue
            }
            do {
                let status = try await hosts.backend(for: host)
                    .batchStatus(clientBatchId: clientBatchId)
                guard !status.isAtRest else {
                    PendingBatch.forget(clientBatchId)
                    continue
                }
                resumable.append(
                    Resumable(status: status, host: host.id, clientBatchId: clientBatchId))
            } catch let error as MoldClientError where !error.isTransient {
                // Unknown to the host (e.g. a 404) or refused outright.
                PendingBatch.forget(clientBatchId)
            } catch {
                // A bad minute on the network or the machine.
            }
        }
        return resumable
    }
}
