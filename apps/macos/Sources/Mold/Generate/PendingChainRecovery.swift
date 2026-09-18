import Foundation
import MoldClient

/// Finding the chain jobs this Mac admitted and never saw through.
///
/// The twin of `PendingRecovery`, and a separate type for the same reason it
/// is: a chain is asked about on a DIFFERENT route, settles into a different
/// answer, and `GenerateController` is at its budget. A chain job is durable —
/// it keeps rendering across a quit and a host restart parks it as `paused` —
/// so recovery is a READ, never a second create.
@MainActor
enum PendingChainRecovery {
    /// A chain the host still has, and what this Mac needs to follow it.
    struct Resumable {
        let jobId: String
        let host: MoldHost.ID
        let stageCount: Int
        /// Where the host says it had got to. Re-attaching at clip 1 would
        /// read as progress being undone.
        let currentStage: Int
        let isPaused: Bool
    }

    /// What every id in `PendingChain` turned out to be.
    struct Found {
        var resumable: [Resumable] = []
        /// Jobs that FINISHED while the app was closed, newest last. Their
        /// prints are worth showing -- the render happened, and silently
        /// dropping the record would leave somebody wondering.
        var finished: [(filename: String, host: MoldHost.ID)] = []
    }

    /// Asks every machine about the ids this Mac never followed to settlement,
    /// and FORGETS the ones nothing can be done about: a machine that is gone,
    /// an id the host refuses outright, and a job already terminal. A bad
    /// minute on the network is left for the next launch rather than guessed
    /// to be gone -- exactly `PendingRecovery`'s rule.
    static func resolve(hosts: HostStore) async -> Found {
        var found = Found()
        for (jobId, hostID) in PendingChain.all() {
            guard let uuid = UUID(uuidString: hostID), let host = hosts.host(uuid) else {
                PendingChain.forget(jobId)
                continue
            }
            do {
                let detail = try await hosts.backend(for: host).chainJob(id: jobId)
                guard !detail.state.isTerminal else {
                    PendingChain.forget(jobId)
                    if detail.state == .completed, let filename = detail.galleryFilename {
                        found.finished.append((filename, host.id))
                    }
                    continue
                }
                // The stage count comes from the JOB, never from a routing
                // decision this launch no longer has.
                found.resumable.append(Resumable(
                    jobId: jobId, host: host.id,
                    stageCount: max(detail.stageCount, 1),
                    currentStage: max(detail.currentStage + 1, 1),
                    isPaused: detail.state == .paused))
            } catch let error as MoldClientError where !error.isTransient {
                // Unknown to the host (a 404) or refused outright.
                PendingChain.forget(jobId)
            } catch {
                // A bad minute on the network or the machine.
            }
        }
        return found
    }
}
