import Foundation
import MoldClient

/// Chain jobs this app admitted and has not seen through to settlement.
///
/// Its OWN store, keyed by the job id, because A CHAIN ID IS NOT A BATCH ID.
/// `PendingBatch`'s ids are recovered through
/// `/api/generation-batches/by-client/…`; a chain has no such route and a
/// chain id handed to it would 404. Mixing the two would also point Stop at
/// the wrong verb: a chain is cancelled through its own route, never the
/// queue.
///
/// Written the moment the host names the job -- not before, because before
/// that there is no id to write. The operation id sent with the POST makes a
/// RETRY safe inside one run; across a relaunch an unanswered POST is
/// recovered the same way any other unattributed work is, from the host's own
/// activity.
enum PendingChain {
    private static let key = "pendingChainJobs"

    static func remember(_ jobId: String, host: MoldHost.ID,
                         in defaults: UserDefaults = AppStorageSuite.defaults) {
        var pending = all(in: defaults)
        pending[jobId] = host.uuidString
        defaults.set(pending, forKey: key)
    }

    static func forget(_ jobId: String, in defaults: UserDefaults = AppStorageSuite.defaults) {
        var pending = all(in: defaults)
        pending.removeValue(forKey: jobId)
        defaults.set(pending, forKey: key)
    }

    /// Everything remembered against ONE machine. A removed machine can never
    /// answer for its jobs, so records keyed on it are orphans that
    /// outlived both Remove and Reset (UAT 2026-09-17 #10) -- the reset keeps
    /// this key on purpose, as in-flight bookkeeping, which is exactly why
    /// the removal has to do its own clearing.
    static func forgetAll(on host: MoldHost.ID, in defaults: UserDefaults = AppStorageSuite.defaults) {
        let kept = all(in: defaults).filter { $0.value != host.uuidString }
        defaults.set(kept, forKey: key)
    }

    static func all(in defaults: UserDefaults = AppStorageSuite.defaults) -> [String: String] {
        defaults.dictionary(forKey: key) as? [String: String] ?? [:]
    }
}
