import Foundation
import MoldClient

/// Client batch ids that were sent but not seen through to settlement.
///
/// Written BEFORE the request goes out. mold's durable queue is idempotent on
/// this id, so a response lost to a crash or a dropped connection is recovered
/// by asking the host what happened to it -- never by submitting again, which
/// would render the same thing twice and bill the GPU for both.
enum PendingBatch {
    private static let key = "pendingBatches"

    static func remember(_ clientBatchId: String, host: MoldHost.ID, in defaults: UserDefaults = AppStorageSuite.defaults) {
        var pending = all(in: defaults)
        pending[clientBatchId] = host.uuidString
        defaults.set(pending, forKey: key)
    }

    static func forget(_ clientBatchId: String, in defaults: UserDefaults = AppStorageSuite.defaults) {
        var pending = all(in: defaults)
        pending.removeValue(forKey: clientBatchId)
        defaults.set(pending, forKey: key)
    }

    /// Everything remembered against ONE machine. A removed machine can never
    /// answer for its batchs, so records keyed on it are orphans that
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
