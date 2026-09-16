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

    static func remember(_ clientBatchId: String, host: MoldHost.ID) {
        var pending = all()
        pending[clientBatchId] = host.uuidString
        UserDefaults.standard.set(pending, forKey: key)
    }

    static func forget(_ clientBatchId: String) {
        var pending = all()
        pending.removeValue(forKey: clientBatchId)
        UserDefaults.standard.set(pending, forKey: key)
    }

    static func all() -> [String: String] {
        UserDefaults.standard.dictionary(forKey: key) as? [String: String] ?? [:]
    }
}
