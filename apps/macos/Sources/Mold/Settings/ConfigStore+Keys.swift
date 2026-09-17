import Foundation
import MoldClient

// Per-key editing for the Advanced table (S3): one PUT or DELETE, then a
// full re-read -- a single `expand.*` write rewrites all eight rows
// (`config_sync.rs:674-688`), so trusting the one row that was answered
// would leave the other seven showing stale numbers.
extension ConfigStore {
    /// Sets one key. Applies the machine's own answer in place first, so the
    /// field does not flicker back to its old value while the re-read is in
    /// flight, then re-reads the whole listing. A refusal is recorded
    /// against this key alone (`ConfigStore.recordFailure`) rather than
    /// reported to `hosts.failures` -- a key the machine refused is about
    /// that row, the same reasoning M3 S2 already applied to a failed
    /// expansion.
    @discardableResult
    func set(_ key: String, to value: ConfigScalar, on host: MoldHost.ID) async -> Bool {
        guard let client = hosts.backend(for: host) else { return false }
        do {
            let entry = try await client.setConfig(key, to: value)
            applyEntry(entry, on: host)
            clearRefusal(key, on: host)
            await refresh(on: host)
            return true
        } catch {
            recordFailure(error, for: key, on: host, doing: "change \(key)")
            return false
        }
    }

    /// Drops the DB row so the key falls back to file/env/default, and shows
    /// the fallback the machine reports rather than predicting it.
    @discardableResult
    func reset(_ key: String, on host: MoldHost.ID) async -> Bool {
        guard let client = hosts.backend(for: host) else { return false }
        do {
            let entry = try await client.resetConfig(key)
            applyEntry(entry, on: host)
            clearRefusal(key, on: host)
            await refresh(on: host)
            return true
        } catch {
            recordFailure(error, for: key, on: host, doing: "reset \(key)")
            return false
        }
    }
}
