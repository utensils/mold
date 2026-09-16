import Foundation
import MoldClient

/// Something a machine could not do, in the terms the person used.
struct HostFailure: Identifiable, Equatable {
    let id = UUID()
    let host: MoldHost.ID
    /// Why it is keyed: a drain that retries four times must leave one line,
    /// not four.
    let verb: String
    let sentence: String
    let at: Date
}

// One funnel for every store's failures, because a failure is always about a
// machine and `HostStore` is the one thing that already knows what each
// machine is called. Not a new type per store, and not three copies of the
// same sentence.
@MainActor
extension HostStore {
    /// Records what a machine couldn't do. Newest first; one entry per
    /// (machine, verb), so a retry replaces its predecessor rather than
    /// piling up.
    ///
    /// `now` exists so a test can pin it -- nothing here reads the clock for
    /// any other reason.
    func report(_ error: Error, on host: MoldHost.ID, doing verb: String, now: Date = Date()) {
        failures.removeAll { $0.host == host && $0.verb == verb }
        let sentence = "Couldn't \(verb) on \(name(of: host) ?? "that machine"). \(error.sentence)"
        failures.insert(HostFailure(host: host, verb: verb, sentence: sentence, at: now), at: 0)
    }

    /// Everything that machine was failing at is no longer true.
    ///
    /// `verb` narrows that to ONE thing, for a passive listing call that
    /// merely happens to run right after some other action -- `reloadTags`
    /// succeeding says the tags are current, not that whatever `deleteTag`
    /// just reported about this same machine has gone away.
    func succeeded(on host: MoldHost.ID, doing verb: String? = nil) {
        failures.removeAll { $0.host == host && (verb == nil || $0.verb == verb) }
    }

    func dismiss(_ failure: HostFailure) {
        failures.removeAll { $0.id == failure.id }
    }
}
