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

    /// The verb every `.unreachable` failure collapses to. A machine that
    /// cannot be reached fails everything for the same one reason, so it
    /// gets the same one line -- and clears the same way, the moment
    /// `HostStore+Reachability` finds it answering again.
    static let reachVerb = "reach"
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
    ///
    /// An unreachable machine is a special case: it fails every verb for the
    /// same one reason, so ALL of that machine's lines -- not just one keyed
    /// to this verb -- collapse into the single `reachVerb` line. A real
    /// refusal (a 409, a bad key, ...) keeps its own verb line and leaves an
    /// existing reach line alone, because the machine answering at all is a
    /// different fact than whatever it just refused.
    func report(_ error: Error, on host: MoldHost.ID, doing verb: String, now: Date = Date()) {
        // A cancelled request is the app changing its mind -- a `.task(id:)`
        // re-keying as a selection settles, a pane going away mid-request --
        // not the machine failing. Recording it would tell the person their
        // reachable machine can't be reached.
        guard !(error is CancellationError) else { return }
        let name = name(of: host) ?? "That machine"
        guard case MoldClientError.unreachable = error else {
            failures.removeAll { $0.host == host && $0.verb == verb }
            let sentence = "\(name) couldn't \(verb) — \(error.reason)"
            failures.insert(HostFailure(host: host, verb: verb, sentence: sentence, at: now), at: 0)
            return
        }
        failures.removeAll { $0.host == host }
        let sentence = "\(name) can't be reached — \(error.reason)"
        failures.insert(HostFailure(host: host, verb: HostFailure.reachVerb, sentence: sentence, at: now), at: 0)
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
