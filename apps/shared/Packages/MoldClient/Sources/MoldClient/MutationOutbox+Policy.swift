import Foundation

/// When to send, how long to wait, and when to stop trying.
///
/// This used to be inline in `LibraryStore`'s drain loop. It belongs here
/// instead: it is arithmetic over the queue's own bookkeeping (an entry's
/// attempt count), not a screen concern, and a driver anywhere can now ask
/// the same question the same way.
public extension MutationOutbox {

    /// What a driver should do about the head of one machine's chain.
    enum Step: Equatable, Sendable {
        /// Never attempted. Send it now.
        case send(Entry)
        /// It failed once and could plausibly work later. Wait this long,
        /// then send the SAME entry -- not whatever `next` answers after.
        case wait(Duration, then: Entry)
        /// It has failed as many times as it is going to. The entry is
        /// already gone from the chain; these are the rows nothing later
        /// still speaks for.
        case giveUp(Entry, orphaned: [String])
        /// Nothing queued for this machine.
        case idle
    }

    /// The wait before the attempt numbered `attempts` (1, 2, 3, ...): 1s,
    /// 2s, 4s. Widening because a link that is merely slow should not be
    /// hammered, and a machine that is truly gone is not helped by trying
    /// faster.
    static func backoff(after attempts: Int) -> Duration {
        .seconds(pow(2.0, Double(attempts - 1)))
    }

    /// The decision for one machine's head entry.
    ///
    /// A fresh entry (never attempted) sends immediately. One that has
    /// failed and could still succeed waits. One that has failed
    /// `maxAttempts` times is removed here -- the same `failed` every other
    /// give-up already goes through -- and the rows to repair come back with
    /// it, so a driver never has to ask twice.
    mutating func next(for host: MoldHost.ID) -> Step {
        guard let entry = head(for: host) else { return .idle }
        if entry.attempts == 0 { return .send(entry) }
        guard entry.attempts < maxAttempts else {
            return .giveUp(entry, orphaned: failed(entry.id))
        }
        return .wait(Self.backoff(after: entry.attempts), then: entry)
    }
}

public extension MutationOutbox.Entry {
    /// The two shapes a queued change goes out on the wire as.
    enum Wire {
        /// A title is the one change that is not a bulk mutation: a PATCH on
        /// each print, carrying no operation id and no fence. That is safe
        /// precisely because it is idempotent -- setting a title twice is
        /// setting a title -- where adding a tag twice would not be.
        case patch(GalleryPatch, filenames: [String])
        case mutate(GalleryBulkMutation)
    }

    /// This entry's wire form.
    ///
    /// The entry's own id rides every attempt: the host applies a given
    /// operation once, so reusing it is what makes a retry safe.
    var wire: Wire {
        switch change {
        case let .title(_, to):
            .patch(GalleryPatch(title: to), filenames: filenames)
        case let .favorite(on):
            .mutate(GalleryBulkMutation(filenames: filenames, favorite: on, operationId: id))
        case let .tag(name, adding):
            .mutate(GalleryBulkMutation(filenames: filenames,
                                        addTags: adding ? [name] : [],
                                        removeTags: adding ? [] : [name],
                                        operationId: id))
        case let .collection(name, slug, filing):
            .mutate(GalleryBulkMutation(filenames: filenames,
                                        addToCollection: filing ? .named(name) : nil,
                                        removeFromCollectionSlug: filing ? nil : slug,
                                        operationId: id))
        }
    }
}
