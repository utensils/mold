import Foundation

/// The queue that carries organization edits to the machines.
///
/// A value, and tested as one. The driving -- when to send, how long to wait,
/// when to stop trying -- belongs to the app; what belongs here is the
/// bookkeeping that decides whether a failure means the screen is now wrong.
///
/// One chain per machine, because they are separate servers with separate
/// stores and there is no fleet-wide order to preserve. A machine on a slow
/// link must not hold up an edit to one sitting on the same desk.
public struct MutationOutbox: Sendable {

    public struct Entry: Identifiable, Hashable, Sendable {
        /// The server's fence. Minted ONCE, with the entry, and reused for
        /// every attempt -- a fresh id per attempt is exactly the double-apply
        /// the fence exists to prevent.
        public let id: String
        public let host: MoldHost.ID
        public let change: PrintChange
        public let filenames: [String]
        public internal(set) var attempts: Int
    }

    private var chains: [MoldHost.ID: [Entry]] = [:]

    public init() {}

    public var isEmpty: Bool { chains.values.allSatisfy(\.isEmpty) }

    /// Every machine with work waiting, so a driver knows which chains to run.
    public var waiting: Set<MoldHost.ID> {
        Set(chains.filter { !$0.value.isEmpty }.keys)
    }

    /// Splits an edit into one entry per machine and queues each behind that
    /// machine's own chain.
    @discardableResult
    public mutating func enqueue(_ edit: PrintEdit) -> [Entry] {
        var queued: [Entry] = []
        for (host, filenames) in edit.targets.sorted(by: { $0.key.uuidString < $1.key.uuidString }) {
            let entry = Entry(id: UUID().uuidString, host: host, change: edit.change,
                              filenames: filenames, attempts: 1)
            chains[host, default: []].append(entry)
            queued.append(entry)
        }
        return queued
    }

    /// What this machine should be sending now.
    public func head(for host: MoldHost.ID) -> Entry? { chain(for: host).first }

    /// Everything still queued for a machine, oldest first.
    ///
    /// A caller replaying optimistic state onto freshly-read rows needs the
    /// whole chain in order, not just the head: each entry was applied to the
    /// screen when it was made, and re-reading the machine threw all of them
    /// away at once.
    public func chain(for host: MoldHost.ID) -> [Entry] { chains[host] ?? [] }

    /// It landed. The chain moves on.
    public mutating func succeeded(_ id: String) {
        remove(id)
    }

    /// It did not land, and we mean to try again. The entry keeps its place at
    /// the head and its operation id, and only the attempt count moves.
    public mutating func retry(_ id: String) {
        guard let (host, index) = locate(id) else { return }
        chains[host]?[index].attempts += 1
    }

    /// We are giving up on it.
    ///
    /// Returns the filenames whose rows on screen are now untrustworthy: the
    /// ones no LATER entry for the same machine still speaks for. A row with a
    /// newer edit pending is deliberately left alone -- the screen is showing
    /// that newer intent, which has not failed, and repairing the row would
    /// undo something the person did afterwards. Same filename on another
    /// machine is a different print and supersedes nothing.
    @discardableResult
    public mutating func failed(_ id: String) -> [String] {
        guard let (host, index) = locate(id) else { return [] }
        let entry = chains[host]![index]
        let newer = Set(chains[host]!.dropFirst(index + 1).flatMap(\.filenames))
        remove(id)
        return entry.filenames.filter { !newer.contains($0) }
    }

    private func locate(_ id: String) -> (MoldHost.ID, Int)? {
        for (host, chain) in chains {
            if let index = chain.firstIndex(where: { $0.id == id }) { return (host, index) }
        }
        return nil
    }

    private mutating func remove(_ id: String) {
        guard let (host, index) = locate(id) else { return }
        chains[host]?.remove(at: index)
    }
}
