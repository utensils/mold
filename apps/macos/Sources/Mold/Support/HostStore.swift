import Foundation
import MoldClient

/// The machines the app knows about, and what each last said about itself.
///
/// Reachability is held per host rather than as one global "connected" flag:
/// mold is multi-host by design, one machine being down says nothing about
/// another, and the Library merges prints from all of them.
@MainActor
@Observable
final class HostStore {
    internal(set) var hosts: [MoldHost]
    internal(set) var reachability: [MoldHost.ID: Reachability] = [:]
    /// What each machine says it can do. Read rather than guessed -- an
    /// absent block has a different meaning per field, so the app never
    /// probes routes to find out.
    internal(set) var capabilities: [MoldHost.ID: Capabilities] = [:]
    /// What each machine will convert a stored print into.
    internal(set) var exportOptions: [MoldHost.ID: ExportOptions] = [:]
    /// What every store's failures funnel into. Newest first. See
    /// `HostStore+Failures`.
    internal(set) var failures: [HostFailure] = []

    /// One live `/api/events` connection per machine. See `HostStore+Events`.
    var watchers: [MoldHost.ID: Task<Void, Never>] = [:]
    /// Registered once each by the composition root's stores, and never
    /// removed -- see `HostStore+Events`.
    var listeners: [(MoldHost.ID, MoldEvent) -> Void] = []
    /// The fleet identity each machine last announced. The same address
    /// answering with a different one is a different library, not a
    /// reconnection.
    var instanceIDs: [MoldHost.ID: String] = [:]

    /// The machine work goes to when nothing else says. Stored, not computed
    /// over the suite, because a fleet card's Default badge and its menu
    /// redraw from it; `HostStore+Default` persists it and follows the suite
    /// so a preferences reset clears it here too.
    var defaultMachine: MoldHost.ID? {
        didSet { if oldValue != defaultMachine { persistDefaultMachine() } }
    }
    @ObservationIgnored var defaultsObserver: (any NSObjectProtocol)?

    /// How a machine's requests actually get made. A stored property, because
    /// it cannot live in an extension -- and this file may not construct a
    /// concrete backend itself, so the default reaches into `+Reachability`
    /// for one. Not `private`: `backend(for:)` reads it from that other file,
    /// and `private` does not cross a file boundary even within one type.
    let makeBackend: @MainActor (MoldHost) -> any MoldBackend

    /// `Equatable` so a view can watch it. Without the conformance
    /// `.onChange(of: hosts.reachability)` has no valid overload, and rather
    /// than saying so the type checker searches until it gives up on the whole
    /// body -- "unable to type-check this expression in reasonable time",
    /// pointing at a line with nothing wrong on it.
    enum Reachability: Equatable {
        case unknown
        case checking
        case up(ServerStatus)
        /// Answering, but refusing us. That is a different problem from being
        /// off: the machine is there and the fix is a key, not a reboot.
        case needsKey
        case down(String)
    }

    init(hosts: [MoldHost],
         makeBackend: @escaping @MainActor (MoldHost) -> any MoldBackend = HostStore.http) {
        self.hosts = hosts
        self.makeBackend = makeBackend
        self.defaultMachine = Self.storedDefaultMachine()
        followDefaultMachineInSuite()
    }
}

extension HostStore {
    /// First-run hosts.
    ///
    /// `MOLD_NATIVE_HOSTS` seeds extra machines as `name=url` pairs so a dev
    /// run can point at real hardware without those addresses living in the
    /// source. Pass it to `macos-dev` or `macos-uat`, which exec the binary
    /// rather than `open`ing it so the variable actually arrives.
    static func seededHosts() -> [MoldHost] {
        // A saved list wins. Seeding over it would resurrect machines the
        // person removed on every launch.
        if let saved = HostPersistence.load() { return saved }

        // No default machine. "This Mac" is now the in-process engine, added
        // when it starts; seeding a second entry at :7680 would put two things
        // called This Mac in the list, only one of which is real.
        var hosts: [MoldHost] = []
        let seed = NativeUAT.hosts.value() ?? ""
        for entry in seed.split(separator: ",") {
            let parts = entry.split(separator: "=", maxSplits: 1)
            // Through the same normalizer the editor uses, so a devshell can
            // seed `workstation=100.105.134.43` without spelling out the port.
            guard parts.count == 2, let url = HostAddress.normalize(String(parts[1]))
            else { continue }
            hosts.append(MoldHost(name: String(parts[0]), baseURL: url))
        }
        return hosts
    }
}
