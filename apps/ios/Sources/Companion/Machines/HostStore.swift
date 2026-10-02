import Foundation
import MoldClient

/// The machines this phone knows, and what each last said about itself.
///
/// Reachability is per machine, as on the Mac: one being down says nothing
/// about another. Keys come from the Keychain (`CredentialStore`), the list
/// from `hosts.json`; neither is ever written into the other.
@Observable
final class HostStore {
    private(set) var hosts: [MoldHost]
    private(set) var reachability: [MoldHost.ID: Reachability] = [:]
    /// What each machine says it can do -- read, never guessed.
    private(set) var capabilities: [MoldHost.ID: Capabilities] = [:]
    /// Each machine's models, as it listed them: Generate's model menu, and
    /// the card's "14 installed".
    private(set) var models: [MoldHost.ID: [Model]] = [:]
    var installed: [MoldHost.ID: Int] {
        models.mapValues { $0.filter { $0.downloaded ?? true }.count }
    }
    /// When each machine last answered, so a silent one can say since when.
    private(set) var lastAnswered: [MoldHost.ID: Date] = [:]
    private(set) var defaultMachine: MoldHost.ID?
    /// Every store's failures funnel here, newest first, for the inline banner.
    private(set) var failures: [HostFailure] = []

    @ObservationIgnored let list: HostListFile
    @ObservationIgnored let credentials: any CredentialStore
    @ObservationIgnored let makeBackend: (MoldHost) -> any MoldBackend
    @ObservationIgnored var watchers: [MoldHost.ID: Task<Void, Never>] = [:]
    @ObservationIgnored var listeners: [(MoldHost.ID, MoldEvent) -> Void] = []
    @ObservationIgnored var watching = false

    enum Reachability: Equatable {
        case unknown
        case checking
        case up(ServerStatus)
        /// Answering, but refusing us: the fix is a key, not a reboot.
        case needsKey
        case down(String)
    }

    init(list: HostListFile, credentials: any CredentialStore,
         makeBackend: @escaping (MoldHost) -> any MoldBackend) {
        self.list = list
        self.credentials = credentials
        self.makeBackend = makeBackend
        let stored = list.load()
        defaultMachine = stored.defaultID
        hosts = []
        hosts = stored.entries.map { entry in
            MoldHost(id: entry.id, name: entry.name, baseURL: entry.baseURL, apiKey: readKey(entry),
                     connectionEndpoints: entry.connectionEndpoints, connectionInstanceID: entry.connectionInstanceID, connectionOriginalURL: entry.connectionOriginalURL)
        }
    }

    private func readKey(_ entry: HostList.Entry) -> String? {
        do {
            return try credentials.apiKey(for: entry.id)
        } catch {
            // A locked Keychain is not "no key": say so, and keep the machine.
            report(entry.id, name: entry.name, doing: "read its key", error)
            return nil
        }
    }

    func host(_ id: MoldHost.ID) -> MoldHost? { hosts.first { $0.id == id } }
    func backend(for host: MoldHost) -> any MoldBackend { makeBackend(host) }
    func backend(for id: MoldHost.ID) -> (any MoldBackend)? { host(id).map(backend(for:)) }
    func reachability(of host: MoldHost) -> Reachability { reachability[host.id] ?? .unknown }

    func isUp(_ host: MoldHost) -> Bool {
        if case .up = reachability(of: host) { return true }
        return false
    }

    /// Where work goes when nothing else says: the Default, even when it is
    /// down (it then says why), else the first machine that answers.
    var preferredHost: MoldHost? {
        defaultMachine.flatMap(host) ?? hosts.first(where: isUp) ?? hosts.first
    }

    var upHosts: [MoldHost] { hosts.filter(isUp) }

    /// The run of the server this machine last answered as: what a retry or
    /// a transfer must name. `nil` until it answers.
    func instanceID(of id: MoldHost.ID) -> String? {
        guard let host = host(id), case let .up(status) = reachability(of: host) else { return nil }
        return status.instanceId
    }

    func persist() throws {
        try list.save(HostList(
            entries: hosts.map { .init(id: $0.id, name: $0.name, baseURL: $0.baseURL,
                                     connectionEndpoints: $0.connectionEndpoints, connectionInstanceID: $0.connectionInstanceID, connectionOriginalURL: $0.connectionOriginalURL) },
            defaultID: defaultMachine))
    }

    func setHosts(_ new: [MoldHost]) { hosts = new }
    func setDefaultMachine(_ id: MoldHost.ID?) { defaultMachine = id }
    func setReachability(_ state: Reachability?, for id: MoldHost.ID) { reachability[id] = state }
    func setCapabilities(_ value: Capabilities?, for id: MoldHost.ID) { capabilities[id] = value }
    func setModels(_ list: [Model]?, for id: MoldHost.ID) { models[id] = list }
    func setLastAnswered(_ date: Date?, for id: MoldHost.ID) { lastAnswered[id] = date }
    func setFailures(_ new: [HostFailure]) { failures = new }
}
