import Foundation
import Network
import Observation

/// Machines on this network advertising `_mold._tcp` (the server's mDNS
/// service, `crates/mold-server/src/mdns.rs`). Browsing starts when a screen
/// that lists Nearby appears and stops when the app leaves the foreground;
/// Local Network permission is asked for the first time it runs.
@Observable
final class NearbyBrowser {
    struct Machine: Identifiable, Hashable {
        let name: String
        /// The TXT record's `id=`: the fleet identity that tells the same
        /// machine found at a second address apart from a new one.
        let instanceID: String?
        let endpoint: NWEndpoint
        var id: String { name }
    }

    private(set) var machines: [Machine] = []
    /// Words for a browser that cannot run -- typically Local Network
    /// permission declined -- rather than an empty list that looks like
    /// "nothing out there".
    private(set) var problem: String?

    @ObservationIgnored private var browser: NWBrowser?

    func start() {
        guard browser == nil else { return }
        let browser = NWBrowser(for: .bonjourWithTXTRecord(type: "_mold._tcp", domain: nil), using: .tcp)
        browser.browseResultsChangedHandler = { results, _ in
            let found = results.compactMap(Self.machine(from:)).sorted { $0.name < $1.name }
            Task { @MainActor in self.machines = found }
        }
        browser.stateUpdateHandler = { state in
            Task { @MainActor in
                switch state {
                case .failed, .waiting:
                    self.problem = String(localized: "Mold Studio can't look on this network. Allow Local Network access for Mold Studio in Settings.")
                case .ready:
                    self.problem = nil
                default:
                    break
                }
            }
        }
        browser.start(queue: .main)
        self.browser = browser
    }

    func stop() {
        browser?.cancel()
        browser = nil
    }

    nonisolated private static func machine(from result: NWBrowser.Result) -> Machine? {
        guard case let .service(name, _, _, _) = result.endpoint else { return nil }
        var instance: String?
        if case let .bonjour(txt) = result.metadata { instance = txt["id"] }
        return Machine(name: name, instanceID: instance, endpoint: result.endpoint)
    }

    /// Turns a found service into an address `HostAddress` accepts, by
    /// opening (and at once closing) a connection to it.
    func address(of machine: Machine) async throws -> String {
        try await withCheckedThrowingContinuation { continuation in
            let connection = NWConnection(to: machine.endpoint, using: .tcp)
            let once = Once()
            connection.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    if case let .hostPort(host, port)? = connection.currentPath?.remoteEndpoint {
                        once.run { continuation.resume(returning: Self.address(host: host, port: port)) }
                    } else {
                        once.run { continuation.resume(throwing: URLError(.cannotFindHost)) }
                    }
                    connection.cancel()
                case let .failed(error), let .waiting(error):
                    once.run { continuation.resume(throwing: error) }
                    connection.cancel()
                default:
                    break
                }
            }
            connection.start(queue: .main)
        }
    }

    nonisolated static func address(host: NWEndpoint.Host, port: NWEndpoint.Port) -> String {
        var text = "\(host)"
        if let scope = text.firstIndex(of: "%") { text = String(text[..<scope]) }
        if text.contains(":") { text = "[\(text)]" }
        return "http://\(text):\(port.rawValue)"
    }
}

/// Resumes a continuation exactly once, whichever state arrives first.
private nonisolated final class Once: @unchecked Sendable {
    private var done = false
    private let lock = NSLock()
    func run(_ body: () -> Void) {
        lock.lock(); defer { lock.unlock() }
        guard !done else { return }
        done = true
        body()
    }
}
