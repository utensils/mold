import Foundation
import MoldClient

/// Why a machine could not be added or changed, in words.
enum HostEditError: Error, Equatable, LocalizedError {
    case address(HostAddress.Problem)
    /// Another machine in the list already answers here.
    case duplicate(String)
    case keychain(String)
    case storage(String)
    case modeRequiresCredential
    case regularAddress

    var errorDescription: String? {
        switch self {
        case let .address(problem): problem.message
        case let .duplicate(name): "\(name) already answers at this address."
        case let .keychain(reason): "The key couldn't be saved: \(reason)"
        case .regularAddress: "Regular connections require a direct HTTP or HTTPS address. Mold proxy requires Secure."
        case .modeRequiresCredential: "Changing connection mode requires a new pairing code or a separate Regular API key."
        case let .storage(reason): "The machine list couldn't be saved: \(reason)"
        }
    }
}

extension HostStore {
    /// Adds a machine. The key goes to the Keychain first: a machine in the
    /// list whose key failed to save would look keyless and silently refuse.
    @discardableResult
    func add(name: String, address: String, apiKey: String?, makeDefault: Bool,
             id: UUID = UUID(), connectionMode: HostConnectionMode = .secure) throws(HostEditError) -> MoldHost {
        let url: URL
        do { url = try HostAddress.resolve(address) } catch { throw .address(error) }
        if connectionMode == .regular && !DirectHostAddress.isValid(url, allowingPathPrefix: true) { throw .regularAddress }
        if let clash = hosts.first(where: { HostAddress.sameOrigin($0.baseURL, url) }) {
            throw .duplicate(clash.name)
        }
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let key = apiKey?.trimmingCharacters(in: .whitespacesAndNewlines)
        let host = MoldHost(id: id, name: trimmed.isEmpty ? HostAddress.suggestedName(for: url) : trimmed,
                            baseURL: url, apiKey: key?.isEmpty == false ? key : nil, connectionMode: connectionMode)
        do { try credentials.setAPIKey(key ?? "", for: host.id) } catch {
            throw .keychain(error.reasonSentence)
        }
        setHosts(hosts + [host])
        if makeDefault || hosts.count == 1 { setDefaultMachine(host.id) }
        try save()
        Task { await refresh(host) }
        return host
    }

    /// Renames a machine, moves it, or changes its key. `apiKey == nil`
    /// leaves the key alone; an empty string clears it.
    func update(_ id: MoldHost.ID, name: String, address: String, apiKey: String?, connectionMode: HostConnectionMode? = nil, paired: Bool = false) throws(HostEditError) {
        guard var host = host(id) else { return }
        let url: URL
        do { url = try HostAddress.resolve(address) } catch { throw .address(error) }
        if let clash = hosts.first(where: { $0.id != id && HostAddress.sameOrigin($0.baseURL, url) }) {
            throw .duplicate(clash.name)
        }
        if (connectionMode ?? host.connectionMode) == .regular && !DirectHostAddress.isValid(url, allowingPathPrefix: true) { throw .regularAddress }
        let previousKey = host.apiKey
        if let mode = connectionMode, mode != host.connectionMode {
            guard paired || (mode == .regular && apiKey != nil && apiKey?.trimmingCharacters(in: .whitespacesAndNewlines) != host.apiKey) else {
                throw .modeRequiresCredential
            }
            host.connectionMode = mode
            host.secure = nil
            host.connectionEndpoints = nil
            host.connectionOriginalURL = nil
        }
        if let apiKey {
            let key = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
            do { try credentials.setAPIKey(key, for: id) } catch { throw .keychain(error.reasonSentence) }
            host.apiKey = key.isEmpty ? nil : key
        }
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty { host.name = trimmed }
        let moved = !HostAddress.sameOrigin(host.baseURL, url)
        host.baseURL = url
        setHosts(hosts.map { $0.id == id ? host : $0 })
        if moved || previousKey != host.apiKey {
            host.connectionOriginalURL = nil
            host.connectionEndpoints = nil
            host.connectionInstanceID = nil
            setHosts(hosts.map { $0.id == id ? host : $0 })
            setCapabilities(nil, for: id)
        }
        try save()
        stopWatching(id)
        Task { await refresh(host) }
    }

    /// Removes a machine and its key. A Default that is removed falls back to
    /// whatever answers, rather than the first row.
    func remove(_ id: MoldHost.ID) {
        guard let host = host(id) else { return }
        do { try credentials.clearAPIKey(for: id) } catch {
            report(id, name: host.name, doing: "forget its secure connection", error)
            return
        }
        Task { await HTTPBackend.disconnect(host: id) }
        stopWatching(id)
        setHosts(hosts.filter { $0.id != id })
        for key in [id] {
            setReachability(nil, for: key); setCapabilities(nil, for: key)
            setModels(nil, for: key); setLastAnswered(nil, for: key)
        }
        if defaultMachine == id { setDefaultMachine(nil) }
        clearFailures(for: id)
        try? save()
    }

    func makeDefault(_ id: MoldHost.ID) {
        setDefaultMachine(id)
        try? save()
    }

    private func save() throws(HostEditError) {
        do { try persist() } catch { throw .storage(error.reasonSentence) }
    }
}
