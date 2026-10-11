import Foundation

/// Connection reuse shares bounded multiplexing; a failed request is never retried here.
actor SecureTransportPool {
    static let shared = SecureTransportPool()
    private var epochs: [MoldHost.ID: UInt64] = [:]
    private var versions: [MoldHost: UUID] = [:]
    private var touched: [MoldHost: Date] = [:]
    private var cleanup: Task<Void, Never>?
    private var connections: [MoldHost: Task<SecureHTTPTransport, any Error>] = [:]

    func response(_ request: URLRequest, body: MoldRequestBody, host: MoldHost) async throws -> (MoldResponseBytes, HTTPURLResponse) {
        touch(host)
        let task = try connection(for: host)
        let version = versions[host]
        do { return try await task.value.response(for: request, body: body) }
        catch {
            if let transport = try? await task.value {
                if !(await transport.isUsable) {
                    if versions[host] == version { connections.removeValue(forKey: host); versions.removeValue(forKey: host) }
                    await transport.close()
                }
            } else if versions[host] == version { connections.removeValue(forKey: host); versions.removeValue(forKey: host) }
            throw error
        }
    }

    private func connection(for host: MoldHost) throws -> Task<SecureHTTPTransport, any Error> {
        if let existing = connections[host] { return existing }
        guard connections.count < 64 else { throw SecureConnectionError.capacity }
        let task = Task {
            guard let secure = host.secure else { throw SecureConnectionError.pairingRequired }
            let identity = try await SecureIdentityStore.shared.identity()
            let endpoint = try Self.socketURL(for: host.baseURL, kind: Self.kind(host.baseURL, host: host), secure: secure)
            let socket = try await SecureWebSocketConnection.connect(host: secure, identity: identity, endpoint: endpoint)
            return SecureHTTPTransport(connection: socket)
        }
        connections[host] = task
        versions[host] = UUID()
        return task
    }

    func resolve(_ host: MoldHost) async throws -> MoldHost {
        guard let secure = host.secure else { throw SecureConnectionError.pairingRequired }
        touch(host)
        let epoch = epochs[host.id, default: 0]
        if let existing = connections[host], let transport = try? await existing.value, await transport.isUsable { return host }
        let identity = try await SecureIdentityStore.shared.identity()
        var endpoints = ConnectionRoutes.sanitized(host.connectionEndpoints ?? [])
        let original = host.connectionOriginalURL ?? host.baseURL
        if !endpoints.contains(where: { $0.url == original.absoluteString }) {
            endpoints.append(ConnectionEndpoint(url: original.absoluteString, kind: .relay))
        }
        guard !endpoints.isEmpty else { throw SecureConnectionError.pairingRequired }
        let candidates = endpoints
        let (index, socket) = try await ConnectionRoutes.select(candidates, connect: { endpoint in
            guard let base = URL(string: endpoint.url) else { throw SecureConnectionError.protocolViolation }
            let socket = try Self.socketURL(for: base, kind: endpoint.kind, secure: secure)
            return try await SecureWebSocketConnection.connect(host: secure, identity: identity, endpoint: socket,
                timeout: endpoint.kind == .relay ? 12 : 2)
        }, close: { await $0.close() })
        guard epochs[host.id, default: 0] == epoch, !Task.isCancelled else { await socket.close(); throw CancellationError() }
        guard let url = URL(string: candidates[index].url) else { await socket.close(); throw SecureConnectionError.protocolViolation }
        var resolved = host; resolved.baseURL = url; resolved.connectionOriginalURL = original
        for key in connections.keys where key.id == resolved.id && key != resolved {
            versions.removeValue(forKey: key); touched.removeValue(forKey: key)
            if let old = connections.removeValue(forKey: key), let transport = try? await old.value { await transport.close() }
        }
        if let old = connections.removeValue(forKey: resolved), let transport = try? await old.value { await transport.close() }
        guard epochs[host.id, default: 0] == epoch, !Task.isCancelled else { await socket.close(); throw CancellationError() }
        guard connections.count < 64 else { await socket.close(); throw SecureConnectionError.capacity }
        touch(resolved)
        connections[resolved] = Task { SecureHTTPTransport(connection: socket) }
        versions[resolved] = UUID()
        return resolved
    }

    private func touch(_ host: MoldHost) {
        touched[host] = .now
        if cleanup == nil {
            cleanup = Task {
                while !Task.isCancelled {
                    do { try await Task.sleep(for: .seconds(30)) } catch { return }
                    await prune()
                }
            }
        }
    }
    private func prune() async {
        for (host, task) in connections {
            let version = versions[host]
            guard Date().timeIntervalSince(touched[host] ?? .distantPast) > 60 else { continue }
            guard let transport = try? await task.value else {
                if versions[host] == version { connections.removeValue(forKey: host); touched.removeValue(forKey: host); versions.removeValue(forKey: host) }
                continue
            }
            if await transport.isIdle, versions[host] == version {
                connections.removeValue(forKey: host); touched.removeValue(forKey: host); versions.removeValue(forKey: host)
                await transport.close()
            }
        }
        if connections.isEmpty { cleanup?.cancel(); cleanup = nil }
    }
    func replace(_ host: MoldHost, with updated: MoldHost) {
        if let task = connections.removeValue(forKey: host) { connections[updated] = task; versions[updated] = versions.removeValue(forKey: host) }
        touched.removeValue(forKey: host); touch(updated)
    }
    func disconnect(_ id: MoldHost.ID) async {
        epochs[id, default: 0] &+= 1
        let removed = connections.filter { $0.key.id == id }
        for (host, task) in removed { connections.removeValue(forKey: host); touched.removeValue(forKey: host); versions.removeValue(forKey: host); task.cancel() }
        for task in removed.values { if let transport = try? await task.value { await transport.close() } }
    }

    static func priority(_ kind: ConnectionEndpoint.Kind) -> Int {
        switch kind { case .lan: 0; case .tailscale: 1; case .relay: 2 }
    }
    static func kind(_ url: URL, host: MoldHost) -> ConnectionEndpoint.Kind {
        host.connectionEndpoints?.first(where: { URL(string: $0.url).map { HostAddress.sameOrigin($0, url) } == true })?.kind ?? .relay
    }
    static func socketURL(for origin: URL, kind: ConnectionEndpoint.Kind, secure: SecureHost) throws -> URL {
        if kind == .relay { return secure.websocketURL }
        guard var parts = URLComponents(url: origin, resolvingAgainstBaseURL: false),
              parts.user == nil, parts.password == nil, parts.query == nil, parts.fragment == nil,
              ["http", "https"].contains(parts.scheme ?? "") else { throw SecureConnectionError.protocolViolation }
        parts.scheme = parts.scheme == "https" ? "wss" : "ws"
        parts.path = "/_mold/secure"
        guard let url = parts.url else { throw SecureConnectionError.protocolViolation }
        return url
    }
}
