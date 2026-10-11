import Foundation

public struct ConnectionEndpoint: Codable, Hashable, Sendable {
    public enum Kind: String, Codable, Sendable { case lan, tailscale, relay }
    public let url: String
    public let kind: Kind
    public init(url: String, kind: Kind) { self.url = url; self.kind = kind }
}

public struct ConnectionAddresses: Codable, Hashable, Sendable {
    public let version: Int
    public let instanceId: String
    public let endpoints: [ConnectionEndpoint]
}

/// Untrusted discovery supplies bounded addresses. Only a pinned Noise handshake can select one.
public enum ConnectionRoutes {
    public static func sanitized(_ endpoints: [ConnectionEndpoint]) -> [ConnectionEndpoint] {
        var seen = Set<String>()
        return endpoints.prefix(8).compactMap { endpoint in
            guard endpoint.url.utf8.count <= 2048, let parts = URLComponents(string: endpoint.url),
                  let scheme = parts.scheme?.lowercased(), ["http", "https"].contains(scheme),
                  let host = parts.host?.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: ".")), !host.isEmpty,
                  parts.user == nil, parts.password == nil, parts.query == nil, parts.fragment == nil,
                  parts.path.isEmpty || parts.path == "/",
                  parts.port.map({ (1...65535).contains($0) }) ?? true,
                  !["localhost", "::1", "[::1]", "0.0.0.0", "::", "[::]"].contains(host),
                  !host.hasPrefix("127."), !host.hasSuffix(".localhost"), !host.hasPrefix("169.254."),
                  !host.contains("%"),
                  host.trimmingCharacters(in: CharacterSet(charactersIn: "[]")).range(of: "^fe[89ab][0-9a-f]:", options: .regularExpression) == nil,
                  endpoint.kind != .relay || scheme == "https",
                  let url = parts.url else { return nil }
            let value = url.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
            guard seen.insert(value).inserted else { return nil }
            return ConnectionEndpoint(url: value, kind: endpoint.kind)
        }
    }

    static func select<Connection: Sendable>(
        _ candidates: [ConnectionEndpoint],
        connect: @escaping @Sendable (ConnectionEndpoint) async throws -> Connection,
        close: @escaping @Sendable (Connection) async -> Void
    ) async throws -> (Int, Connection) {
        guard let bestClass = candidates.map({ SecureTransportPool.priority($0.kind) }).min() else {
            throw URLError(.cannotConnectToHost)
        }
        let chosen = await withTaskGroup(of: (Int, Connection?).self, returning: (Int, Connection)?.self) { group in
            for (index, endpoint) in candidates.enumerated() {
                group.addTask {
                    do {
                        let delay = SecureTransportPool.priority(endpoint.kind) - bestClass
                        if delay > 0 { try await Task.sleep(for: .milliseconds(delay * 150)) }
                        try Task.checkCancellation()
                        return (index, try await connect(endpoint))
                    } catch { return (index, nil) }
                }
            }
            var pending = Set(candidates.indices), valid: [Int: Connection] = [:]
            var winner: Int?
            for await (index, connection) in group {
                pending.remove(index)
                if let connection { valid[index] = connection }
                if winner == nil, let best = valid.keys.min(by: {
                    let a = SecureTransportPool.priority(candidates[$0].kind), b = SecureTransportPool.priority(candidates[$1].kind)
                    return a == b ? $0 < $1 : a < b
                }), !pending.contains(where: { SecureTransportPool.priority(candidates[$0].kind) < SecureTransportPool.priority(candidates[best].kind) }) {
                    winner = best; group.cancelAll()
                }
            }
            for (index, connection) in valid where index != winner { await close(connection) }
            guard let winner, let connection = valid[winner] else { return nil }
            return (winner, connection)
        }
        if Task.isCancelled {
            if let chosen { await close(chosen.1) }
            throw CancellationError()
        }
        guard let chosen else { throw URLError(.cannotConnectToHost) }
        return chosen
    }
}
