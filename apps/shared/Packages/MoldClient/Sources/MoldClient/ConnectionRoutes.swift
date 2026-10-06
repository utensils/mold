import CryptoKit
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

/// Address selection is independent of mutation transport. A probe never
/// carries a key, follows a redirect, or consumes a pairing token.
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

    static func digest(_ secret: String) -> Data { Data(SHA256.hash(data: Data(secret.utf8))) }
    static func hex(_ bytes: some Sequence<UInt8>) -> String { bytes.map { String(format: "%02x", $0) }.joined() }
    static func tag(_ secret: String) -> String { String(hex(digest(secret)).prefix(16)) }
    static func proof(secret: String, kind: String, nonce: String, instanceID: String) -> String {
        let message = "mold-connection-proof-v1\n\(kind)\n\(nonce)\n\(instanceID)"
        return hex(HMAC<SHA256>.authenticationCode(for: Data(message.utf8), using: SymmetricKey(data: digest(secret))))
    }
    static func verifies(_ value: String, secret: String, kind: String, nonce: String, instanceID: String) -> Bool {
        let expected = Array(proof(secret: secret, kind: kind, nonce: nonce, instanceID: instanceID).utf8)
        let actual = Array(value.utf8)
        guard actual.count == expected.count else { return false }
        return zip(actual, expected).reduce(UInt8(0)) { $0 | ($1.0 ^ $1.1) } == 0
    }

    public static func supportsAutomaticRouting(_ secret: String) -> Bool {
        secret.range(of: "^mold_pair_[A-Za-z0-9_-]{43}$", options: .regularExpression) != nil
    }

    static let defaultProbeTimeout: TimeInterval = 2

    static func probeConfiguration(from session: URLSession,
                                   timeout: TimeInterval = defaultProbeTimeout) -> URLSessionConfiguration {
        let configuration = session.configuration
        configuration.httpAdditionalHeaders = nil
        configuration.httpCookieStorage = nil
        configuration.urlCredentialStorage = nil
        configuration.urlCache = nil
        configuration.timeoutIntervalForResource = timeout
        return configuration
    }

    static func probeRequest(base: URL, body: Data,
                             timeout: TimeInterval = defaultProbeTimeout) -> URLRequest {
        var request = URLRequest(url: base.appending(path: "api/connection-probe"))
        request.httpMethod = "POST"
        request.timeoutInterval = timeout
        request.httpShouldHandleCookies = false
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = body
        return request
    }

    public static func select(
        endpoints: [ConnectionEndpoint], secret: String, kind: String, instanceID: String,
        session: URLSession = APISession.api
    ) async throws -> URL {
        try await select(endpoints: endpoints, secret: secret, kind: kind, instanceID: instanceID,
                         session: session, probeTimeout: defaultProbeTimeout)
    }

    // Test fixtures use an explicit budget so hosted Simulator scheduling does
    // not turn an in-process response into an unreachable-route assertion.
    static func select(endpoints: [ConnectionEndpoint], secret: String, kind: String,
                       instanceID: String, session: URLSession, probeTimeout: TimeInterval) async throws -> URL {
        guard kind != "api" || supportsAutomaticRouting(secret) else { throw MoldClientError.unauthorized }
        let probeSession = URLSession(configuration: probeConfiguration(from: session, timeout: probeTimeout))
        defer { probeSession.invalidateAndCancel() }
        let candidates = sanitized(endpoints)
        guard !candidates.isEmpty else { throw MoldClientError.malformedResponse }
        let nonce = hex((0..<32).map { _ in UInt8.random(in: .min ... .max) })
        let body = try JSONSerialization.data(withJSONObject: ["kind": kind, "key_tag": tag(secret), "nonce": nonce])
        // Every candidate has its own deadline; the group is bounded to eight.
        let cacheKey = kind + ":" + instanceID + ":" + tag(secret) + ":" + candidates.map(\.url).joined(separator: "|")
        let preferred = await ConnectionRouteMemory.shared.preferred(for: cacheKey)
        let valid = await withTaskGroup(of: Int?.self, returning: [Int].self) { group in
            for (index, endpoint) in candidates.enumerated() {
                group.addTask {
                    guard let base = URL(string: endpoint.url) else { return nil }
                    let request = probeRequest(base: base, body: body, timeout: probeTimeout)
                    do {
                        let (bytes, response) = try await probeSession.bytes(for: request, delegate: RelayNoRedirect())
                        defer { bytes.task.cancel() }
                        var data = Data()
                        for try await byte in bytes {
                            guard data.count < 1024 else { return nil }
                            data.append(byte)
                        }
                        guard let http = response as? HTTPURLResponse, http.statusCode == 200,
                              data.count <= 1024,
                              let value = try? MoldJSON.decoder.decode(ConnectionProbeResponse.self, from: data),
                              value.instanceId == instanceID,
                              verifies(value.proof, secret: secret, kind: kind, nonce: nonce, instanceID: instanceID)
                        else { return nil }
                        return index
                    } catch { return nil }
                }
            }
            var result: [Int] = []
            for await index in group { if let index { result.append(index) } }
            return result
        }
        try Task.checkCancellation()
        func priority(_ kind: ConnectionEndpoint.Kind) -> Int {
            switch kind { case .lan: 0; case .tailscale: 1; case .relay: 2 }
        }
        let preferredIndex = preferred.flatMap { route in valid.first { candidates[$0].url == route } }
        guard let winner = preferredIndex ?? valid.min(by: {
            let a = priority(candidates[$0].kind), b = priority(candidates[$1].kind)
            return a == b ? $0 < $1 : a < b
        }), let url = URL(string: candidates[winner].url) else {
            throw URLError(.cannotConnectToHost)
        }
        await ConnectionRouteMemory.shared.remember(url.absoluteString, for: cacheKey)
        return url
    }
}

private actor ConnectionRouteMemory {
    static let shared = ConnectionRouteMemory()
    private var routes: [String: (String, Date)] = [:]
    func preferred(for key: String) -> String? {
        guard let (url, date) = routes[key], Date().timeIntervalSince(date) < 20 else { return nil }
        return url
    }
    func remember(_ url: String, for key: String) {
        if routes[key]?.0 == url, let date = routes[key]?.1, Date().timeIntervalSince(date) < 20 { return }
        if routes.count >= 128 { routes.removeAll() }
        routes[key] = (url, Date())
    }
}

private struct ConnectionProbeResponse: Decodable {
    let instanceId: String
    let proof: String
}
