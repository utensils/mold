import Foundation

public extension HTTPBackend {
    func connectionAddresses() async throws -> ConnectionAddresses? {
        do {
            let info: ConnectionAddresses = try await get("/api/connection-addresses")
            guard info.version == 1 else { return nil }
            return info
        } catch MoldClientError.http(let status, _, _) where status == 404 { return nil }
    }

    func resolvedConnection() async throws -> MoldHost? {
        guard let endpoints = host.connectionEndpoints, !endpoints.isEmpty,
              let identity = host.connectionInstanceID, let key = host.apiKey, !key.isEmpty else { return nil }
        var resolved = host
        resolved.baseURL = try await ConnectionRoutes.select(endpoints: endpoints, secret: key, kind: "api",
                                                             instanceID: identity, session: session)
        return resolved
    }
}
