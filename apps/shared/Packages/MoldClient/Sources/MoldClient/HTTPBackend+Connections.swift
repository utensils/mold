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
        try Task.checkCancellation()
        resolved.connectionOriginalURL = host.connectionOriginalURL ?? host.baseURL
        guard key.range(of: "^mold_pair_[A-Za-z0-9_-]{43}$", options: .regularExpression) != nil else {
            guard let original = host.connectionOriginalURL, original != host.baseURL else { return nil }
            resolved.baseURL = original
            return resolved
        }
        do {
            resolved.baseURL = try await ConnectionRoutes.select(endpoints: endpoints, secret: key, kind: "api",
                                                                 instanceID: identity, session: session)
        } catch {
            try Task.checkCancellation()
            if error is CancellationError || (error as? URLError)?.code == .cancelled { throw error }
            // Only the user's saved origin is trusted without a new route proof.
            // The subsequent authenticated status read learns fresh addresses.
            resolved.baseURL = resolved.connectionOriginalURL!
        }
        return resolved
    }
}
