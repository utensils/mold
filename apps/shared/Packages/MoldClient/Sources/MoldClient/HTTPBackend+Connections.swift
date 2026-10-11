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
        if host.secure != nil {
            let resolved = try await SecureTransportPool.shared.resolve(host)
            guard let secure = resolved.secure,
                  secure.grant.expiresAt < UInt64(Date().timeIntervalSince1970) + 7 * 86_400 else { return resolved }
            let client = HTTPBackend(host: resolved)
            let renewed: Renewal = try await client.post("/api/pairing/secure/renew", body: EmptyRenewal())
            let updatedPins = SecureHost(binding: secure.binding, namespace: secure.namespace, websocketURL: secure.websocketURL, grant: renewed.grant)
            let identity = try await SecureIdentityStore.shared.identity()
            try SecureAdmission.validate(host: updatedPins, device: identity.binding())
            guard renewed.grant.id == secure.grant.id, renewed.grant.epoch == secure.grant.epoch,
                  renewed.grant.expiresAt > secure.grant.expiresAt else { throw SecureConnectionError.authenticationFailed }
            var updated = resolved; updated.secure = updatedPins
            await SecureTransportPool.shared.replace(resolved, with: updated)
            return updated
        }
        guard Self.isEmbeddedLoopback(host.baseURL) else { throw SecureConnectionError.pairingRequired }
        return nil
    }
}

private struct Renewal: Decodable { let grant: SecureDeviceGrant }
private struct EmptyRenewal: Encodable {}

public extension HTTPBackend {
    /// Removing a host closes its authenticated channels and all pending operations.
    static func disconnect(host id: MoldHost.ID) async { await SecureTransportPool.shared.disconnect(id) }
}
