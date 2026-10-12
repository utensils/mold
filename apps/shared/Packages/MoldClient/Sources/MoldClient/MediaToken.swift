import Foundation

public struct MediaToken: Codable, Hashable, Sendable {
    public let token: String?
    public let expiresAt: UInt64?
    public let authRequired: Bool
}

public extension HTTPBackend {
    func playableMedia(for filename: String, trashed: Bool = false) async throws -> PlayableMedia {
        guard (host.connectionMode == .regular ? host.permitsRegularDirect : (host.secure != nil || Self.isEmbeddedLoopback(host.baseURL))) || injectedTransport != nil else {
            throw SecureConnectionError.pairingRequired
        }
        return PlayableMedia(backend: self, url: MediaURL(baseURL: host.baseURL).media(filename, trashed: trashed))
    }

    /// Kept for protocol fixtures that own a local file. Shipping HTTP playback must
    /// use playableMedia so AVPlayer cannot bypass encrypted transport or Range checks.
    func playableURL(for filename: String) async throws -> URL { try await playableURL(for: filename, trashed: false) }
    func playableURL(for filename: String, trashed: Bool) async throws -> URL {
        throw SecureConnectionError.pairingRequired
    }
}
