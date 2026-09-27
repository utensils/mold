import Foundation

/// One machine, as it is written to preferences.
///
/// Separate from `MoldHost` for two reasons. The API key is a credential and
/// travels through `SecretStore`'s owner-only file instead, so a preferences
/// file that leaks carries nothing. And the keys are spelled out rather than
/// derived from the property names: this is the app's own format, read back by
/// the app, and `MoldJSON`'s two snake_case strategies are not inverses --
/// `baseURL` is written as `base_url` and read back as `baseUrl`, which is nobody's
/// property. Spelling `base_url` here is also what lets this read the
/// preferences already sitting on disk.
public struct StoredHost: Codable, Hashable, Sendable {
    public let id: UUID
    public var name: String
    public var baseURL: URL

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case baseURL = "base_url"
    }

    public init(id: UUID, name: String, baseURL: URL) {
        self.id = id
        self.name = name
        self.baseURL = baseURL
    }

    public init(_ host: MoldHost) {
        self.init(id: host.id, name: host.name, baseURL: host.baseURL)
    }

    /// Rejoined with the key the secrets file was holding for it.
    public func host(apiKey: String?) -> MoldHost {
        MoldHost(id: id, name: name, baseURL: baseURL, apiKey: apiKey)
    }
}
