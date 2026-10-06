import Foundation
import MoldClient

/// The relay owner's credential is separate from any Mold API or pairing key.
struct ManagedRelayEnrollment: Codable, Equatable, Sendable {
    let hostId: String
    let token: String
    let publicUrl: URL
    let relayUrl: URL
    let expiresAt: UInt64

    static let controlOrigin = URL(string: "https://mold-link.urandom.io")!
    static let secretName = SecretStore.managedRelayOwnerName

    func validate(control: URL = Self.controlOrigin, now: Date = .now, requiresLiveLease: Bool = true) throws {
        guard hostId.range(of: "^[a-f0-9]{32}$", options: .regularExpression) != nil,
              token.range(of: "^[A-Za-z0-9_-]{43}$", options: .regularExpression) != nil,
              control.scheme == "https", let domain = control.host,
              publicUrl.scheme == "https", publicUrl.host == "\(hostId).\(domain)",
              publicUrl.port == nil, publicUrl.user == nil, publicUrl.password == nil,
              ["", "/"].contains(publicUrl.path), publicUrl.query == nil, publicUrl.fragment == nil,
              relayUrl.scheme == "wss", relayUrl.host != nil,
              relayUrl.user == nil, relayUrl.password == nil, relayUrl.query == nil, relayUrl.fragment == nil,
              !requiresLiveLease || TimeInterval(expiresAt) > now.timeIntervalSince1970
        else { throw ManagedRelayFailure.invalidEnrollment }
    }

    static func load(secrets: SecretStore = .shared) throws -> Self? {
        guard let value = try secrets.value(for: secretName) else { return nil }
        let record = try MoldJSON.decoder.decode(Self.self, from: Data(value.utf8))
        try record.validate(requiresLiveLease: false)
        return record
    }

    static func save(_ record: Self?, secrets: SecretStore = .shared) throws {
        guard let record else { try secrets.clear(secretName); return }
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let data = try encoder.encode(record)
        guard let value = String(data: data, encoding: .utf8) else { throw ManagedRelayFailure.invalidEnrollment }
        try secrets.set(value, for: secretName)
    }
}

enum ManagedRelayFailure: Error, LocalizedError {
    case unavailable, capacity, invalidEnrollment, engineNotRunning, connector, notReady
    var errorDescription: String? {
        switch self {
        case .unavailable: "Mold proxy is unavailable. Try again shortly."
        case .capacity: "Mold proxy is busy. Try pairing again shortly."
        case .invalidEnrollment: "Mold proxy returned an invalid connection. Try again."
        case .engineNotRunning: "Start This Mac’s engine in Settings ▸ This Mac, then try pairing again."
        case .connector: "This Mac couldn’t start its remote connection. Try again."
        case .notReady: "This Mac hasn’t connected to Mold proxy yet. Try again."
        }
    }
}
