import Foundation
import MoldClient

/// Public routing metadata only. The engine owns its signing identity.
struct ManagedRelayEnrollment: Codable, Equatable, Sendable {
    let hostId: String
    let publicUrl: URL
    let relayUrl: URL
    var protocolVersion = 3

    static let controlOrigin = URL(string: "https://mold-link.urandom.io")!
    static let secretName = SecretStore.managedRelayOwnerName

    func validate(control: URL = Self.controlOrigin) throws {
        guard hostId.range(of: "^[a-f0-9]{32}$", options: .regularExpression) != nil,
              protocolVersion == 3,
              control.scheme == "https", let domain = control.host,
              publicUrl.scheme == "https", publicUrl.host == "\(hostId).\(domain)",
              publicUrl.port == nil, publicUrl.user == nil, publicUrl.password == nil,
              ["", "/"].contains(publicUrl.path), publicUrl.query == nil, publicUrl.fragment == nil,
              relayUrl.scheme == "wss", relayUrl.host != nil,
              relayUrl.user == nil, relayUrl.password == nil, relayUrl.query == nil, relayUrl.fragment == nil
        else { throw ManagedRelayFailure.invalidEnrollment }
    }

    static func load(secrets: SecretStore = .shared) throws -> Self? {
        guard let value = try secrets.value(for: secretName) else { return nil }
        // Hard cutover: erase the retired cloud bearer rather than retaining it.
        guard let record = try? MoldJSON.decoder.decode(Self.self, from: Data(value.utf8)),
              (try? record.validate()) != nil else {
            try secrets.clear(secretName)
            return nil
        }
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
    case unavailable, capacity, invalidEnrollment, engineNotRunning, connector, notReady, stopping
    var errorDescription: String? {
        switch self {
        case .unavailable: "Mold proxy is unavailable. Try again shortly."
        case .capacity: "Mold proxy is busy. Try pairing again shortly."
        case .invalidEnrollment: "Mold proxy returned an invalid connection. Try again."
        case .engineNotRunning: "Start This Mac’s engine in Settings ▸ This Mac, then try pairing again."
        case .connector: "This Mac couldn’t start its remote connection. Try again."
        case .notReady: "This Mac hasn’t connected to Mold proxy yet. Try again."
        case .stopping: "Remote access is stopping. Try pairing again when it finishes."
        }
    }
}
