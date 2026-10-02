import Foundation

/// The host identity and its three read-everything probes.
public protocol MoldStatusBackend: Sendable {
    var host: MoldHost { get }

    func connectionAddresses() async throws -> ConnectionAddresses?
    func resolvedConnection() async throws -> MoldHost?

    func status() async throws -> ServerStatus
    func capabilities() async throws -> Capabilities
    func models() async throws -> [Model]
}

public extension MoldStatusBackend {
    func connectionAddresses() async throws -> ConnectionAddresses? { nil }
    func resolvedConnection() async throws -> MoldHost? { nil }
}
