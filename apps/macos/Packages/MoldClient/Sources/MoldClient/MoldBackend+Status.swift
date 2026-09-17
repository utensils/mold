import Foundation

/// The host identity and its three read-everything probes.
public protocol MoldStatusBackend: Sendable {
    var host: MoldHost { get }

    func status() async throws -> ServerStatus
    func capabilities() async throws -> Capabilities
    func models() async throws -> [Model]
}
