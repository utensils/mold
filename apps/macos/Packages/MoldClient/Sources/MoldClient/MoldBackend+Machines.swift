import Foundation

/// Device lifecycle and this machine's own resource telemetry and peers.
public protocol MoldMachinesBackend: Sendable {
    func devices() async throws -> DeviceState
    /// Answers with the device's new state on both the 200 that reached it
    /// and the 202 that is draining or starting.
    @discardableResult
    func setDevice(_ id: String, enabled: Bool) async throws -> DeviceInfo
    /// 503 for about a second after the server boots -- not yet, not a fault.
    func resources() async throws -> ResourceSnapshot
    func resourceStream() -> AsyncThrowingStream<ResourceSnapshot, Error>
    /// What this machine can see on its own network. Per host, never merged.
    func peers() async throws -> [DiscoveryPeer]
}
