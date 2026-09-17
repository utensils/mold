import Foundation

// Devices, their moving telemetry, and the peers a host can see on its own
// network. Split from `HTTPBackend+Work.swift` for size -- this is a machine
// concern, not queued work.
extension HTTPBackend {
    public func devices() async throws -> DeviceState { try await get("/api/devices") }

    /// Answers with the device's NEW state on both the 200 that reached the
    /// requested state and the 202 that is draining or starting, so a caller
    /// never needs a follow-up listing to find out what happened.
    @discardableResult
    public func setDevice(_ id: String, enabled: Bool) async throws -> DeviceInfo {
        try await send(deviceMutationPath(id), method: "PATCH", body: DeviceMutation(enabled: enabled))
    }

    /// Split out so the URL construction can be pinned without a network call
    /// -- `id` is OPAQUE and must ride as one path component.
    func deviceMutationPath(_ id: String) -> String {
        "/api/devices/\(escaped(id))"
    }

    /// 503 for about a second after the server boots, which is "not yet",
    /// not a fault -- the caller decides.
    public func resources() async throws -> ResourceSnapshot { try await get("/api/resources") }

    /// This machine's own telemetry, as it samples it.
    ///
    /// A telemetry snapshot is a whole picture of one moment, and a stale one
    /// is worth nothing beside a fresh one -- so the policy is `latestOnly`
    /// (`StreamBuffering`).
    public func resourceStream() -> AsyncThrowingStream<ResourceSnapshot, Error> {
        AsyncThrowingStream(bufferingPolicy: .bufferingNewest(StreamBuffering.latestOnly)) { continuation in
            let task = Task {
                do {
                    for try await frame in stream("/api/resources/stream", timeout: 86_400) {
                        // `ping` keepalives fall here.
                        guard frame.name == "snapshot",
                              let data = frame.data.data(using: .utf8),
                              let snapshot = try? MoldJSON.decoder.decode(
                                  ResourceSnapshot.self, from: data)
                        else { continue }
                        continuation.yield(snapshot)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    public func peers() async throws -> [DiscoveryPeer] { try await get("/api/discovery/peers") }
}
