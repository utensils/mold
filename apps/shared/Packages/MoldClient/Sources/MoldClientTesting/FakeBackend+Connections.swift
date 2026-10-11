import MoldClient

/// Additive connection methods keep the legacy nil default unless a test
/// explicitly exercises routing. Protocol defaults otherwise bypass stubs.
extension FakeBackend {
    public func resolvedConnection() async throws -> MoldHost? {
        guard state.withLock({ $0.handlers["resolvedConnection()"] != nil }) else { return nil }
        return try await respond("resolvedConnection()", [])
    }

    public func connectionAddresses() async throws -> ConnectionAddresses? {
        guard state.withLock({ $0.handlers["connectionAddresses()"] != nil }) else { return nil }
        return try await respond("connectionAddresses()", [])
    }
}
