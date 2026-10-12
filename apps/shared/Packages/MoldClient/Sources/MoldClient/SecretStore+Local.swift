import Foundation

public extension SecretStore {
    /// Explicit environment overrides win. Otherwise all hosting shells use
    /// one owner-private credential in this Mold home. The old app-local value
    /// is adopted only when the home has no canonical key, and is kept intact.
    func localEngineAPIKey(
        home: URL,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> String {
        if let override = environment["MOLD_API_KEY"], !override.isEmpty { return try SharedOperatorKey.explicit(override) }
        let legacy = try value(for: Self.localEngineKeyName).flatMap { $0.isEmpty ? nil : $0 }
        return try SharedOperatorKey.resolve(home: home, legacyCandidate: legacy)
    }
}
