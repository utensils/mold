import Foundation

public extension SecretStore {
    /// The credential the in-process engine is started with.
    ///
    /// The precedence is `SecretStore::local_server_api_key`'s
    /// (`desktop/src-tauri/src/secrets.rs:103-119`): an explicit non-empty
    /// `MOLD_API_KEY` is the operator's override and wins; otherwise whatever
    /// this install already minted, which survives updates and relaunches;
    /// otherwise a fresh UUID, STORED before it is handed back so the next
    /// launch finds the same one.
    ///
    /// An empty `MOLD_API_KEY` is an unset variable spelled out, not a request
    /// for a keyless engine -- a keyless engine is what review 05-H1 is about.
    func localEngineAPIKey(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> String {
        if let override = environment["MOLD_API_KEY"], !override.isEmpty { return override }
        if let stored = try value(for: Self.localEngineKeyName), !stored.isEmpty { return stored }
        let minted = UUID().uuidString
        try set(minted, for: Self.localEngineKeyName)
        return minted
    }
}
