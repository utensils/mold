import Foundation
import MoldClient
import Testing

/// The phone's keys live ONLY in the Keychain. Hosted in the app, because the
/// Keychain answers per signed app; each test uses its own service name so a
/// run can never touch the real `remote-api-key` items.
@Suite(.serialized)
struct KeychainCredentialStoreTests {
    private func store() -> KeychainCredentialStore {
        KeychainCredentialStore(service: "io.utensils.mold.companion.tests.\(UUID().uuidString)")
    }

    @Test func roundTripsReplacesAndClears() throws {
        let keys = store()
        let host = UUID()
        #expect(try keys.apiKey(for: host) == nil)
        try keys.setAPIKey("mold_pair_one", for: host)
        #expect(try keys.apiKey(for: host) == "mold_pair_one")
        try keys.setAPIKey("mold_pair_two", for: host)
        #expect(try keys.apiKey(for: host) == "mold_pair_two")
        try keys.clearAPIKey(for: host)
        #expect(try keys.apiKey(for: host) == nil)
    }

    @Test func machinesAreKeptApart() throws {
        let keys = store()
        let first = UUID(), second = UUID()
        try keys.setAPIKey("a", for: first)
        try keys.setAPIKey("b", for: second)
        #expect(try keys.apiKey(for: first) == "a")
        #expect(try keys.apiKey(for: second) == "b")
        try keys.clearAPIKey(for: first)
        try keys.clearAPIKey(for: second)
    }

    /// The `CredentialStore` contract: an empty key is a clear, never a "".
    @Test func anEmptyKeyIsAClear() throws {
        let keys = store()
        let host = UUID()
        try keys.setAPIKey("k", for: host)
        try keys.setAPIKey("", for: host)
        #expect(try keys.apiKey(for: host) == nil)
    }

    @Test func clearingAKeyThatIsNotThereIsFine() throws {
        #expect(throws: Never.self) { try store().clearAPIKey(for: UUID()) }
    }
}
