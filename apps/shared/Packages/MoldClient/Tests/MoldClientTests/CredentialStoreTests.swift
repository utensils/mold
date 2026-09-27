import Foundation
import Testing

@testable import MoldClient

/// `CredentialStore` is the seam both apps put a machine's API key behind: the
/// Mac's owner-only `secrets.json` (`SecretStore`) and the phone's Keychain.
/// These drive the Mac store ONLY through the protocol, so a conformance that
/// quietly reaches for a different name, or swallows a clear, fails here.
struct CredentialStoreTests {
    private func store() throws -> any CredentialStore {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(path: "mold-credential-store-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return SecretStore(directory: dir)
    }

    @Test func roundTripsOneMachinesKey() throws {
        let credentials = try store()
        let host = UUID()
        #expect(try credentials.apiKey(for: host) == nil)
        try credentials.setAPIKey("mold_pair_abc", for: host)
        #expect(try credentials.apiKey(for: host) == "mold_pair_abc")
        try credentials.clearAPIKey(for: host)
        #expect(try credentials.apiKey(for: host) == nil)
    }

    @Test func keepsEachMachineApart() throws {
        let credentials = try store()
        let first = UUID()
        let second = UUID()
        try credentials.setAPIKey("one", for: first)
        try credentials.setAPIKey("two", for: second)
        try credentials.clearAPIKey(for: first)
        #expect(try credentials.apiKey(for: first) == nil)
        #expect(try credentials.apiKey(for: second) == "two")
    }

    /// `nil` means "no key" and nothing else, so an empty field saved through
    /// the protocol is a clear -- the rule `HostPersistence.setAPIKey` already
    /// keeps on the Mac. A blank key read back as `""` would be sent as an
    /// empty `X-Api-Key`.
    @Test func anEmptyKeyIsAClear() throws {
        let credentials = try store()
        let host = UUID()
        try credentials.setAPIKey("k", for: host)
        try credentials.setAPIKey("", for: host)
        #expect(try credentials.apiKey(for: host) == nil)
    }

    /// The protocol is a view over the Mac's existing names, not a new file
    /// format: a key saved through it is the one `SecretStore` already reads.
    @Test func secretStoreUsesItsPerHostName() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(path: "mold-credential-store-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let secrets = SecretStore(directory: dir)
        let host = UUID()
        try secrets.setAPIKey("k", for: host)
        #expect(try secrets.value(for: SecretStore.remoteAPIKeyName(for: host)) == "k")
    }
}
