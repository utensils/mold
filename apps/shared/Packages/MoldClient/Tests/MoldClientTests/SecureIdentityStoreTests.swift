import Foundation
import Testing
@testable import MoldClient

struct SecureIdentityStoreTests {
    @Test func installationIdentitySurvivesIndependentStoreInstances() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let credentials = SecretStore(directory: directory)
        let first = try await SecureIdentityStore(credentials: credentials).identity()
        let second = try await SecureIdentityStore(credentials: credentials).identity()
        #expect(try first.binding() == second.binding())
        #expect(try first.secret() == second.secret())
    }
    @Test func corruptStoredIdentityCannotBeSilentlyReplaced() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let credentials = SecretStore(directory: directory)
        try credentials.setAPIKey("corrupt", for: SecureIdentityStore.identitySlot)
        await #expect(throws: SecureConnectionError.authenticationFailed) {
            try await SecureIdentityStore(credentials: credentials).identity()
        }
        #expect(try credentials.apiKey(for: SecureIdentityStore.identitySlot) == "corrupt")
    }
    @Test func lockedStoreReadFailsClosedBeforeWritingANewIdentity() async {
        await #expect(throws: LockedCredentials.Failure.locked) {
            try await SecureIdentityStore(credentials: LockedCredentials()).identity()
        }
    }
}
private struct LockedCredentials: CredentialStore {
    enum Failure: Error { case locked, attemptedWrite }
    func apiKey(for host: UUID) throws -> String? { throw Failure.locked }
    func setAPIKey(_ key: String, for host: UUID) throws { throw Failure.attemptedWrite }
    func clearAPIKey(for host: UUID) throws { throw Failure.attemptedWrite }
}
