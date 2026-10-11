import Foundation
import Testing
@testable import MoldClient

struct MediaTokenTests {
    @Test(arguments: [nil, "secret"] as [String?])
    func remotePlaybackNeverReturnsAnUnencryptedOrTicketedURL(key: String?) async {
        let backend = HTTPBackend(host: .init(name: "Legacy", baseURL: URL(string: "https://not-fetched.invalid")!, apiKey: key))
        await #expect(throws: SecureConnectionError.pairingRequired) { try await backend.playableURL(for: "clip.mp4") }
        await #expect(throws: SecureConnectionError.pairingRequired) { try await backend.playableURL(for: "clip.mp4", trashed: true) }
    }
}
