import Foundation
import Testing
@testable import MoldClient

struct PairingClaimTests {
    @Test func claimProofIsBoundToHandshakeAndDeviceIdentity() throws {
        let payload = try #require(PairingLinkContractTests.fixtures.links.first).expected
        let invitation = try #require(payload.secureInvitation)
        let device = try SecureIdentity().binding()
        let proof = try invitation.claimProof(handshakeHash: Data(repeating: 7, count: 32), device: device)
        #expect(proof.count == 128)
        #expect(proof != (try invitation.claimProof(handshakeHash: Data(repeating: 8, count: 32), device: device)))
        #expect(proof != (try invitation.claimProof(handshakeHash: Data(repeating: 7, count: 32), device: SecureIdentity().binding())))
    }
    @Test func expiredCodeIsRefusedBeforeOpeningAnySocket() async throws {
        let payload = try #require(PairingLinkContractTests.fixtures.links.first).expected
        await #expect(throws: PairingClaimError.expiredOrUsed) {
            try await HTTPBackend.claimPairing(payload, clientName: "Fixture", clientKind: "iphone", now: Date(timeIntervalSince1970: 4_000_000_001))
        }
    }
    @Test func legacyClaimCannotFallBackToHTTP() async {
        let payload = MobilePairingPayload(baseURL: "http://untrusted.invalid", token: "never-send", expiresAt: nil, instanceId: "legacy", name: "Legacy")
        await #expect(throws: SecureConnectionError.pairingRequired) {
            try await HTTPBackend.claimPairing(payload, clientName: "Fixture", clientKind: "iphone")
        }
    }
}
