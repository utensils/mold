import Foundation
import MoldClient

private final class PairingFixtureBundle: NSObject {}
enum SecurePairingFixture {
    static func session() throws -> PairingSession {
        let payload = try payload()
        let secure = payload.secureInvitation!
        var session = PairingSession(token: nil, expiresAt: payload.expiresAt, authRequired: true,
                                     instanceId: payload.instanceId, hostname: payload.name)
        session.hostKey = secure.binding.signingKey; session.hostNoiseKey = secure.binding.noiseKey
        session.hostBindingSignature = secure.binding.signature; session.namespace = secure.namespace
        session.websocketUrl = secure.websocketURL?.absoluteString; session.inviteSeed = secure.seed; session.invite = secure.invite
        return session
    }
    static func payload() throws -> MobilePairingPayload {
        let url = Bundle(for: PairingFixtureBundle.self).url(forResource: "pairing.fixtures", withExtension: "json")!
        let data = try Data(contentsOf: url)
        let root = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let links = root["links"] as! [[String: Any]]
        return try MobilePairingPayload.parse(links[0]["link"] as! String)
    }
}
