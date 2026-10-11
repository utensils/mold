import Foundation

/// A short-lived QR secret. Never persisted with a machine, sent to the cloud, or logged.
public struct SecureInvitation: Hashable, Sendable {
    public let binding: SecureBinding
    public let namespace: String
    public let websocketURL: URL
    public let seed: String
    public let invite: SecureInvite
    public init(binding: SecureBinding, namespace: String, websocketURL: URL, seed: String, invite: SecureInvite) {
        self.binding = binding; self.namespace = namespace; self.websocketURL = websocketURL; self.seed = seed; self.invite = invite
    }
    func validate() throws {
        try SecureCrypto.verify(binding)
        guard try SecureAdmission.namespace(binding) == namespace, websocketURL.scheme == "wss",
              websocketURL.absoluteString.utf8.count <= 2048, websocketURL.host?.isEmpty == false,
              websocketURL.user == nil, websocketURL.password == nil, websocketURL.fragment == nil, websocketURL.query == nil,
              websocketURL.port.map({ (1...65535).contains($0) }) ?? true,
              invite.expiresAt <= 9_007_199_254_740_991, invite.epoch <= 9_007_199_254_740_991 else { throw SecureConnectionError.authenticationFailed }
        let identity = try invitationIdentity()
        guard try identity.binding().signingKey == invite.signingKey else { throw SecureConnectionError.authenticationFailed }
        var bytes = Data("MOLD3-INVITE\0".utf8)
        bytes.append(try SecureHex.decode(namespace, count: 16)); bytes.append(try SecureHex.decode(invite.id, count: 16))
        bytes.append(try SecureHex.decode(invite.signingKey, count: 32))
        bytes.append(SecureHex.u64(invite.expiresAt)); bytes.append(SecureHex.u64(invite.epoch))
        try SecureCrypto.verify(key: binding.signingKey, message: bytes, signature: invite.signature)
    }
    func invitationIdentity() throws -> SecureIdentity {
        // Only the Ed seed signs admission/claim. This temporary identity never enters Noise.
        var secret = try SecureHex.decode(seed, count: 32); secret.append(Data(repeating: 0, count: 32))
        return try SecureIdentity(secret: secret)
    }
    func claimProof(handshakeHash: Data, device: SecureBinding) throws -> String {
        guard handshakeHash.count == 32 else { throw SecureConnectionError.authenticationFailed }
        var bytes = Data("MOLD3-CLAIM\0".utf8)
        bytes.append(try SecureHex.decode(namespace, count: 16)); bytes.append(try SecureHex.decode(invite.id, count: 16))
        bytes.append(handshakeHash); bytes.append(try SecureHex.decode(device.signingKey, count: 32)); bytes.append(try SecureHex.decode(device.noiseKey, count: 32))
        return try SecureHex.encode(invitationIdentity().sign(bytes))
    }
}
