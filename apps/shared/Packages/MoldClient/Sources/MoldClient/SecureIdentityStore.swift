import Foundation

/// A reserved credential slot holds one installation identity, independently of machine
/// removal. Losing or refusing access to that slot must never silently mint a replacement.
actor SecureIdentityStore {
    static let identitySlot = UUID(uuidString: "80E2A5EC-488B-4FE4-8EA5-96E317620FCA")!
    #if os(iOS)
    static let shared = SecureIdentityStore(credentials: KeychainCredentialStore())
    #else
    static let shared = SecureIdentityStore(credentials: SecretStore.shared)
    #endif
    private let credentials: any CredentialStore
    private var cached: SecureIdentity?
    init(credentials: any CredentialStore) { self.credentials = credentials }
    func identity() throws -> SecureIdentity {
        if let cached { return cached }
        let identity: SecureIdentity
        if let encoded = try credentials.apiKey(for: Self.identitySlot) {
            guard let bytes = Data(base64Encoded: encoded), bytes.count == 64 else { throw SecureConnectionError.authenticationFailed }
            identity = try SecureIdentity(secret: bytes)
        } else {
            identity = try SecureIdentity()
            try credentials.setAPIKey(identity.secret().base64EncodedString(), for: Self.identitySlot)
        }
        cached = identity
        return identity
    }
}
