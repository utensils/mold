import Foundation

/// Reading a pairing code: a port of `parseMobilePairingPayload`
/// (`studio/api/pairing.ts`, both held to `studio/api/pairing.fixtures.json`).
/// Two forms are accepted -- the JSON payload and a link: the
/// `https://utensils.io/mold/pair#...` universal link the Mac, desktop and
/// web produce -- and
/// refused the same two ways: not a pairing code at all, or one this version
/// cannot use.
///
/// One deliberate difference: `expires_at` must be a non-negative whole number
/// of seconds (what every producer writes). The TypeScript accepts any finite
/// number because JavaScript has no other kind.
extension MobilePairingPayload {
    public enum ParseError: Error, Equatable, LocalizedError {
        case notPairingCode
        case unsupported

        public var errorDescription: String? {
            switch self {
            case .notPairingCode: "That QR code is not a Mold pairing code."
            case .unsupported: "That QR code is not a supported Mold pairing code."
            }
        }
    }

    public static func parse(_ raw: String) throws(ParseError) -> MobilePairingPayload {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        let fields: [String: Any]
        if let data = text.data(using: .utf8),
           let object = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) {
            guard let dict = object as? [String: Any] else { throw .notPairingCode }
            fields = dict
        } else {
            fields = try linkFields(text)
        }
        return try validated(fields)
    }

    /// Seconds since 1970, as the server mints it. A code with no expiry (a
    /// keyless machine's) never expires.
    public func isExpired(at now: Date = .now) -> Bool {
        guard let expiresAt else { return false }
        return Double(expiresAt) <= now.timeIntervalSince1970
    }

    /// A link this parser reads -- for routing an opened URL to pairing
    /// before parsing it (and reporting what is wrong with it).
    public static func isPairingLink(_ url: URL) -> Bool {
        guard let parts = URLComponents(url: url, resolvingAgainstBaseURL: false) else { return false }
        return linkQuery(parts) != nil
    }

    private static func linkFields(_ text: String) throws(ParseError) -> [String: Any] {
        guard let parts = URLComponents(string: text), let query = linkQuery(parts) else { throw .notPairingCode }
        let items = FormURLEncoded.decode(query)
        func value(_ name: String) -> String? { items.first { $0.0 == name }?.1 }
        var fields: [String: Any] = ["type": "mold.mobile-pairing"]
        fields["version"] = value("version").flatMap(Double.init) ?? Double.nan
        fields["base_url"] = value("base_url")
        fields["token"] = value("token") ?? NSNull()
        fields["expires_at"] = value("expires_at").map { Double($0) ?? .nan } ?? NSNull()
        fields["instance_id"] = value("instance_id")
        fields["name"] = value("name")
        fields["mode"] = value("mode")
        fields["pairing_token"] = value("pairing_token")
        for key in ["namespace", "host_key", "host_noise_key", "host_binding_signature", "websocket_url", "invite_seed"] {
            fields[key] = value(key)
        }
        if let raw = value("invite"), let data = raw.data(using: .utf8) {
            fields["invite"] = try? JSONSerialization.jsonObject(with: data)
        }
        if let raw = value("endpoints"), let data = raw.data(using: .utf8) {
            guard let list = try? JSONSerialization.jsonObject(with: data) else { throw .unsupported }
            fields["endpoints"] = list
        }
        return fields
    }

    /// The still form-encoded fields: the universal link's fragment (never
    /// its query -- that would reach the web server) or `mold://pair`'s query.
    private static func linkQuery(_ url: URLComponents) -> String? {
        guard url.user == nil, url.password == nil else { return nil }
        switch url.scheme?.lowercased() {
        case link.scheme:
            let path = url.path.hasSuffix("/") ? String(url.path.dropLast()) : url.path
            // `:443` is the default port and an empty `?` sends nothing:
            // both read as `pairing.ts`'s WHATWG URL does.
            guard url.host?.lowercased() == link.host, url.port == nil || url.port == 443, path == link.path,
                  url.query?.isEmpty ?? true else { return nil }
            return url.percentEncodedFragment ?? ""
        default:
            return nil
        }
    }

    private static func validated(_ f: [String: Any]) throws(ParseError) -> MobilePairingPayload {
        guard f["type"] as? String == "mold.mobile-pairing",
              let version = f["version"] as? NSNumber, CFGetTypeID(version) != CFBooleanGetTypeID(),
              version.doubleValue == 2,
              let base = f["base_url"] as? String,
              base.utf8.count <= 2048, let origin = URLComponents(string: base),
              ["http", "https"].contains(origin.scheme?.lowercased() ?? ""),
              origin.host?.isEmpty == false, origin.user == nil, origin.password == nil,
              origin.query == nil, origin.fragment == nil, origin.path.isEmpty || origin.path == "/",
              origin.port.map({ (1...65535).contains($0) }) ?? true,
              let instance = f["instance_id"] as? String,
              let name = f["name"] as? String
        else { throw .unsupported }

        let mode: HostConnectionMode
        if let raw = f["mode"] {
            guard let name = raw as? String, let parsed = HostConnectionMode(rawValue: name) else { throw .unsupported }
            mode = parsed
        } else { mode = .secure }
        if mode == .regular {
            let forbidden = ["websocket_url", "host_key", "host_noise_key", "host_binding_signature", "namespace", "invite_seed", "invite", "endpoints"]
            guard forbidden.allSatisfy({ f[$0] == nil || f[$0] is NSNull }),
                  let url = origin.url, DirectHostAddress.isValid(url),
                  let token = f["pairing_token"] as? String, token.utf8.count == 64,
                  token.allSatisfy({ $0.isASCII && $0.isHexDigit }),
                  let expiry = f["expires_at"] as? NSNumber, CFGetTypeID(expiry) != CFBooleanGetTypeID(),
                  expiry.doubleValue >= 0, expiry.doubleValue.isFinite,
                  expiry.doubleValue.rounded(.down) == expiry.doubleValue, expiry.doubleValue < Double(UInt64.max)
            else { throw .unsupported }
            return MobilePairingPayload(baseURL: base, token: token, expiresAt: expiry.uint64Value,
                                        instanceId: instance, name: name, mode: .regular)
        }

        guard let namespace = f["namespace"] as? String, let hostKey = f["host_key"] as? String,
              let noiseKey = f["host_noise_key"] as? String, let signature = f["host_binding_signature"] as? String,
              let seed = f["invite_seed"] as? String, let inviteValue = f["invite"],
              let inviteData = try? JSONSerialization.data(withJSONObject: inviteValue),
              let invite = try? MoldJSON.localDecoder.decode(SecureInvite.self, from: inviteData) else { throw .unsupported }
        let websocketURL: URL?
        if let value = f["websocket_url"], !(value is NSNull) {
            guard let raw = value as? String, let url = URL(string: raw) else { throw .unsupported }
            websocketURL = url
        } else { websocketURL = nil }
        let invitation = SecureInvitation(binding: .init(signingKey: hostKey, noiseKey: noiseKey, signature: signature),
                                          namespace: namespace, websocketURL: websocketURL, seed: seed, invite: invite)
        guard (try? invitation.validate()) != nil else { throw .unsupported }
        var endpoints: [ConnectionEndpoint]?
        if let value = f["endpoints"] {
            guard let data = try? JSONSerialization.data(withJSONObject: value),
                  let decoded = try? MoldJSON.decoder.decode([ConnectionEndpoint].self, from: data),
                  decoded.count <= 8, ConnectionRoutes.sanitized(decoded).count == decoded.count
            else { throw .unsupported }
            endpoints = decoded
        }
        return MobilePairingPayload(baseURL: base, token: nil, expiresAt: invite.expiresAt,
                                    instanceId: instance, name: name, endpoints: endpoints, secureInvitation: invitation)
    }
}
