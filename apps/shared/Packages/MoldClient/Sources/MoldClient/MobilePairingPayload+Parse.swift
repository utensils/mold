import Foundation

/// Reading a pairing code: a port of `parseMobilePairingPayload`
/// (`studio/api/pairing.ts`). Two forms are accepted -- the JSON payload and
/// the `mold://pair?...` link the Mac, desktop and web all produce -- and
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

    private static func linkFields(_ text: String) throws(ParseError) -> [String: Any] {
        guard let url = URLComponents(string: text),
              url.scheme?.lowercased() == "mold", url.host?.lowercased() == "pair",
              url.user == nil, url.password == nil, url.fragment == nil
        else { throw .notPairingCode }
        func value(_ name: String) -> String? { url.queryItems?.first { $0.name == name }?.value }
        var fields: [String: Any] = ["type": "mold.mobile-pairing"]
        fields["version"] = value("version").flatMap(Double.init) ?? Double.nan
        fields["base_url"] = value("base_url")
        fields["token"] = value("token") ?? NSNull()
        fields["expires_at"] = value("expires_at").map { Double($0) ?? .nan } ?? NSNull()
        fields["instance_id"] = value("instance_id")
        fields["name"] = value("name")
        return fields
    }

    private static func validated(_ f: [String: Any]) throws(ParseError) -> MobilePairingPayload {
        guard f["type"] as? String == "mold.mobile-pairing",
              let version = f["version"] as? NSNumber, CFGetTypeID(version) != CFBooleanGetTypeID(),
              version.doubleValue == 1,
              let base = f["base_url"] as? String,
              base.range(of: "^https?://", options: [.regularExpression, .caseInsensitive]) != nil,
              let instance = f["instance_id"] as? String,
              let name = f["name"] as? String
        else { throw .unsupported }

        let token: String?
        switch f["token"] {
        case nil, is NSNull: token = nil
        case let value as String: token = value
        default: throw .unsupported
        }

        let expiresAt: UInt64?
        switch f["expires_at"] {
        case nil, is NSNull: expiresAt = nil
        // A JSON `true` is an NSNumber too, and `is Bool` says yes to any 0 or
        // 1 -- only the CoreFoundation type tells a real boolean apart.
        case let number as NSNumber where CFGetTypeID(number) != CFBooleanGetTypeID():
            let seconds = number.doubleValue
            guard seconds.isFinite, seconds >= 0, seconds == seconds.rounded(), seconds < 1.8e19 else {
                throw .unsupported
            }
            expiresAt = UInt64(seconds)
        default: throw .unsupported
        }
        return MobilePairingPayload(baseURL: base, token: token, expiresAt: expiresAt,
                                    instanceId: instance, name: name)
    }
}
