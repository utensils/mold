import Foundation

/// A deliberate connection choice, never a fallback policy.
public enum HostConnectionMode: String, Codable, CaseIterable, Hashable, Sendable {
    case secure
    case regular

    public var title: String { self == .secure ? "Secure" : "Regular" }
    public var explanation: String {
        self == .secure
            ? "End-to-end encrypted. Works directly or through Mold proxy."
            : "Direct HTTP or HTTPS with an API key. Encryption depends on HTTPS; Mold proxy is unavailable."
    }
}

public enum DirectHostAddress {
    /// Direct WAN addresses are allowed. Credentials and routing data never belong in the URL.
    public static func isValid(_ url: URL, allowingPathPrefix: Bool = false) -> Bool {
        guard let parts = URLComponents(url: url, resolvingAgainstBaseURL: false),
              ["http", "https"].contains(parts.scheme?.lowercased() ?? ""),
              let name = parts.host?.lowercased(), !name.isEmpty,
              parts.user == nil, parts.password == nil, parts.query == nil, parts.fragment == nil,
              allowingPathPrefix || parts.path.isEmpty || parts.path == "/",
              parts.port.map({ (1...65535).contains($0) }) ?? true else { return false }
        let host = name.trimmingCharacters(in: CharacterSet(charactersIn: "."))
        return host != "mold-link.urandom.io" && !host.hasSuffix(".mold-link.urandom.io")
    }
}
