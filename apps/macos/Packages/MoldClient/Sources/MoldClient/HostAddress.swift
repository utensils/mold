import Foundation

/// Turning what a person types into an address the client can use.
///
/// Swift twin of `desktop/src/lib/hosts.ts::normalizeHostUrl` and
/// `web/src/lib/hostRegistry.ts::normalizeHostAddress`, and it must agree with
/// them: the same box typed into three apps has to resolve to one origin, or
/// the same machine ends up in the list two or three times.
///
/// The rules, in the order they apply:
///
/// - A bare name or IP gets `http://` and mold's port, so `workstation` is enough.
/// - An explicit scheme or port is never overridden.
/// - A scheme's own default port is dropped, so `https://box:443` and
///   `https://box` are the same machine.
/// - Path, query and fragment are discarded -- pasting `.../api/status` out of
///   a browser is a normal thing to do and should just work.
///
/// Everything here is pure, which is what makes the table of real inputs in
/// `HostAddressTests` the specification rather than a description.
public enum HostAddress {
    /// The port `mold serve` binds by default.
    public static let defaultPort = 7680

    /// What went wrong, phrased for someone who is mid-typing.
    public enum Problem: Error, Equatable, Sendable {
        case empty
        case unsupportedScheme
        case unparseable

        public var message: String {
            switch self {
            case .empty:
                "Enter an address, like workstation or 10.0.0.5:7680."
            case .unsupportedScheme:
                "Mold speaks HTTP. Use http:// or https://, or leave the scheme off."
            case .unparseable:
                "That doesn't look like an address. Try workstation, 10.0.0.5:7680, or https://box.ts.net."
            }
        }
    }

    /// The address, or `nil` if it isn't one yet. Use ``resolve(_:)`` when you
    /// need to say why.
    public static func normalize(_ input: String) -> URL? {
        try? resolve(input)
    }

    public static func resolve(_ input: String) throws(Problem) -> URL {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw .empty }

        let hasScheme = trimmed.range(of: "^https?://", options: [.regularExpression, .caseInsensitive]) != nil
        // `ftp://workstation` would otherwise be prefixed into `http://ftp://workstation`,
        // parse as the host `ftp`, and silently point at the wrong machine.
        if !hasScheme, trimmed.contains("://") { throw .unsupportedScheme }

        let candidate = hasScheme ? trimmed : "http://\(bracketingBareIPv6(trimmed))"
        guard let parts = URLComponents(string: candidate),
              let scheme = parts.scheme?.lowercased(),
              let host = parts.percentEncodedHost, !host.isEmpty
        else { throw .unparseable }

        // A schemeless entry is someone naming a machine, not an origin, so it
        // gets mold's port. An explicit `http://workstation` is a complete URL and
        // keeps port 80, exactly as a browser would read it.
        var port = parts.port
        if !hasScheme, port == nil { port = defaultPort }
        if port == defaultPort(for: scheme) { port = nil }

        let authority = host.lowercased() + (port.map { ":\($0)" } ?? "")
        guard let url = URL(string: "\(scheme)://\(authority)") else { throw .unparseable }
        return url
    }

    /// A name to offer for a machine at this address, before it has told us
    /// its own hostname. `.local` comes off because Bonjour's suffix is
    /// plumbing, not what anyone calls the box.
    public static func suggestedName(for url: URL) -> String {
        guard var host = url.host(percentEncoded: false), !host.isEmpty else { return "" }
        if host.hasPrefix("["), host.hasSuffix("]") { host = String(host.dropFirst().dropLast()) }
        if host.hasSuffix(".local"), host.count > ".local".count {
            host = String(host.dropLast(".local".count))
        }
        return host
    }

    /// How an address reads in a list: the authority, with the scheme hidden
    /// when it is the ordinary one.
    public static func displayString(for url: URL) -> String {
        let text = url.absoluteString
        return text.hasPrefix("http://") ? String(text.dropFirst("http://".count)) : text
    }

    /// True when two addresses name the same origin.
    public static func sameOrigin(_ lhs: URL, _ rhs: URL) -> Bool {
        normalize(lhs.absoluteString) == normalize(rhs.absoluteString)
    }

    private static func defaultPort(for scheme: String) -> Int? {
        switch scheme {
        case "http": 80
        case "https": 443
        default: nil
        }
    }

    /// `::1` is an address; `workstation:7680` is a host and a port. More than one
    /// colon with no brackets is the only thing that tells them apart.
    private static func bracketingBareIPv6(_ text: String) -> String {
        guard !text.hasPrefix("["), text.filter({ $0 == ":" }).count > 1 else { return text }
        return "[\(text)]"
    }
}
