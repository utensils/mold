import Foundation

/// `moldstudio://` links from widgets, notifications and the Live Activity
/// (DESIGN.md §A). The Tauri app owns `mold://`; this app never registers it.
nonisolated enum DeepLink: Equatable {
    case print(host: UUID, filename: String)
    case queue(job: String?)
    case generate(inbox: String?)

    static let scheme = "moldstudio"

    var url: URL {
        var parts = URLComponents()
        parts.scheme = Self.scheme
        switch self {
        case let .print(host, filename):
            parts.host = "print"
            parts.path = "/\(host.uuidString)/\(filename)"
        case let .queue(job):
            parts.host = "queue"
            if let job { parts.path = "/\(job)" }
        case let .generate(inbox):
            parts.host = "generate"
            if let inbox { parts.queryItems = [URLQueryItem(name: "inbox", value: inbox)] }
        }
        return parts.url!
    }

    init?(_ url: URL) {
        guard url.scheme == Self.scheme, let parts = URLComponents(url: url, resolvingAgainstBaseURL: false)
        else { return nil }
        let path = parts.path.split(separator: "/", maxSplits: 1).map(String.init)
        switch parts.host {
        case "print":
            guard path.count == 2, let host = UUID(uuidString: path[0]), !path[1].isEmpty else { return nil }
            self = .print(host: host, filename: path[1])
        case "queue":
            self = .queue(job: path.first)
        case "generate":
            self = .generate(inbox: parts.queryItems?.first { $0.name == "inbox" }?.value)
        default:
            return nil
        }
    }
}
