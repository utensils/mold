import Foundation

/// Where a model stands on ONE machine.
public enum ModelInstallState: Hashable, Sendable {
    /// Not on disk. The payload is what it would cost to fetch -- for a
    /// model nobody has started this is its WHOLE size, which is normal, not
    /// disrepair (`manifest.rs:4950-4968`).
    case available(Int?)
    /// On disk and incomplete: a companion or a shard is missing, and
    /// re-running the SAME install resumes it rather than starting over.
    case needsRepair(Int)
    case installed
    /// Installed and GPU-resident right now. A delete is refused while this
    /// is true (`routes.rs:5833-5843`).
    case loaded

    /// Order for a State column: what needs attention first.
    public var sortRank: Int {
        switch self {
        case .needsRepair: 0
        case .available: 1
        case .installed: 2
        case .loaded: 3
        }
    }
}

public extension Model {
    /// The one place the repair rule is written.
    ///
    /// Repair is `downloaded && remaining > 0` and NOTHING ELSE: a
    /// not-yet-installed model also carries a positive remainder in
    /// `remainingDownloadBytes` (it is its whole size), so reading the
    /// remainder alone without checking `downloaded` calls every available
    /// model broken (design fact 6, M5).
    var installState: ModelInstallState {
        if isLoaded == true { return .loaded }
        if downloaded == true {
            if let remaining = remainingDownloadBytes, remaining > 0 {
                return .needsRepair(remaining)
            }
            return .installed
        }
        return .available(remainingDownloadBytes)
    }

    /// True for `cv:…` / `hf:…`, which decides which install route a name
    /// takes (design fact 2, M5). Derived from `catalogPrefixes`, so there is
    /// one spelling of the namespace check.
    var isCatalogModel: Bool { Self.isCatalogName(name) }

    static func isCatalogName(_ name: String) -> Bool {
        guard let colon = name.firstIndex(of: ":") else { return false }
        return catalogPrefixes.contains(String(name[..<colon]))
    }

    /// The part before the em-dash: "FLUX.1 Dev Q4".
    ///
    /// The manifest already writes every description this way, so the app
    /// splits rather than inventing copy of its own. Mirrors `human_name`'s
    /// fallback order (`types.rs:4048-4076`): a catalog row with no
    /// `displayName` and an EMPTY description -- an `hf:` repo with nothing
    /// scraped, say -- falls to the same two tails the server derives from
    /// the name itself, rather than showing nothing.
    var headline: String {
        guard let range = description.range(of: " — ") else {
            if let displayName, !displayName.isEmpty { return displayName }
            if description.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                if name.hasPrefix("cv:") { return "Civitai model #\(name.dropFirst(3))" }
                if name.hasPrefix("hf:") {
                    let repo = name.dropFirst(3)
                    let title = repo.split(separator: "/").last.map(String.init) ?? String(repo)
                    return title.replacingOccurrences(of: "-", with: " ")
                        .replacingOccurrences(of: "_", with: " ")
                }
            }
            return description
        }
        return String(description[..<range.lowerBound])
    }
}
