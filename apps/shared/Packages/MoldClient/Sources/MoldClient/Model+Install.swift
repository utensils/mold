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
    /// takes (design fact 2, M5). Reads `catalogPrefixes`, so the namespace
    /// check has one spelling.
    static func isCatalogName(_ name: String) -> Bool {
        guard let colon = name.firstIndex(of: ":") else { return false }
        return catalogPrefixes.contains(String(name[..<colon]))
    }
}
