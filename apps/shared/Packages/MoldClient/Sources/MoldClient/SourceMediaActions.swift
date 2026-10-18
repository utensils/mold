import Foundation

/// A source address is meaningful only on its exact owning host copy.
public enum SourceMediaActions {
    public static func identifier(filename: String, member: RetainedSourceMedia.Member) -> String {
        "source:\(filename)#\(member.memberId)"
    }
    public static func shellQuote(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\"'\"'") + "'"
    }
    public static func command(filename: String, member: RetainedSourceMedia.Member, host: URL) -> String {
        // Never publish URL credentials, query tokens, or scoped relay URLs.
        var components = URLComponents(url: host, resolvingAgainstBaseURL: false)
        components?.user = nil; components?.password = nil; components?.query = nil; components?.fragment = nil
        let origin = components?.url?.absoluteString ?? ""
        return "mold --host \(shellQuote(origin)) library source-media \(shellQuote(filename)) --member \(shellQuote(member.memberId)) --output \(shellQuote(member.displayName))"
    }
}

/// An authenticated inventory describes retained files, not whether the job
/// once had inputs. Missing provenance keeps a legacy answer unavailable.
public enum SourceMediaState: Equatable, Sendable {
    case loading, none, available, unavailable, failed
    public init(inventory: RetainedSourceMedia.Inventory?, failed: Bool, metadata: OutputMetadata? = nil) {
        if failed { self = .failed }
        else if let inventory {
            if inventory.availability == .unavailableLegacy, let metadata, !RetainedSourceMedia.disclosable(metadata) { self = .none }
            else { self = inventory.availability == .available ? (inventory.members.isEmpty ? .none : .available) : .unavailable }
        } else { self = .loading }
    }
    public var canRetry: Bool { self == .failed || self == .unavailable }
}
