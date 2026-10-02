import Foundation
import MoldClient

/// The machine list on disk: names and addresses, and which is the Default.
/// NEVER a key -- those are Keychain-only (`KeychainCredentialStore`). It
/// lives in the App Group so the widget can name a machine without asking
/// the app.
struct HostList: Codable, Equatable {
    struct Entry: Codable, Equatable {
        let id: UUID
        var name: String
        var baseURL: URL
        var connectionEndpoints: [ConnectionEndpoint]? = nil
        var connectionInstanceID: String? = nil
        var connectionOriginalURL: URL? = nil
    }

    var entries: [Entry] = []
    var defaultID: UUID?
}

/// Reads and writes `hosts.json`. A file that exists but will not parse is
/// moved aside once, never written over: losing someone's machine list to a
/// transient read error is not an acceptable failure mode.
struct HostListFile: Sendable {
    let url: URL

    static var shared: HostListFile {
        let root = AppGroup.container
            ?? URL.applicationSupportDirectory.appending(path: "io.utensils.mold.companion")
        return HostListFile(url: root.appending(path: "hosts.json"))
    }

    func load() -> HostList {
        guard let data = try? Data(contentsOf: url) else { return HostList() }
        if let list = try? JSONDecoder().decode(HostList.self, from: data) { return list }
        let parked = url.deletingPathExtension().appendingPathExtension("corrupt.json")
        try? FileManager.default.removeItem(at: parked)
        try? FileManager.default.moveItem(at: url, to: parked)
        return HostList()
    }

    func save(_ list: HostList) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(list).write(to: url, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
    }
}
