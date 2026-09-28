import Foundation

/// Handoff of a print being looked at, between the Mac's Mold Studio and the
/// iPhone/iPad companion (`NSUserActivity`). The two devices name the same
/// machine differently -- the Mac's own server is `127.0.0.1` there and a LAN
/// address here, and host ids are per device -- so the activity carries the
/// server's run (`instance_id`) first and its address second.
public enum PrintHandoff {
    public static let activityType = "io.utensils.mold.print"

    public static func userInfo(filename: String, address: URL, instanceId: String?) -> [String: String] {
        var info = ["filename": filename, "address": address.absoluteString]
        if let instanceId { info["instance"] = instanceId }
        return info
    }

    /// The print on a machine this device knows, or `nil`: by the server's
    /// run where one answers with it, else by address.
    public static func resolve(
        _ info: [AnyHashable: Any], hosts: [MoldHost], instanceOf: (MoldHost.ID) -> String?
    ) -> PrintID? {
        guard let filename = info["filename"] as? String, !filename.isEmpty else { return nil }
        if let instance = info["instance"] as? String,
           let host = hosts.first(where: { instanceOf($0.id) == instance }) {
            return PrintID(host: host.id, filename: filename)
        }
        if let address = (info["address"] as? String).flatMap(URL.init(string:)),
           let host = hosts.first(where: { HostAddress.sameOrigin($0.baseURL, address) }) {
            return PrintID(host: host.id, filename: filename)
        }
        return nil
    }
}
