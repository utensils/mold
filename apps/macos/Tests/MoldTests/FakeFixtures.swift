import Foundation
import MoldClient

/// Two-line fixtures: this bundle sees only MoldClient's PUBLIC surface, and
/// neither `GalleryPrint`'s memberwise init nor `QueueEntry`'s is public, so a
/// print or a queue row is built the way the wire builds one -- by decoding it.
enum FakeFixtures {
    static func print(_ filename: String, prompt: String? = nil) -> GalleryPrint {
        let json = """
        {"filename": "\(filename)", "metadata": {"prompt": \(prompt.map { "\"\($0)\"" } ?? "null")},
         "timestamp": 1000}
        """
        return try! MoldJSON.decoder.decode(GalleryPrint.self, from: Data(json.utf8))
    }

    static func queueEntry(_ id: String, state: String = "accepted") -> QueueEntry {
        let json = #"{"id": "\#(id)", "state": "\#(state)"}"#
        return try! MoldJSON.decoder.decode(QueueEntry.self, from: Data(json.utf8))
    }

    /// A listing with no `liveOnlyEntries` -- `QueueListing`'s own init is not
    /// public, so a planted answer is decoded the way the wire produces one.
    static func queueListing(_ ids: [String]) -> QueueListing {
        let entries = ids.map { #"{"id": "\#($0)", "state": "accepted"}"# }.joined(separator: ",")
        let json = #"{"entries": [\#(entries)], "liveOnlyEntries": null}"#
        return try! MoldJSON.decoder.decode(QueueListing.self, from: Data(json.utf8))
    }
}

// The rest of the wire shapes a lifecycle test plants: a machine's answer to
// `/api/status`, what it says it can do, and the ticket a download starts
// with. Same reason as above -- none of these has a public memberwise init.
extension FakeFixtures {
    static func serverStatus(instanceId: String? = nil) -> ServerStatus {
        let json = """
        {"version": "0.29.0", "hostname": "fake", "busy": false,
         "instance_id": \(instanceId.map { "\"\($0)\"" } ?? "null")}
        """
        return try! MoldJSON.decoder.decode(ServerStatus.self, from: Data(json.utf8))
    }

    /// `events.available` is what `wantsEvents` reads, so it is the one field
    /// a lifecycle test cares about.
    static func capabilities(events: Bool) -> Capabilities {
        let json = #"{"events": {"available": \#(events)}}"#
        return try! MoldJSON.decoder.decode(Capabilities.self, from: Data(json.utf8))
    }

    static func exportOptions(_ formats: [String] = ["png"]) -> ExportOptions {
        let list = formats.map { "\"\($0)\"" }.joined(separator: ",")
        return try! MoldJSON.decoder.decode(ExportOptions.self, from: Data(#"{"formats": [\#(list)]}"#.utf8))
    }

    static func model(_ name: String, family: String = "flux") -> Model {
        let json = #"{"name": "\#(name)", "family": "\#(family)", "description": "\#(name) — fake"}"#
        return try! MoldJSON.decoder.decode(Model.self, from: Data(json.utf8))
    }

    static func downloadTicket(_ id: String) -> DownloadTicket {
        try! MoldJSON.decoder.decode(DownloadTicket.self, from: Data(#"{"id": "\#(id)"}"#.utf8))
    }
}
