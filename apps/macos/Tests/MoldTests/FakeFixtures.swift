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
        {"version": "0.29.0", "hostname": "fake", "busy": false, "uptime_secs": 0,
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

    /// One GPU, as `MachineStore` sees it. `DeviceInfo` has no public
    /// memberwise init either.
    static func deviceInfo(_ id: String, ordinal: Int, adminState: String = "enabled",
                           desiredEnabled: Bool = true) -> DeviceInfo {
        let json = """
        {"id": "\(id)", "name": "GPU \(ordinal)", "ordinal": \(ordinal), "device_kind": "full_gpu",
         "memory": {}, "telemetry": {}, "desired_enabled": \(desiredEnabled),
         "admin_state": "\(adminState)", "health": "healthy", "activity": "idle",
         "schedulable": true, "loaded_models": []}
        """
        return try! MoldJSON.decoder.decode(DeviceInfo.self, from: Data(json.utf8))
    }

    /// `GET /api/devices`'s envelope, wrapping already-built rows -- re-encoded
    /// through `MoldJSON.encoder` rather than typed out twice.
    static func deviceState(_ rows: [DeviceInfo] = [FakeFixtures.deviceInfo("cuda:0", ordinal: 0)]) -> DeviceState {
        let encoded = rows.map { String(data: try! MoldJSON.encoder.encode($0), encoding: .utf8)! }
            .joined(separator: ",")
        let json = #"{"plan_version": 1, "devices": [\#(encoded)]}"#
        return try! MoldJSON.decoder.decode(DeviceState.self, from: Data(json.utf8))
    }

    /// One 1 Hz sample, with only the GPUs a test needs to plant.
    static func resourceSnapshot(_ gpus: [(ordinal: Int, vramUsed: UInt64)]) -> ResourceSnapshot {
        let rows = gpus.map {
            #"{"ordinal": \#($0.ordinal), "vram_total": 1000, "vram_used": \#($0.vramUsed)}"#
        }.joined(separator: ",")
        let json = #"""
        {"hostname": "fake", "gpus": [\#(rows)], "system_ram": {"total": 1, "used": 1, "used_by_mold": 0}}
        """#
        return try! MoldJSON.decoder.decode(ResourceSnapshot.self, from: Data(json.utf8))
    }

    /// One `GET /api/discovery/peers` row. `DiscoveryPeer` has no public
    /// memberwise init either, so this decodes it the way the wire produces it.
    static func discoveryPeer(_ name: String, url: String, authRequired: Bool = false,
                              instanceId: String? = nil, isThisMachine: Bool = false) -> DiscoveryPeer {
        let json = """
        {"name": "\(name)", "url": "\(url)", "auth_required": \(authRequired),
         "instance_id": \(instanceId.map { "\"\($0)\"" } ?? "null"),
         "is_this_machine": \(isThisMachine)}
        """
        return try! MoldJSON.decoder.decode(DiscoveryPeer.self, from: Data(json.utf8))
    }
}
