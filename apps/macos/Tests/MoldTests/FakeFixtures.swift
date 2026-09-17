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

    /// `gallery.organize` is what `canOrganize` reads -- absence (an older
    /// host) is a separate case from an explicit `false`, but both mean no
    /// File-under group.
    static func capabilities(organize: Bool) -> Capabilities {
        let json = #"{"gallery": {"organize": \#(organize)}}"#
        return try! MoldJSON.decoder.decode(Capabilities.self, from: Data(json.utf8))
    }

    /// A host that HAS said something about prompt expansion. Omitting the
    /// whole `expand` key (rather than calling this) is how a test plants the
    /// "hasn't said" host `mayExpandPrompts` treats as unknown-not-no.
    static func expandCapabilities(
        configured: Bool = true, modelPresent: Bool? = true,
        remix: Bool? = true, model: String? = nil
    ) -> Capabilities {
        let json = #"""
        {"expand": {"configured": \#(configured),
         "model_present": \#(modelPresent.map { "\($0)" } ?? "null"),
         "backend": null, "remix": \#(remix.map { "\($0)" } ?? "null"),
         "model": \#(model.map { "\"\($0)\"" } ?? "null")}}
        """#
        return try! MoldJSON.decoder.decode(Capabilities.self, from: Data(json.utf8))
    }

    /// One recipe, decoded rather than built -- `GenerationRecipe` has no
    /// public memberwise init either. `prompt` and `stepsMax` are the only
    /// things that vary across the tests that need one at all -- the latter
    /// for a test pinning a stored default gets CLAMPED, not just applied.
    static func recipe(prompt: PromptRequirement = .required, stepsMax: Int = 100) -> GenerationRecipe {
        let json = #"""
        {"id": "r", "label": "R",
         "defaults": {"width": 1024, "height": 1024, "steps": 20, "guidance": 3.5,
                      "frames": null, "fps": null, "negative_prompt": null},
         "resolution": {"domain": "dynamic", "alignment": 16, "min_width": 256, "min_height": 256,
                        "max_pixels": null, "max_axis_pixels": null, "off_bucket": null, "aspect_groups": null},
         "steps": {"default": 20, "min": 1, "max": \#(stepsMax), "step": 1, "recommended": null, "mode": "adjustable", "note": null},
         "guidance": {"default": 3.5, "min": 0, "max": 10, "step": 0.1, "mode": "adjustable", "note": null},
         "temporal": null,
         "capabilities": {"prompt": {"mode": "\#(prompt.rawValue)", "reason": null}, "negative_prompt": null,
                          "output": null, "reference_images": null, "supports_strength": null,
                          "supports_lora": null, "supports_identity": null, "supports_sequence": null,
                          "supports_extend": null, "supports_audio": null, "source_image": null}}
        """#
        return try! MoldJSON.decoder.decode(GenerationRecipe.self, from: Data(json.utf8))
    }

    static func exportOptions(_ formats: [String] = ["png"]) -> ExportOptions {
        let list = formats.map { "\"\($0)\"" }.joined(separator: ",")
        return try! MoldJSON.decoder.decode(ExportOptions.self, from: Data(#"{"formats": [\#(list)]}"#.utf8))
    }

    static func model(
        _ name: String, family: String = "flux", sizeGb: Double? = nil, downloaded: Bool? = nil
    ) -> Model {
        let json = #"""
        {"name": "\#(name)", "family": "\#(family)", "description": "\#(name) — fake",
         "size_gb": \#(sizeGb.map { "\($0)" } ?? "null"),
         "downloaded": \#(downloaded.map { "\($0)" } ?? "null")}
        """#
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

    /// One child of a batch, as `batchStatus` composes it. `state` also
    /// decides whether `result` rides along -- a live or failed child has
    /// none.
    struct BatchChildSpec {
        let index: Int
        let jobId: String
        let state: String
        let seed: UInt64?
        let error: String?

        init(_ index: Int, jobId: String? = nil, state: String = "running",
             seed: UInt64? = nil, error: String? = nil) {
            self.index = index
            self.jobId = jobId ?? "job-\(index)"
            self.state = state
            self.seed = seed
            self.error = error
        }
    }

    /// A batch's status, decoded the way the wire produces one -- `BatchStatus`
    /// and `BatchChild` have no public memberwise init either.
    static func batchStatus(id: String = "batch-1", clientBatchId: String = "client-1",
                            _ children: [BatchChildSpec]) -> BatchStatus {
        let rows = children.map { child -> String in
            let result = child.state == "complete"
                ? #"{"filename": "\#(child.jobId).png", "seed": \#(child.seed.map { "\($0)" } ?? "null")}"#
                : "null"
            let errorJSON = child.error.map { "\"\($0)\"" } ?? "null"
            return #"{"index": \#(child.index), "job_id": "\#(child.jobId)", "state": "\#(child.state)", "error": \#(errorJSON), "result": \#(result)}"#
        }.joined(separator: ",")
        let json = #"{"id": "\#(id)", "client_batch_id": "\#(clientBatchId)", "children": [\#(rows)]}"#
        return try! MoldJSON.decoder.decode(BatchStatus.self, from: Data(json.utf8))
    }

    /// A real `GET /api/config` from plato: 63 entries, 16 `models.*` rows
    /// for two configured models (`flux-dev:q8`, `flux2-klein:q8`), every
    /// value `null`, `source:"db"` -- what "configured but nothing set"
    /// looks like on a real host. Loaded by a path relative to THIS file
    /// rather than `MoldClientTests`' own `RepoFixtures`: this app test
    /// bundle is a separate target and cannot see that package's fixtures
    /// or its resource bundle.
    static func configListing(fixture name: String = "config-plato.json") -> ConfigListing {
        let fixtures = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Tests/MoldTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // apps/macos
            .appending(path: "Packages/MoldClient/Tests/MoldClientTests/Fixtures")
        let data = try! Data(contentsOf: fixtures.appending(path: name))
        return try! MoldJSON.decoder.decode(ConfigListing.self, from: data)
    }
}
