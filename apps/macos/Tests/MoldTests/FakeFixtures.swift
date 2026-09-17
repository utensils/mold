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

    /// `state` defaults to `"queued"` -- the ordinary waiting row. This used
    /// to default to `"accepted"`, a string `/api/queue` never actually sends
    /// (design M6 fact 1): every test that didn't override `state` was
    /// silently exercising `.unknown`, not a live row.
    static func queueEntry(
        _ id: String, state: String = "queued", batchId: String? = nil,
        clientBatchId: String? = nil, batchIndex: Int? = nil
    ) -> QueueEntry {
        let json = #"""
        {"id": "\#(id)", "state": "\#(state)",
         "batch_id": \#(batchId.map { "\"\($0)\"" } ?? "null"),
         "client_batch_id": \#(clientBatchId.map { "\"\($0)\"" } ?? "null"),
         "batch_index": \#(batchIndex.map { "\($0)" } ?? "null")}
        """#
        return try! MoldJSON.decoder.decode(QueueEntry.self, from: Data(json.utf8))
    }

    /// A listing with no `liveOnlyEntries` -- `QueueListing`'s own init is not
    /// public, so a planted answer is decoded the way the wire produces one.
    static func queueListing(_ ids: [String]) -> QueueListing {
        let entries = ids.map { #"{"id": "\#($0)", "state": "queued"}"# }.joined(separator: ",")
        let json = #"{"entries": [\#(entries)], "liveOnlyEntries": null}"#
        return try! MoldJSON.decoder.decode(QueueListing.self, from: Data(json.utf8))
    }

    /// A listing built from already-decoded rows -- for a test that needs
    /// more than a bare id, such as a batch id. Round-trips each row through
    /// `QueueEntry`'s own `Encodable` conformance rather than a second copy
    /// of the JSON `queueEntry` already built.
    static func queueListing(entries: [QueueEntry]) -> QueueListing {
        let rows = entries.map { String(data: try! MoldJSON.encoder.encode($0), encoding: .utf8)! }
        let json = #"{"entries": [\#(rows.joined(separator: ","))], "liveOnlyEntries": null}"#
        return try! MoldJSON.decoder.decode(QueueListing.self, from: Data(json.utf8))
    }

    /// One child of a batch, decoded the way `/api/generation-batches/status`
    /// produces one -- `BatchChild` has no public memberwise init either.
    static func batchChild(
        _ jobId: String, state: String = "held", errorCode: String? = nil, revision: UInt64? = nil
    ) -> BatchChild {
        let json = #"""
        {"index": 0, "job_id": "\#(jobId)", "state": "\#(state)",
         "error_code": \#(errorCode.map { "\"\($0)\"" } ?? "null"),
         "revision": \#(revision.map { "\($0)" } ?? "null")}
        """#
        return try! MoldJSON.decoder.decode(BatchChild.self, from: Data(json.utf8))
    }
}

// The rest of the wire shapes a lifecycle test plants: a machine's answer to
// `/api/status`, what it says it can do, and the ticket a download starts
// with. Same reason as above -- none of these has a public memberwise init.
extension FakeFixtures {
    static func serverStatus(
        instanceId: String? = nil, modelsDiskTotal: UInt64? = nil, modelsDiskFree: UInt64? = nil
    ) -> ServerStatus {
        let disk = modelsDiskTotal.map { total in
            #"{"total_bytes": \#(total), "free_bytes": \#(modelsDiskFree ?? 0)}"#
        }
        let json = """
        {"version": "0.29.0", "hostname": "fake", "busy": false, "uptime_secs": 0,
         "instance_id": \(instanceId.map { "\"\($0)\"" } ?? "null"),
         "models_disk": \(disk ?? "null")}
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

    /// `capabilities.queue` -- absence of the whole block, or of any one
    /// field, is a definitive `false` for `canReorderQueue` /
    /// `canCancelAllQueued` / `canPauseOneJob` (design M6 decision 5), which
    /// is exactly why the defaults here are `false` rather than omitted.
    static func capabilities(
        canReorder: Bool = false, canCancelAll: Bool = false, canPauseJob: Bool = false
    ) -> Capabilities {
        let json = #"""
        {"queue": {"can_reorder": \#(canReorder), "can_cancel_all": \#(canCancelAll),
         "can_pause_job": \#(canPauseJob)}}
        """#
        return try! MoldJSON.decoder.decode(Capabilities.self, from: Data(json.utf8))
    }

    /// The whole `identity` block present or entirely absent -- absence is
    /// the definitive no `supportsIdentity` reads (`types.rs:11555-11557`).
    static func capabilities(identity: Bool) -> Capabilities {
        let json = identity
            ? #"{"identity": {"multi_photo": true, "max_photos": 4, "true_cfg": true}}"#
            : "{}"
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
    /// public memberwise init either. `prompt`, `stepsMax` and
    /// `supportsIdentity` are the only things that vary across the tests
    /// that need one at all -- `stepsMax` for a test pinning a stored default
    /// gets CLAMPED, not just applied.
    static func recipe(
        prompt: PromptRequirement = .required, stepsMax: Int = 100, supportsIdentity: Bool? = nil
    ) -> GenerationRecipe {
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
                          "supports_lora": null, "supports_identity": \#(supportsIdentity.map { "\($0)" } ?? "null"),
                          "supports_sequence": null,
                          "supports_extend": null, "supports_audio": null, "source_image": null}}
        """#
        return try! MoldJSON.decoder.decode(GenerationRecipe.self, from: Data(json.utf8))
    }

    /// A recipe's `RecipeCapabilities` block alone, for the tests that ask a
    /// group's own `isShown` a question rather than reading a whole recipe
    /// -- `RecipeCapabilities` has no public memberwise init either. Each
    /// feature flag becomes an `adjustable`/`hidden` `FeatureControl`
    /// (`nil` stays absent, which every reader treats as unavailable).
    static func recipeCapabilities(
        supportsAudio: Bool? = nil, supportsExtend: Bool? = nil,
        keyframes: Bool? = nil, audio: Bool? = nil, sourceVideo: Bool? = nil
    ) -> RecipeCapabilities {
        func feature(_ available: Bool?) -> String {
            guard let available else { return "null" }
            return #"{"mode": "\#(available ? "adjustable" : "hidden")", "required": false, "reason": null}"#
        }
        let json = """
        {"supports_audio": \(supportsAudio.map { "\($0)" } ?? "null"),
         "supports_extend": \(supportsExtend.map { "\($0)" } ?? "null"),
         "keyframes": \(feature(keyframes)), "audio": \(feature(audio)),
         "source_video": \(feature(sourceVideo))}
        """
        return try! MoldJSON.decoder.decode(RecipeCapabilities.self, from: Data(json.utf8))
    }

    static func exportOptions(_ formats: [String] = ["png"]) -> ExportOptions {
        let list = formats.map { "\"\($0)\"" }.joined(separator: ",")
        return try! MoldJSON.decoder.decode(ExportOptions.self, from: Data(#"{"formats": [\#(list)]}"#.utf8))
    }

    static func model(
        _ name: String, family: String = "flux", sizeGb: Double? = nil, downloaded: Bool? = nil,
        remainingDownloadBytes: Int? = nil, isLoaded: Bool? = nil, diskUsageBytes: Int? = nil,
        description: String? = nil
    ) -> Model {
        let json = #"""
        {"name": "\#(name)", "family": "\#(family)", "description": "\#(description ?? "\(name) — fake")",
         "size_gb": \#(sizeGb.map { "\($0)" } ?? "null"),
         "downloaded": \#(downloaded.map { "\($0)" } ?? "null"),
         "remaining_download_bytes": \#(remainingDownloadBytes.map { "\($0)" } ?? "null"),
         "is_loaded": \#(isLoaded.map { "\($0)" } ?? "null"),
         "disk_usage_bytes": \#(diskUsageBytes.map { "\($0)" } ?? "null")}
        """#
        return try! MoldJSON.decoder.decode(Model.self, from: Data(json.utf8))
    }

    static func downloadTicket(_ id: String) -> DownloadTicket {
        try! MoldJSON.decoder.decode(DownloadTicket.self, from: Data(#"{"id": "\#(id)"}"#.utf8))
    }

    /// `POST /api/catalog/:id/download`'s 202 -- `CatalogInstall` has no
    /// public memberwise init, so this decodes it the way the wire produces
    /// one. A `nil` primary with non-empty companions is the "only
    /// companions were missing" answer, not a failure (design fact 2, M5).
    static func catalogInstall(primary: String?, companions: [(name: String, jobId: String)] = []) -> CatalogInstall {
        let companionJSON = companions
            .map { #"{"name": "\#($0.name)", "job_id": "\#($0.jobId)"}"# }
            .joined(separator: ",")
        let json = #"""
        {"primary_job_id": \#(primary.map { "\"\($0)\"" } ?? "null"), "companion_jobs": [\#(companionJSON)]}
        """#
        return try! MoldJSON.decoder.decode(CatalogInstall.self, from: Data(json.utf8))
    }

    /// `DELETE /api/models/:model`'s answer -- `ModelRemoval` has no public
    /// memberwise init either.
    static func modelRemoval(
        removed: [String] = [], kept: [(component: String, usedBy: [String])] = [], freedBytes: Int64 = 0
    ) -> ModelRemoval {
        let keptJSON = kept.map {
            let usedByJSON = $0.usedBy.map { "\"\($0)\"" }.joined(separator: ",")
            return #"{"component": "\#($0.component)", "used_by": [\#(usedByJSON)]}"#
        }.joined(separator: ",")
        let removedJSON = removed.map { "\"\($0)\"" }.joined(separator: ",")
        let json = #"{"removed": [\#(removedJSON)], "kept": [\#(keptJSON)], "freed_bytes": \#(freedBytes)}"#
        return try! MoldJSON.decoder.decode(ModelRemoval.self, from: Data(json.utf8))
    }

    /// `GET /api/models/:model/components`'s answer, one row per component
    /// -- `ModelComponentsResponse` and `ModelComponentStatus` have no public
    /// memberwise init either.
    static func modelComponents(
        _ model: String, rows: [(kind: String, name: String, present: Bool)]
    ) -> ModelComponentsResponse {
        let rowsJSON = rows.map {
            #"""
            {"kind": "\#($0.kind)", "name": "\#($0.name)", "present": \#($0.present),
             "path": null, "repair_model": null, "options": null}
            """#
        }.joined(separator: ",")
        let json = #"{"model": "\#(model)", "components": [\#(rowsJSON)]}"#
        return try! MoldJSON.decoder.decode(ModelComponentsResponse.self, from: Data(json.utf8))
    }

    /// One component row with a path and a repair name -- what a MISSING
    /// component actually carries, which the plain three-field `rows:` above
    /// always leaves null.
    static func modelComponentRow(
        kind: String, name: String, present: Bool, path: String? = nil, repairModel: String? = nil,
        optionsCount: Int = 0
    ) -> ModelComponentStatus {
        let options = (0 ..< optionsCount).map {
            #"{"label": "option-\#($0)", "path": "/models/option-\#($0)", "present": true}"#
        }.joined(separator: ",")
        let json = #"""
        {"kind": "\#(kind)", "name": "\#(name)", "present": \#(present),
         "path": \#(path.map { "\"\($0)\"" } ?? "null"),
         "repair_model": \#(repairModel.map { "\"\($0)\"" } ?? "null"),
         "options": [\#(options)]}
        """#
        return try! MoldJSON.decoder.decode(ModelComponentStatus.self, from: Data(json.utf8))
    }

    /// Assembles a listing from already-built rows, for a test that needs a
    /// row `modelComponents(_:rows:)`'s plain tuple can't express -- a
    /// missing component's repair name, or the 103-option `transformer` slot
    /// measured on plato (design fact 4, M5).
    static func modelComponents(_ model: String, statuses: [ModelComponentStatus]) -> ModelComponentsResponse {
        let componentsData = try! MoldJSON.encoder.encode(statuses)
        let componentsJSON = String(data: componentsData, encoding: .utf8)!
        let json = #"{"model": "\#(model)", "components": \#(componentsJSON)}"#
        return try! MoldJSON.decoder.decode(ModelComponentsResponse.self, from: Data(json.utf8))
    }

    /// A frame from `GET /api/downloads/stream` -- `DownloadEvent` has no
    /// public memberwise init either. `listing` is what a `snapshot` frame
    /// carries; every other frame leaves it `nil`.
    static func downloadEvent(
        type: String, id: String? = nil, model: String? = nil,
        bytesDone: Int64? = nil, bytesTotal: Int64? = nil, error: String? = nil,
        listing: DownloadsListing? = nil
    ) -> DownloadEvent {
        let listingJSON = listing.map { String(data: try! MoldJSON.encoder.encode($0), encoding: .utf8)! } ?? "null"
        let json = """
        {"type": "\(type)", "id": \(id.map { "\"\($0)\"" } ?? "null"),
         "model": \(model.map { "\"\($0)\"" } ?? "null"), "position": null,
         "files_done": null, "files_total": null,
         "bytes_done": \(bytesDone.map { "\($0)" } ?? "null"),
         "bytes_total": \(bytesTotal.map { "\($0)" } ?? "null"),
         "current_file": null, "error": \(error.map { "\"\($0)\"" } ?? "null"),
         "listing": \(listingJSON)}
        """
        return try! MoldJSON.decoder.decode(DownloadEvent.self, from: Data(json.utf8))
    }

    /// `capabilities.licenses` -- a bare bool, not a block (design fact, M5).
    static func capabilities(licenses: Bool) -> Capabilities {
        let json = #"{"licenses": \#(licenses)}"#
        return try! MoldJSON.decoder.decode(Capabilities.self, from: Data(json.utf8))
    }

    /// `capabilities.catalog` -- absent (`available: false`) is a host that
    /// browses nothing, which `canBrowseCatalog` reads as a definitive no
    /// (design M5 S6).
    static func capabilities(catalog available: Bool, families: [String] = [], sort: [String] = []) -> Capabilities {
        guard available else { return try! MoldJSON.decoder.decode(Capabilities.self, from: Data("{}".utf8)) }
        let familiesJSON = families.map { "\"\($0)\"" }.joined(separator: ",")
        let sortJSON = sort.map { "\"\($0)\"" }.joined(separator: ",")
        let json = #"{"catalog": {"available": true, "families": [\#(familiesJSON)], "sort": [\#(sortJSON)]}}"#
        return try! MoldJSON.decoder.decode(Capabilities.self, from: Data(json.utf8))
    }

    /// One catalog row -- `CatalogEntry` has no public memberwise init
    /// either, so this decodes it the way the wire produces one.
    static func catalogEntry(
        id: String, name: String? = nil, family: String = "sd15", kind: String = "checkpoint",
        sizeBytes: Int64? = nil, downloadCount: Int64 = 0, rating: Double? = nil, nsfw: Bool = false,
        supported: Bool = true, installed: Bool = false, pageUrl: String? = nil,
        license: String? = nil, commercial: Bool? = nil, derivatives: Bool? = nil, differentLicense: Bool? = nil,
        author: String? = nil, description: String? = nil, thumbnailUrl: String? = nil,
        tags: [String] = [], trainedWords: [String] = [], companions: [String] = [],
        companionDetails: [(name: String, kind: String, repo: String?, sizeBytes: Int64?)] = []
    ) -> CatalogEntry {
        func str(_ value: String?) -> String { value.map { "\"\($0)\"" } ?? "null" }
        func bool(_ value: Bool?) -> String { value.map { "\($0)" } ?? "null" }
        let tagsJSON = tags.map { "\"\($0)\"" }.joined(separator: ",")
        let wordsJSON = trainedWords.map { "\"\($0)\"" }.joined(separator: ",")
        let companionsJSON = companions.map { "\"\($0)\"" }.joined(separator: ",")
        let detailsJSON = companionDetails.map {
            #"{"name": "\#($0.name)", "kind": "\#($0.kind)", "repo": \#(str($0.repo)), "size_bytes": \#($0.sizeBytes.map { "\($0)" } ?? "null")}"#
        }.joined(separator: ",")
        let json = """
        {"id": "\(id)", "source": "civitai", "source_id": "0", "name": "\(name ?? id)",
         "author": \(str(author)), "family": "\(family)", "kind": "\(kind)", "modality": "image",
         "size_bytes": \(sizeBytes.map { "\($0)" } ?? "null"), "download_count": \(downloadCount),
         "rating": \(rating.map { "\($0)" } ?? "null"), "likes": 0, "nsfw": \(nsfw),
         "thumbnail_url": \(str(thumbnailUrl)), "description": \(str(description)),
         "license": \(str(license)),
         "license_flags": {"commercial": \(bool(commercial)), "derivatives": \(bool(derivatives)),
                           "different_license": \(bool(differentLicense))},
         "tags": [\(tagsJSON)], "companions": [\(companionsJSON)], "companion_details": [\(detailsJSON)],
         "supported": \(supported), "installed": \(installed), "page_url": \(str(pageUrl)),
         "trained_words": [\(wordsJSON)]}
        """
        return try! MoldJSON.decoder.decode(CatalogEntry.self, from: Data(json.utf8))
    }

    /// `GET /api/catalog/search`'s answer, assembled from already-built rows
    /// -- `CatalogListing` has no public memberwise init either, so this
    /// re-encodes the entries and decodes the whole page the way the wire
    /// produces one (the same round-trip `modelComponents(_:statuses:)` uses).
    static func catalogListing(
        _ entries: [CatalogEntry], page: Int = 1, pageSize: Int = 20, total: Int? = nil,
        providerErrors: [(source: String, message: String)] = []
    ) -> CatalogListing {
        let entriesJSON = String(data: try! MoldJSON.encoder.encode(entries), encoding: .utf8)!
        let errorsJSON = providerErrors.map {
            #"{"source": "\#($0.source)", "message": "\#($0.message)", "code": null, "retry_after_seconds": null}"#
        }.joined(separator: ",")
        let json = """
        {"entries": \(entriesJSON), "page": \(page), "page_size": \(pageSize),
         "total": \(total ?? entries.count), "provider_errors": [\(errorsJSON)]}
        """
        return try! MoldJSON.decoder.decode(CatalogListing.self, from: Data(json.utf8))
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

    /// A batch's status built from already-decoded children -- for a test
    /// that needs a `revision` or an `errorCode`, which `BatchChildSpec`
    /// does not carry. Use `FakeFixtures.batchChild(...)` to build them.
    static func batchStatus(id: String = "batch-1", clientBatchId: String = "client-1",
                            children: [BatchChild]) -> BatchStatus {
        let rows = children.map { String(data: try! MoldJSON.encoder.encode($0), encoding: .utf8)! }
        let json = #"{"id": "\#(id)", "client_batch_id": "\#(clientBatchId)", "children": [\#(rows.joined(separator: ","))]}"#
        return try! MoldJSON.decoder.decode(BatchStatus.self, from: Data(json.utf8))
    }

    /// `POST /api/generation-batches/status`'s answer -- `BatchStatusListing`
    /// has no public memberwise init either. `missingBatchIds` is what the
    /// machine says it never heard of, among the ids a test asked about.
    static func batchStatusListing(
        _ batches: [BatchStatus], missingBatchIds: [String] = [], instanceId: String = "fake-instance"
    ) -> BatchStatusListing {
        let rows = batches.map { String(data: try! MoldJSON.encoder.encode($0), encoding: .utf8)! }
        let missing = missingBatchIds.map { "\"\($0)\"" }.joined(separator: ",")
        let json = #"""
        {"instance_id": "\#(instanceId)", "batches": [\#(rows.joined(separator: ","))],
         "missing": {"client_batch_ids": [], "batch_ids": [\#(missing)]}}
        """#
        return try! MoldJSON.decoder.decode(BatchStatusListing.self, from: Data(json.utf8))
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

    /// Live on plato (design fact 9): `hf` configured from the environment,
    /// masked `hf_••••hhml`; `civitai` not configured. The same relative-path
    /// trick as `configListing` -- this bundle cannot see `MoldClientTests`'
    /// own `RepoFixtures`.
    static func credentialsFixture(_ name: String = "credentials-plato.json") -> CatalogCredentialStatus {
        let fixtures = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Tests/MoldTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // apps/macos
            .appending(path: "Packages/MoldClient/Tests/MoldClientTests/Fixtures")
        let data = try! Data(contentsOf: fixtures.appending(path: name))
        return try! MoldJSON.decoder.decode(CatalogCredentialStatus.self, from: data)
    }

    /// A composed status for the cases the plato fixture doesn't cover (a
    /// token stored ON this machine, or nothing at all) -- `CatalogCredentialStatus`
    /// and `CatalogCredentialState` have no public memberwise init either.
    static func credentialStatus(
        hfConfigured: Bool = false, hfSource: String? = nil, hfMasked: String? = nil,
        civitaiConfigured: Bool = false, civitaiSource: String? = nil, civitaiMasked: String? = nil
    ) -> CatalogCredentialStatus {
        func state(_ configured: Bool, _ source: String?, _ masked: String?) -> String {
            let sourceJSON = source.map { "\"\($0)\"" } ?? "null"
            let maskedJSON = masked.map { "\"\($0)\"" } ?? "null"
            return #"{"configured": \#(configured), "source": \#(sourceJSON), "masked": \#(maskedJSON)}"#
        }
        let hfJSON = state(hfConfigured, hfSource, hfMasked)
        let civitaiJSON = state(civitaiConfigured, civitaiSource, civitaiMasked)
        let json = #"{"hf": \#(hfJSON), "civitai": \#(civitaiJSON)}"#
        return try! MoldJSON.decoder.decode(CatalogCredentialStatus.self, from: Data(json.utf8))
    }

    /// `DELETE /api/queue`'s answer -- `QueueCancelResult` has no public
    /// memberwise init either.
    static func queueCancelResult(_ cancelled: Int) -> QueueCancelResult {
        try! MoldJSON.decoder.decode(
            QueueCancelResult.self, from: Data(#"{"cancelled": \#(cancelled)}"#.utf8))
    }
}
