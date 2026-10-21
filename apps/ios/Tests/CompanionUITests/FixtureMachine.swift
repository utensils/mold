import CryptoKit
import CoreGraphics
import ImageIO
import Foundation
import Network
import Synchronization

/// A loopback-only machine for UI regression tests. It serves real generated
/// profiles and cannot generate or download. Optional collection and model-memory mutations
/// update only the fixture’s own in-memory state.
final class FixtureMachine: @unchecked Sendable {
    private let landscapePlaybackFixture: Bool
    private let playbackData: Data?
    let exportFixture: Bool
    private let unsupportedExportFormats: Bool
    private let capturedExports = Mutex<[Data]>([])
    var exportRequests: [Data] { capturedExports.withLock { $0 } }
    private let downloadedModels = Mutex<[String]>([])
    var installedRequests: [String] { downloadedModels.withLock { $0 } }
    private let listener: NWListener
    private let queue = DispatchQueue(label: "iphone-ui-fixture")
    private let models: Data
    let durationQueueFixture: Bool
    let durationCompleted = Mutex(false)
    private let keyboardSequenceFixture: Bool
    private let aspectFixture: Bool
    private let referenceFixture: Bool
    private let capturedGenerations = Mutex<[Data]>([])
    var generationRequests: [Data] { capturedGenerations.withLock { $0 } }
    private var trashGallery = Data("[]".utf8)
    private var gallery: Data
    private let libraryMutations: Bool
    private let removePrintOnFavorite: String?
    private let retainedMediaFixture: Bool
    private let sourceMediaLibraryFixture: Bool
    private let sourceAspectFixture: Bool
    private let sourceThumbnailFailures: Mutex<Int>
    private let eventConnections = Mutex<[NWConnection]>([])
    let retainedFrameFixture: Bool
    private let sourceFreeQueueFixture: Bool
    private let batchQueueFixture: Bool
    private let queueFixture: Bool
    private let unavailableGallery = Mutex(false)
    func setGalleryUnavailable(_ unavailable: Bool) { unavailableGallery.withLock { $0 = unavailable } }
    private let allRequests = Mutex<[String]>([])
    func requestLog() -> [String] { allRequests.withLock { $0 } }
    private let queueRequests = Mutex<[String]>([])
    func queueActionRequests() -> [String] { queueRequests.withLock { $0 } }
    private let memoryErrorFixture: String?
    private let queueDownloadFixture: Bool
    private let requiresDownloadLicense: Bool
    private var downloadLicenseAccepted = false
    private var queueDownloadStarted = false
    private var queueDownloadComplete = false
    private let queueFailureFixture: Bool
    private let enrichedQueue = Mutex(false)
    func enrichQueueFailure() { enrichedQueue.withLock { $0 = true } }
    private let promptExpansionFixture: Bool
    private let queueControls: Bool
    private let queueSwipeFixture: Bool
    private let queueDestinationFixture: Bool
    private var jobStates = ["fixture-video": "queued", "fixture-held": "held"]
    private var clearedHistory = false
    private let collectionFixture: Bool
    private let omitOrganizationCapabilities: Bool
    private let capturedLibraryMutations = Mutex<[Data]>([])
    var libraryMutationRequests: [Data] { capturedLibraryMutations.withLock { $0 } }
    private var collectionHidden = false
    private var createdCollections: [String: String] = [:]
    private let modelMemoryFixture: Bool
    private var residentModels: Set<String> = []

    init(justifiedFixture: Bool = false, landscapePlaybackFixture: Bool = false, exportFixture: Bool = false, unsupportedExportFormats: Bool = false, aspectFixture: Bool = false, referenceFixture: Bool = false, galleryPrints: Int = 0, galleryID: String? = nil, galleryFavorites: Int = 0, collectionFixture: Bool = false, hiddenCollectionFixture: Bool = false, omitOrganizationCapabilities: Bool = false, mixedMedia: Bool = false, queueFixture: Bool = false, retainedMediaFixture: Bool = false, retainedFrameFixture: Bool = false, loadedModels: Bool = false, queueControls: Bool = false, libraryMutations: Bool = false, removePrintOnFavorite: String? = nil, memoryErrorFixture: String? = nil, queueDownloadFixture: Bool = false, requiresDownloadLicense: Bool = false, trashFixture: Bool = false, queueFailureFixture: Bool = false, queueSwipeFixture: Bool = false, queueDestinationFixture: Bool = false, promptExpansionFixture: Bool = false, batchQueueFixture: Bool = false, sourceFreeQueueFixture: Bool = false, sourceMediaLibraryFixture: Bool = false, sourceAspectFixture: Bool = false, failFirstSourceThumbnail: Bool = false, playbackData: Data? = nil, keyboardSequenceFixture: Bool = false, durationQueueFixture: Bool = false) throws {
        self.durationQueueFixture = durationQueueFixture
        self.keyboardSequenceFixture = keyboardSequenceFixture
        self.landscapePlaybackFixture = landscapePlaybackFixture
        self.playbackData = playbackData
        self.exportFixture = exportFixture
        self.unsupportedExportFormats = unsupportedExportFormats
        self.aspectFixture = aspectFixture
        self.referenceFixture = referenceFixture
        self.removePrintOnFavorite = removePrintOnFavorite
        self.libraryMutations = libraryMutations
        self.collectionHidden = hiddenCollectionFixture
        self.retainedMediaFixture = retainedMediaFixture
        self.sourceMediaLibraryFixture = sourceMediaLibraryFixture
        self.sourceAspectFixture = sourceAspectFixture
        self.sourceThumbnailFailures = Mutex(failFirstSourceThumbnail ? 1 : 0)
        self.retainedFrameFixture = retainedFrameFixture
        self.sourceFreeQueueFixture = sourceFreeQueueFixture
        self.batchQueueFixture = batchQueueFixture
        self.queueFixture = queueFixture
        self.queueDownloadFixture = queueDownloadFixture
        self.requiresDownloadLicense = requiresDownloadLicense
        self.queueControls = queueControls
        self.promptExpansionFixture = promptExpansionFixture
        self.queueSwipeFixture = queueSwipeFixture
        self.queueDestinationFixture = queueDestinationFixture
        if queueSwipeFixture {
            jobStates["fixture-running"] = "running"
            jobStates["fixture-cancelling"] = "cancelling"
        }
        self.queueFailureFixture = queueFailureFixture
        self.memoryErrorFixture = memoryErrorFixture
        self.collectionFixture = collectionFixture
        self.omitOrganizationCapabilities = omitOrganizationCapabilities
        modelMemoryFixture = loadedModels
        if loadedModels { residentModels = ["flux-dev:q4", "ltx-2.5-22b-distilled:bf16"] }
        var root = URL(fileURLWithPath: #filePath)
        while root.pathComponents.count > 1,
              !FileManager.default.fileExists(atPath: root.appending(path: "docs/generated").path) {
            root.deleteLastPathComponent()
        }
        let document = try JSONSerialization.jsonObject(with: Data(contentsOf:
            root.appending(path: "docs/generated/generation-profiles-v1.json"))) as! [String: Any]
        let profiles = document["profiles"] as! [[String: Any]]
        let names = referenceFixture
            ? ["minimax-h3-ref2va:comfy-pruned-int8-turbo-4step", "minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p",
               "qwen-image-2.1:q8", "qwen-image-edit-2511:q4", "flux2-klein:bf16", "sdxl-base:fp16",
               "hunyuan3d-2mv:fp16", "wan22-ti2v-5b:fp16"]
            : ["flux-dev:q4", "ltx-2.5-22b-distilled:bf16"]
        models = try JSONSerialization.data(withJSONObject: names.map { name -> [String: Any] in
            let row = profiles.first { ($0["models"] as! [[String: Any]]).contains { $0["model"] as? String == name } }!
            return ["name": name, "family": (row["models"] as! [[String: Any]]).first { $0["model"] as? String == name }?["family"] ?? "unknown",
                    "description": name, "downloaded": !(queueFixture && name.hasPrefix("flux")), "display_name": referenceFixture ? name : name.hasPrefix("flux") ? "FLUX.1 Dev Q4" : "LTX-2.5 Distilled BF16", "hf_repo": name.hasPrefix("flux") ? "black-forest-labs/FLUX.1-dev" : "Lightricks/LTX-2.5", "generation_profile": row["profile"]!]
        })
        gallery = try JSONSerialization.data(withJSONObject: (0..<galleryPrints).map { index in
            ["filename": "fixture-\(galleryID.map { $0 + "-" } ?? "")\(index).\(mixedMedia ? ["png", "mp4", "glb"][index % 3] : "png")", "timestamp": 1_790_000_000 - index,
             "favorite": index < galleryFavorites,
             "collections": collectionFixture && index == 0 ? ["fixture-collection"] : [],
             "metadata": retainedMediaFixture ? ["prompt": "\(galleryID.map { "Photos-" + $0 } ?? "Fixture") \(index)", "model": "flux-dev:q4"] : ["prompt": "\(galleryID.map { "Photos-" + $0 } ?? "Fixture") \(index)"]] as [String: Any]
        })
        if justifiedFixture, var rows = try JSONSerialization.jsonObject(with: gallery) as? [[String: Any]] {
            for index in rows.indices {
                rows[index]["metadata"] = ["prompt": "Fixture \(index)", "width": [200, 600, 400][index % 3], "height": 400]
            }
            gallery = try JSONSerialization.data(withJSONObject: rows)
        }
        if trashFixture, var rows = try JSONSerialization.jsonObject(with: gallery) as? [[String: Any]] {
            for index in rows.indices { rows[index]["trashed_at"] = 1_790_000_100 }
            trashGallery = try JSONSerialization.data(withJSONObject: rows)
            gallery = Data("[]".utf8)
        }
        if retainedFrameFixture { gallery = try Self.frameReuseGallery(gallery) }
        if exportFixture, var rows = try JSONSerialization.jsonObject(with: gallery) as? [[String: Any]] {
            for index in rows.indices where index % 3 == 2 {
                let bytes = Self.exportImage(format: "png")
                rows[index]["assets"] = [["asset_id": "base-color", "role": "base_color", "display_name": "base-color.png", "media_type": "image/png", "size_bytes": bytes.count, "sha256": SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()]]
            }
            gallery = try JSONSerialization.data(withJSONObject: rows)
        }
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
        listener = try NWListener(using: parameters)
        if exportFixture { _ = try JSONSerialization.jsonObject(with: response("/api/capabilities")) }
    }

    func start() async throws -> UInt16 {
        try await withCheckedThrowingContinuation { continuation in
            listener.stateUpdateHandler = { [weak self] state in
                guard let self else { return }
                switch state {
                case .ready:
                    listener.stateUpdateHandler = nil
                    continuation.resume(returning: listener.port!.rawValue)
                case .failed(let error):
                    listener.stateUpdateHandler = nil
                    continuation.resume(throwing: error)
                default: break
                }
            }
            listener.newConnectionHandler = { [weak self] connection in
                guard let self else { connection.cancel(); return }
                connection.start(queue: queue)
                receive(connection, buffer: Data())
            }
            listener.start(queue: queue)
        }
    }

    func stop() {
        listener.cancel()
        eventConnections.withLock { connections in
            for connection in connections { connection.cancel() }
            connections.removeAll()
        }
    }

    var eventConnectionCount: Int { eventConnections.withLock { $0.count } }

    /// Change presentation-only owner fields while its source bytes stay fixed.
    func emitSourceDisplayOnlyUpdates(count: Int) async {
        await withCheckedContinuation { continuation in
            queue.async { [self] in
                var rows = (try? JSONSerialization.jsonObject(with: gallery)) as? [[String: Any]] ?? []
                guard !rows.isEmpty else { continuation.resume(); return }
                for index in 0..<count {
                    rows[0]["title"] = "Source owner update \(index)"
                    rows[0]["favorite"] = index.isMultiple(of: 2)
                    let payload: [String: Any] = ["type": "gallery_updated", "filename": rows[0]["filename"]!, "image": rows[0]]
                    let json = try! JSONSerialization.data(withJSONObject: payload)
                    var event = Data("event: event\ndata: ".utf8)
                    event.append(json); event.append(Data("\n\n".utf8))
                    var chunk = Data((String(event.count, radix: 16) + "\r\n").utf8)
                    chunk.append(event); chunk.append(Data("\r\n".utf8))
                    eventConnections.withLock { connections in
                        for connection in connections { connection.send(content: chunk, completion: .contentProcessed { _ in }) }
                    }
                }
                gallery = (try? JSONSerialization.data(withJSONObject: rows)) ?? gallery
                continuation.resume()
            }
        }
    }

    func completeQueueDownload() async {
        await withCheckedContinuation { continuation in
            queue.async { self.queueDownloadComplete = true; continuation.resume() }
        }
    }

    func addNewClip() async { await addNewMedia(filename: "fixture-new.mp4", title: "New clip", clip: true) }

    func addNewMedia(filename: String, title: String, clip: Bool = false) async {
        await withCheckedContinuation { continuation in
            queue.async { [self] in
                var rows = (try? JSONSerialization.jsonObject(with: gallery)) as? [[String: Any]] ?? []
                var metadata: [String: Any] = ["prompt": title]
                if clip { metadata["frames"] = 270; metadata["fps"] = 30 }
                rows.insert(["filename": filename, "timestamp": Int(Date().timeIntervalSince1970) + 1,
                             "metadata": metadata], at: 0)
                gallery = (try? JSONSerialization.data(withJSONObject: rows)) ?? gallery
                continuation.resume()
            }
        }
    }


    func restoreResidentModels() async {
        await withCheckedContinuation { continuation in
            queue.async { [self] in
                residentModels = ["flux-dev:q4", "ltx-2.5-22b-distilled:bf16"]
                continuation.resume()
            }
        }
    }

    private var refuseExport = false
    func refuseNextExport() { queue.sync { refuseExport = true } }

    private func receive(_ connection: NWConnection, buffer: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, done, error in
            guard let self else { connection.cancel(); return }
            var buffer = buffer
            if let data { buffer.append(data) }
            guard let text = String(data: buffer, encoding: .utf8), text.contains("\r\n\r\n") else {
                if done || error != nil { connection.cancel() } else { receive(connection, buffer: buffer) }
                return
            }
            let parts = text.components(separatedBy: "\r\n\r\n")
            let headers = parts[0].components(separatedBy: "\r\n")
            let length = headers.first { $0.lowercased().hasPrefix("content-length:") }
                .flatMap { Int($0.split(separator: ":", maxSplits: 1)[1].trimmingCharacters(in: .whitespaces)) } ?? 0
            let bodyText = parts.dropFirst().joined(separator: "\r\n\r\n")
            guard bodyText.utf8.count >= length else {
                if done || error != nil { connection.cancel() } else { receive(connection, buffer: buffer) }
                return
            }
            let request = headers[0].split(separator: " ")
            let path = request.count > 1 ? String(request[1]).components(separatedBy: "?")[0] : ""
            allRequests.withLock { $0.append(String(request.first ?? "") + " " + path) }
            if sourceMediaLibraryFixture, path == "/api/events" {
                // An idle real machine keeps SSE open. Returning the generic
                // JSON fixture here causes artificial reconnect/resync events,
                // invalidating sources during an otherwise unchanged UI test.
                eventConnections.withLock { $0.append(connection) }
                let comment = ": fixture-connected\n\n"
                let reply = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n"
                    + String(comment.utf8.count, radix: 16) + "\r\n" + comment + "\r\n"
                connection.send(content: Data(reply.utf8), completion: .contentProcessed { _ in })
                return
            }
            if sourceAspectFixture, path == "/api/gallery/source-media/fixture-0.png/fixture-portrait/thumbnail",
               sourceThumbnailFailures.withLock({ remaining in
                   guard remaining > 0 else { return false }
                   remaining -= 1; return true
               }) {
                let body = Data(#"{"error":"Synthetic transient thumbnail failure"}"#.utf8)
                var reply = Data("HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n".utf8)
                reply.append(body)
                connection.send(content: reply, completion: .contentProcessed { _ in connection.cancel() })
                return
            }
            if path == "/api/gallery", unavailableGallery.withLock({ $0 }) {
                let body = Data(#"{"error":"relay gateway unavailable; reconcile mutations before retrying"}"#.utf8)
                var reply = Data("HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n".utf8)
                reply.append(body)
                connection.send(content: reply, completion: .contentProcessed { _ in connection.cancel() })
                return
            }
            // AVKit probes MP4 byte ranges before decoding; serve a real
            // partial response rather than the export fixture's whole-file reply.
            if landscapePlaybackFixture, path.hasPrefix("/api/gallery/image/fixture-"), path.hasSuffix(".mp4"),
               request.first == "GET" || request.first == "HEAD" {
                let media = playbackData ?? Self.fixtureFile("landscape-playback.mp4")
                let range = headers.first { $0.lowercased().hasPrefix("range: bytes=") }
                    .flatMap { $0.split(separator: "=").last }.map { $0.split(separator: "-", omittingEmptySubsequences: false) }
                let start = range.flatMap { Int($0[0]) } ?? 0
                let end = min(range.flatMap { $0.count > 1 ? Int($0[1]) : nil } ?? media.count - 1, media.count - 1)
                guard start >= 0, start <= end else { connection.cancel(); return }
                let partial = range != nil
                let status = partial ? "206 Partial Content" : "200 OK"
                let contentRange = partial ? "Content-Range: bytes \(start)-\(end)/\(media.count)\r\n" : ""
                var reply = Data("HTTP/1.1 \(status)\r\nContent-Type: video/mp4\r\nAccept-Ranges: bytes\r\n\(contentRange)Content-Length: \(end - start + 1)\r\nConnection: close\r\n\r\n".utf8)
                if request.first != "HEAD" { reply.append(media[start...end]) }
                connection.send(content: reply, completion: .contentProcessed { _ in connection.cancel() })
                return
            }
            let patchCollection = collectionFixture && request.first == "PATCH"
                && path == "/api/gallery/collections/fixture-collection"
            if patchCollection,
               let object = try? JSONSerialization.jsonObject(with: Data(bodyText.utf8)) as? [String: Any],
               let hidden = object["hidden"] as? Bool { collectionHidden = hidden }
            if queueControls, request.first == "DELETE", path.hasPrefix("/api/queue/") {
                queueRequests.withLock { $0.append(String(request[1])) }
                let job = String(path.split(separator: "/").last ?? "")
                jobStates[job] = "cancelled"
            }
            if queueControls, path == "/api/history", request.first == "DELETE" { clearedHistory = true }
            if queueControls, request.first == "POST", path.hasPrefix("/api/queue/") {
                queueRequests.withLock { $0.append(path) }
                let parts = path.split(separator: "/")
                if parts.count == 4 {
                    let job = String(parts[2]); let action = parts[3]
                    if jobStates[job] != nil { jobStates[job] = action == "pause" ? "paused" : "queued" }
                }
            }
            let acceptLicense = queueDownloadFixture && request.first == "POST" && path == "/api/licenses/accept"
            if acceptLicense { downloadLicenseAccepted = true }
            let install = queueFixture && request.first == "POST" && path == "/api/downloads"
            let refuseLicense = install && requiresDownloadLicense && !downloadLicenseAccepted
            if install && !refuseLicense { queueDownloadStarted = true }
            if install, let object = try? JSONSerialization.jsonObject(with: Data(bodyText.utf8)) as? [String: Any],
               let model = object["model"] as? String { downloadedModels.withLock { $0.append(model) } }
            let unload = modelMemoryFixture && request.first == "DELETE" && path == "/api/models/unload"
            if unload {
                let object = (try? JSONSerialization.jsonObject(with: Data(bodyText.utf8))) as? [String: Any] ?? [:]
                if let name = object["model"] as? String { residentModels.remove(name) }
                else { residentModels.removeAll() }
            }
            let libraryMutation = libraryMutations && request.first == "POST"
                && ["/api/gallery/mutations", "/api/gallery/trash"].contains(path)
            if libraryMutation { capturedLibraryMutations.withLock { $0.append(Data(bodyText.utf8)) } }
            if libraryMutation,
               let object = try? JSONSerialization.jsonObject(with: Data(bodyText.utf8)) as? [String: Any],
               let filenames = object["filenames"] as? [String],
               var rows = try? JSONSerialization.jsonObject(with: gallery) as? [[String: Any]] {
                if path == "/api/gallery/trash" { rows.removeAll { filenames.contains($0["filename"] as? String ?? "") } }
                else if let favorite = object["favorite"] as? Bool {
                    for index in rows.indices where filenames.contains(rows[index]["filename"] as? String ?? "") {
                        rows[index]["favorite"] = favorite
                    }
                }
                if let collection = object["add_to_collection"] as? [String: String],
                   let name = collection["name"] {
                    let slug = name.lowercased().replacingOccurrences(of: " ", with: "-")
                    let id = slug == "uat-drafts" ? "fixture-collection" : slug
                    if id != "fixture-collection" { createdCollections[id] = name }
                    for index in rows.indices where filenames.contains(rows[index]["filename"] as? String ?? "") {
                        var members = rows[index]["collections"] as? [String] ?? []
                        if !members.contains(id) { members.append(id) }
                        rows[index]["collections"] = members
                    }
                }
                if let slug = object["remove_from_collection_slug"] as? String {
                    let id = slug == "uat-drafts" ? "fixture-collection" : slug
                    for index in rows.indices where filenames.contains(rows[index]["filename"] as? String ?? "") {
                        rows[index]["collections"] = (rows[index]["collections"] as? [String] ?? []).filter { $0 != id }
                    }
                }
                if path == "/api/gallery/mutations", let removePrintOnFavorite {
                    rows.removeAll { $0["filename"] as? String == removePrintOnFavorite }
                }
                gallery = (try? JSONSerialization.data(withJSONObject: rows)) ?? gallery
            }
            let captureGeneration = referenceFixture && request.first == "POST" && path == "/api/generation-batches"
            if captureGeneration { capturedGenerations.withLock { $0.append(Data(bodyText.utf8)) } }
            let export = exportFixture && request.first == "POST" && path.hasPrefix("/api/gallery/export/")
            if export { capturedExports.withLock { $0.append(Data(bodyText.utf8)) } }
            let exportRequest = (try? JSONSerialization.jsonObject(with: Data(bodyText.utf8))) as? [String: Any] ?? [:]
            let allowed = (promptExpansionFixture && path == "/api/expand") || acceptLicense || (queueDownloadFixture && path == "/api/generation-batches/status") || export || libraryMutation || (queueControls && (path.hasPrefix("/api/queue/") || path == "/api/history")) || install || unload || request.first == "GET" || path == "/api/generate/placement-preview" || patchCollection
            let historyQuery = request.count > 1 ? URLComponents(string: "http://fixture" + String(request[1]))?.queryItems?.first { $0.name == "query" }?.value : nil
            let isTrashListing = libraryMutations && path == "/api/gallery" && String(request[1]).contains("view=trash")
            let refusedExport = export && refuseExport
            if refusedExport { refuseExport = false }
            let licenseBody = Data(#"{"error":"Accept the model license","code":"LICENSE_NOT_ACCEPTED","license":{"id":"fixture-terms","name":"Fixture Model Terms","url":"https://example.com/terms","canonical":"https://example.com/terms","sha256":"abc","summary":"Fixture terms for a simulated download."}}"#.utf8)
            let body = refuseLicense ? licenseBody : acceptLicense ? Data("[]".utf8) : refusedExport ? Data(#"{"error":"Fixture refused this export"}"#.utf8) : export ? exportResponse(exportRequest) : libraryMutation ? Data("{}".utf8) : isTrashListing ? trashGallery : install ? Data(#"{"id":"fixture-download"}"#.utf8) : unload ? Data("{}".utf8) : patchCollection ? collection() : allowed ? response(path, historyQuery: historyQuery) : Data(#"{"error":"Fixture is read-only"}"#.utf8)
            let status = refuseLicense ? "403 Forbidden" : refusedExport ? "500 Internal Server Error" : allowed ? "200 OK" : "405 Method Not Allowed"
            let contentType = export ? "application/octet-stream" : path.hasSuffix(".mp4") ? "video/mp4" : path.hasPrefix("/api/gallery/image/") || path.hasPrefix("/api/gallery/thumbnail/") || path.hasSuffix("/input-thumbnail") || ((retainedMediaFixture || sourceAspectFixture) && path.hasPrefix("/api/gallery/source-media/fixture-0.png/") && (path.hasSuffix("/thumbnail") || path.hasSuffix("/content")))
                ? "image/png" : "application/json"
            var reply = Data("HTTP/1.1 \(status)\r\nContent-Type: \(contentType)\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n".utf8)
            reply.append(body)
            connection.send(content: reply, completion: .contentProcessed { _ in connection.cancel() })
        }
    }

    /// Color, visible edges and real aspect ratios make letterboxing observable.
    private static func sourceAspectImage(portrait: Bool) -> Data {
        let width = portrait ? 180 : 320, height = portrait ? 320 : 180
        let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.setFillColor(CGColor(red: 0.08, green: 0.35, blue: 0.75, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.setFillColor(CGColor(red: 1, green: 0.8, blue: 0.12, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: 12))
        context.fill(CGRect(x: 0, y: height - 12, width: width, height: 12))
        context.setFillColor(CGColor(red: 0.9, green: 0.15, blue: 0.2, alpha: 1))
        context.fill(CGRect(x: width / 4, y: height / 4, width: width / 2, height: height / 2))
        let data = NSMutableData()
        let destination = CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, context.makeImage()!, nil)
        precondition(CGImageDestinationFinalize(destination))
        return data as Data
    }

    private func queueRow(_ id: String) -> [String: Any] {
        let missing = queueDownloadFixture && id == "fixture-held"
        if queueFailureFixture, id == "fixture-held", !enrichedQueue.withLock({ $0 }) {
            return ["id": id, "state": jobStates[id] ?? "held", "model": "minimax-h3", "held_reason": "The render failed.", "error_detail": "CUDA_ERROR_ILLEGAL_ADDRESS in attention", "retryable": false]
        }
        var row: [String: Any] = ["id": id, "state": jobStates[id] ?? "queued", "model": missing ? "flux-dev:q4" : "ltx-2.5-22b-distilled:bf16",
         "model_display_name": "A legacy verbose title that must not appear",
         "position": id == "fixture-video" ? 0 : 1, "durable": true,
         "batch_id": batchQueueFixture ? "fixture-batch" : id, "client_batch_id": batchQueueFixture ? "client-fixture-batch" : "client-" + id, "retryable": true,
         "explicitly_paused": true,
         "held_reason": missing ? "deferred generation preparation failed: model 'flux-dev:q4' is not downloaded. Run: mold pull flux-dev:q4" : memoryErrorFixture ?? "Temporary machine pressure",
         "error": memoryErrorFixture ?? "Temporary machine pressure",
         "metadata": ["prompt": id == "fixture-video" ? "A coastal path at sunrise" : "A quiet mountain lake",
                      "model": "ltx-2.5-22b-distilled:bf16", "seed": 42, "steps": 8, "width": 768, "height": 512, "frames": 49, "fps": 24]]
        if queueFailureFixture, id == "fixture-held" {
            row["held_reason"] = "The render failed."
            row["error_detail"] = "CUDA_ERROR_ILLEGAL_ADDRESS in attention"
        }
        return row
    }

    private func collection() -> Data {
        Data(#"{"id":"fixture-collection","name":"UAT Drafts","slug":"uat-drafts","count":1,"hidden":\#(collectionHidden)}"#.utf8)
    }

    private func response(_ path: String, historyQuery: String? = nil) -> Data {
        if durationQueueFixture, let response = durationQueueResponse(path) { return response }
        if let retainedFrames = retainedFrameResponse(path) { return retainedFrames }
        if sourceAspectFixture, path == "/api/gallery/source-media/fixture-0.png" {
            let members: [[String: Any]] = [
                ["member_id": "fixture-portrait", "role": "source_image", "display_name": "Portrait source.png", "media_kind": "image", "size_bytes": 4_096],
                ["member_id": "fixture-landscape", "role": "last_frame", "display_name": "Landscape source.png", "media_kind": "image", "size_bytes": 8_192]
            ]
            return try! JSONSerialization.data(withJSONObject: ["availability": "available", "members": members])
        }
        if sourceAspectFixture, path.hasPrefix("/api/gallery/source-media/fixture-0.png/"),
           path.hasSuffix("/thumbnail") || path.hasSuffix("/content") {
            return Self.sourceAspectImage(portrait: path.contains("fixture-portrait"))
        }
        // A visible coastal illustration makes queue screenshots useful for
        // visual acceptance, rather than a white one-pixel placeholder.
        if sourceFreeQueueFixture && path.hasSuffix("/fixture-video/inputs") { return Data("[]".utf8) }
        if queueFixture && path.hasSuffix("/inputs") {
            return Data(#"[{"index":0,"label":"Reference image 1","preview":true},{"index":1,"label":"Reference image 2","preview":true},{"index":2,"label":"Reference 3 · audio","preview":false}]"#.utf8)
        }
        if path.hasSuffix("/input-thumbnail") {
            return Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAGAAAABACAIAAABqVuVZAAABE0lEQVR4nO3QPQ0CQRRF4fWDDxos0CIACxjAACpQQE+FA4yQUNBB2J2defPz7jvJqW9uvul8f9JMU/cHgwcQQAABBNDAAQQQQAABZNTrtv8NoL80JkwKQIs6JUYAqQMl6mQbAQQQQAABBBBADozyxgEKAJRilL0sAjTDVLgpBVQjgAACCCCABm56XE+r6v54dKBoTJlAcZiKgD5tjhfhDIC0mcyAVJmMgfSYqgApMVUE0mCqDuSdqRGQX6amQB6ZOgD5YuoG5IVp2h52X8G0AARTEhBMSUAwJQFFZloBFJNpNVA0pkygOExFQBGYDIC0mcyAVJmMgfSYqgApMVUE0mCqDuSdqRGQX6amQB6ZOgD5YnoDLYwrtRN2YTcAAAAASUVORK5CYII=")!
        }
        // A valid tiny PNG lets previews decode and source selection exercise
        // real image import; no generation or external machine is involved.
        if (path.hasPrefix("/api/gallery/image/fixture-") && path.hasSuffix(".png"))
            || path.hasPrefix("/api/gallery/thumbnail/fixture-") || path.hasSuffix("/input-thumbnail") || (retainedMediaFixture && path.hasPrefix("/api/gallery/source-media/fixture-0.png/fixture-source")) {
            if aspectFixture {
                let portrait = path.contains("fixture-1")
                let width = portrait ? 90 : 160, height = portrait ? 160 : 90
                let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                    bytesPerRow: 0, space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)!
                let data = NSMutableData()
                let destination = CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil)!
                CGImageDestinationAddImage(destination, context.makeImage()!, nil)
                precondition(CGImageDestinationFinalize(destination))
                return data as Data
            }
            return Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=")!
        }
        // One second of blue H.264 video, made with ffmpeg lavfi color at
        // 32x32, 2 fps, yuv420p and +faststart: exercise a real Photos write.
        if path.hasPrefix("/api/gallery/image/fixture-"), path.hasSuffix(".mp4") {
            if landscapePlaybackFixture { return playbackData ?? Self.fixtureFile("landscape-playback.mp4") }
            return Data(base64Encoded: "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAMxbW9vdgAAAGxtdmhkAAAAAAAAAAAAAAAAAAAD6AAAA+gAAQAAAQAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAlx0cmFrAAAAXHRraGQAAAADAAAAAAAAAAAAAAABAAAAAAAAA+gAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAACAAAAAgAAAAAAAkZWR0cwAAABxlbHN0AAAAAAAAAAEAAAPoAAAAAAABAAAAAAHUbWRpYQAAACBtZGhkAAAAAAAAAAAAAAAAAABAAAAAQABVxAAAAAAALWhkbHIAAAAAAAAAAHZpZGUAAAAAAAAAAAAAAABWaWRlb0hhbmRsZXIAAAABf21pbmYAAAAUdm1oZAAAAAEAAAAAAAAAAAAAACRkaW5mAAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAAAT9zdGJsAAAAv3N0c2QAAAAAAAAAAQAAAK9hdmMxAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAACAAIABIAAAASAAAAAAAAAABFExhdmM2My4xLjEwMSBsaWJ4MjY0AAAAAAAAAAAAAAAAGP//AAAANWF2Y0MBZAAK/+EAGGdkAAqs2UlsBEAAAAMAQAAAAwEDxIllgAEABmjr48siwP34+AAAAAAQcGFzcAAAAAEAAAABAAAAFGJ0cnQAAAAAAAAWeAAAAAAAAAAYc3R0cwAAAAAAAAABAAAAAgAAIAAAAAAUc3RzcwAAAAAAAAABAAAAAQAAABxzdHNjAAAAAAAAAAEAAAABAAAAAgAAAAEAAAAcc3RzegAAAAAAAAAAAAAAAgAAAsIAAAANAAAAFHN0Y28AAAAAAAAAAQAAA2EAAABhdWR0YQAAAFltZXRhAAAAAAAAACFoZGxyAAAAAAAAAABtZGlyYXBwbAAAAAAAAAAAAAAAACxpbHN0AAAAJKl0b28AAAAcZGF0YQAAAAEAAAAATGF2ZjYzLjEuMTAxAAAACGZyZWUAAALXbWRhdAAAAp8GBf//m9xF6b3m2Ui3lizYINkj7u94MjY0IC0gY29yZSAxNjUgLSBILjI2NC9NUEVHLTQgQVZDIGNvZGVjIC0gQ29weWxlZnQgMjAwMy0yMDI1IC0gaHR0cDovL3d3dy52aWRlb2xhbi5vcmcveDI2NC5odG1sIC0gb3B0aW9uczogY2FiYWM9MSByZWY9MyBkZWJsb2NrPTE6MDowIGFuYWx5c2U9MHgzOjB4MTEzIG1lPWhleCBzdWJtZT03IHBzeT0xIHBzeV9yZD0xLjAwOjAuMDAgbWl4ZWRfcmVmPTEgbWVfcmFuZ2U9MTYgY2hyb21hX21lPTEgdHJlbGxpcz0xIDh4OGRjdD0xIGNxbT0wIGRlYWR6b25lPTIxLDExIGZhc3RfcHNraXA9MSBjaHJvbWFfcXBfb2Zmc2V0PS0yIHRocmVhZHM9MSBsb29rYWhlYWRfdGhyZWFkcz0xIHNsaWNlZF90aHJlYWRzPTAgbnI9MCBkZWNpbWF0ZT0xIGludGVybGFjZWQ9MCBibHVyYXlfY29tcGF0PTAgY29uc3RyYWluZWRfaW50cmE9MCBiZnJhbWVzPTMgYl9weXJhbWlkPTIgYl9hZGFwdD0xIGJfYmlhcz0wIGRpcmVjdD0xIHdlaWdodGI9MSBvcGVuX2dvcD0wIHdlaWdodHA9MiBrZXlpbnQ9MjUwIGtleWludF9taW49MiBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNoPTAgcmNfbG9va2FoZWFkPTQwIHJjPWNyZiBtYnRyZWU9MSBjcmY9MjMuMCBxY29tcD0wLjYwIHFwbWluPTAgcXBtYXg9NjkgcXBzdGVwPTQgaXBfcmF0aW89MS40MCBhcT0xOjEuMDAAgAAAABtliIQAFP/+7Np+BTcMVvn10yG94AC3K4+Aln0AAAAJQZohbEEv/rXA")!
        }
        if exportFixture, path.hasPrefix("/api/gallery/image/"), path.hasSuffix(".glb") { return Self.fixtureFile("export-object.glb") }
        if exportFixture, path.hasPrefix("/api/gallery/assets/") { return Self.exportImage(format: "png") }
        if path.hasPrefix("/api/gallery/source-media/fixture-"), path.split(separator: "/").count == 4,
           path != "/api/gallery/source-media/fixture-0.png" {
            return Data(#"{"availability":"available","members":[]}"#.utf8)
        }
        let json: String
        switch path {
        case "/api/gallery/export-options":
            if unsupportedExportFormats { return Data(#"{"formats":["glb","future-animation"]}"#.utf8) }
            json = exportFixture ? #"{"formats":["gif","apng"],"gif_playback":["loop","bounce"],"gif_repeat":["forever","once"],"gif_pause":{"min":0,"max":5000,"step":10,"default":0}}"# : #"{"formats":["gif"]}"#
        case "/api/gallery/source-media/fixture-0.png":
            json = retainedMediaFixture
                ? #"{"availability":"available","members":[{"member_id":"fixture-source","role":"source_image","display_name":"Original.png","size_bytes":68}]}"#
                : #"{"availability":"unavailable_legacy"}"#
        case "/api/models":
            guard modelMemoryFixture else { return models }
            let rows = (try! JSONSerialization.jsonObject(with: models)) as! [[String: Any]]
            return try! JSONSerialization.data(withJSONObject: rows.map { row in
                var row = row
                row["is_loaded"] = residentModels.contains(row["name"] as! String)
                return row
            })
        case "/api/capabilities/chain-limits" where keyboardSequenceFixture:
            json = #"{"model":"ltx-2.5-22b-distilled:bf16","fps":24,"frames_per_clip_cap":257,"frames_per_clip_recommended":121,"max_stages":16,"max_total_frames":4096,"supports_sequence":true,"supports_audio":true,"authoring":{"motion_tail_frames":17,"carry_mode":"latent","min_frames":25,"frame_step":8,"frame_offset":1}}"#
        case "/api/status":
            if keyboardSequenceFixture { return Data(#"{"version":"0.36.0","busy":false,"uptime_secs":1,"instance_id":"00000000-0000-4000-8000-00000000c001"}"#.utf8) }
            if queueDestinationFixture { return Data(#"{"version":"0.32.0","busy":false,"uptime_secs":1,"instance_id":"queue-destination"}"#.utf8) }
            json = queueControls ? #"{"version":"0.32.0","busy":false,"uptime_secs":1,"instance_id":"queue-fixture"}"# : #"{"version":"0.32.0","busy":false,"uptime_secs":1}"#
        case "/api/expand":
            json = #"{"original":"Fixture prompt","expanded":["Expanded fixture prompt"]}"#
        case "/api/capabilities":
            if omitOrganizationCapabilities { return Data("{}".utf8) }
            if promptExpansionFixture { return Data(#"{"max_batch_outputs":4,"queue":{"heterogeneous_batch_max_outputs":4},"expand":{"configured":true}}"#.utf8) }
            if queueDestinationFixture { return Data(#"{"max_batch_outputs":4,"queue":{"heterogeneous_batch_max_outputs":4}}"#.utf8) }
            if exportFixture { return Data(#"{"max_batch_outputs":4,"mesh":{"generation":true,"formats":["glb"],"export_formats":["glb","obj","zip","stl","ply","gif","apng"],"export_geometry":{"size_mm":{"min":1,"max":10000,"default":100},"up_axes":["y","z"],"origins":["center","floor"],"defaults":{"obj":{"up_axis":"y","origin":"floor"},"stl":{"size_mm":100,"up_axis":"z","origin":"floor"},"ply":{"size_mm":100,"up_axis":"z","origin":"floor"}}}}}"#.utf8) }
            if queueControls { return Data(#"{"max_batch_outputs":4,"queue":{"can_pause_job":true,"cooperative_cancellation":true}}"#.utf8) }
            if libraryMutations { return Data(#"{"max_batch_outputs":4,"gallery":{"organize":true,"bulk_mutations":true,"trash":{"enabled":true}}}"#.utf8) }
            json = collectionFixture
            ? #"{"max_batch_outputs":4,"gallery":{"organize":true}}"#
            : #"{"max_batch_outputs":4}"#
        case "/api/queue":
            if queueControls {
                let ids = ["fixture-video", "fixture-held"] + (queueSwipeFixture ? ["fixture-running", "fixture-cancelling"] : [])
                return try! JSONSerialization.data(withJSONObject: ["entries": ids.map(queueRow)])
            }
            json = queueFixture
            ? #"{"entries":[{"id":"fixture-video","state":"queued","model":"ltx-2.5-22b-distilled:bf16","model_display_name":"LTX-2.5 Distilled BF16","position":0,"durable":true,"metadata":{"prompt":"A coastal path at sunrise","model":"ltx-2.5-22b-distilled:bf16"}}]}"#
            : #"{"entries":[]}"#
        case "/api/gallery": return gallery
        case "/api/gallery/collections":
            if collectionFixture {
                var data = Data("[".utf8); data.append(collection())
                for (id, name) in createdCollections.sorted(by: { $0.key < $1.key }) {
                    data.append(Data(",".utf8))
                    data.append(try! JSONSerialization.data(withJSONObject: ["id": id, "slug": id, "name": name]))
                }
                data.append(Data("]".utf8)); return data
            }
            json = "[]"
        case "/api/gallery/tags": json = "[]"
        case "/api/queue/fixture-video", "/api/queue/fixture-held":
            return try! JSONSerialization.data(withJSONObject: ["job": queueRow(path.hasSuffix("fixture-held") ? "fixture-held" : "fixture-video")])
        case "/api/generation-batches/status":
            if queueDownloadFixture {
                return try! JSONSerialization.data(withJSONObject: ["instance_id": "queue-fixture", "missing": ["batch_ids": [], "client_batch_ids": []], "batches": [
                    ["id": "fixture-held", "client_batch_id": "client-fixture-held", "instance_id": "queue-fixture", "children": [["index": 0, "job_id": "fixture-held", "state": jobStates["fixture-held"] == "held" ? "held" : "accepted", "error_code": "MODEL_NOT_FOUND", "retryable": true]]]
                ]])
            }
            json = "{}"
        case "/api/downloads":
            let job: [String: Any] = ["id": "fixture-download", "model": "flux-dev:q4", "status": queueDownloadComplete ? "completed" : "active", "bytes_done": queueDownloadComplete ? 100_000_000 : 35_000_000, "bytes_total": 100_000_000, "files_done": queueDownloadComplete ? 1 : 0, "files_total": 1]
            return try! JSONSerialization.data(withJSONObject: ["active_jobs": queueDownloadStarted && !queueDownloadComplete ? [job] : [], "queued": [], "history": queueDownloadComplete ? [job] : []])
        case "/api/history":
            if queueControls, !clearedHistory, historyQuery?.isEmpty != false || "A lighthouse in winter".localizedCaseInsensitiveContains(historyQuery ?? "") {
                return try! JSONSerialization.data(withJSONObject: ["entries": [["prompt": "A lighthouse in winter", "model": "flux-dev:q4", "used_at": 1791017129000]]])
            }
            json = #"{"entries":[]}"#
        default: json = "{}"
        }
        return Data(json.utf8)
    }
}
