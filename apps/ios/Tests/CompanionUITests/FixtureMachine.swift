import Foundation
import Network
import Synchronization

/// A loopback-only machine for UI regression tests. It serves real generated
/// profiles and cannot generate or download. Optional collection and model-memory mutations
/// update only the fixture’s own in-memory state.
final class FixtureMachine: @unchecked Sendable {
    private let downloadedModels = Mutex<[String]>([])
    var installedRequests: [String] { downloadedModels.withLock { $0 } }
    private let listener: NWListener
    private let queue = DispatchQueue(label: "iphone-ui-fixture")
    private let models: Data
    private let gallery: Data
    private let retainedMediaFixture: Bool
    private let queueFixture: Bool
    private let collectionFixture: Bool
    private var collectionHidden = false
    private let modelMemoryFixture: Bool
    private var residentModels: Set<String> = []

    init(galleryPrints: Int = 0, galleryFavorites: Int = 0, collectionFixture: Bool = false, mixedMedia: Bool = false, queueFixture: Bool = false, retainedMediaFixture: Bool = false, loadedModels: Bool = false) throws {
        self.retainedMediaFixture = retainedMediaFixture
        self.queueFixture = queueFixture
        self.collectionFixture = collectionFixture
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
        let names = ["flux-dev:q4", "ltx-2.5-22b-distilled:bf16"]
        models = try JSONSerialization.data(withJSONObject: names.map { name -> [String: Any] in
            let row = profiles.first { ($0["models"] as! [[String: Any]]).contains { $0["model"] as? String == name } }!
            return ["name": name, "family": name.hasPrefix("flux") ? "flux" : "ltx",
                    "description": name, "downloaded": !(queueFixture && name.hasPrefix("flux")), "display_name": name.hasPrefix("flux") ? "FLUX.1 Dev Q4" : "LTX-2.5 Distilled BF16", "hf_repo": name.hasPrefix("flux") ? "black-forest-labs/FLUX.1-dev" : "Lightricks/LTX-2.5", "generation_profile": row["profile"]!]
        })
        gallery = try JSONSerialization.data(withJSONObject: (0..<galleryPrints).map { index in
            ["filename": "fixture-\(index).\(mixedMedia ? ["png", "mp4", "glb"][index % 3] : "png")", "timestamp": 1_790_000_000 - index,
             "favorite": index < galleryFavorites,
             "collections": collectionFixture && index == 0 ? ["fixture-collection"] : [],
             "metadata": retainedMediaFixture ? ["prompt": "Fixture \(index)", "model": "flux-dev:q4"] : ["prompt": "Fixture \(index)"]] as [String: Any]
        })
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
        listener = try NWListener(using: parameters)
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

    func stop() { listener.cancel() }

    func restoreResidentModels() async {
        await withCheckedContinuation { continuation in
            queue.async { [self] in
                residentModels = ["flux-dev:q4", "ltx-2.5-22b-distilled:bf16"]
                continuation.resume()
            }
        }
    }

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
            let patchCollection = collectionFixture && request.first == "PATCH"
                && path == "/api/gallery/collections/fixture-collection"
            if patchCollection,
               let object = try? JSONSerialization.jsonObject(with: Data(bodyText.utf8)) as? [String: Any],
               let hidden = object["hidden"] as? Bool { collectionHidden = hidden }
            let install = queueFixture && request.first == "POST" && path == "/api/downloads"
            if install, let object = try? JSONSerialization.jsonObject(with: Data(bodyText.utf8)) as? [String: Any],
               let model = object["model"] as? String { downloadedModels.withLock { $0.append(model) } }
            let unload = modelMemoryFixture && request.first == "DELETE" && path == "/api/models/unload"
            if unload {
                let object = (try? JSONSerialization.jsonObject(with: Data(bodyText.utf8))) as? [String: Any] ?? [:]
                if let name = object["model"] as? String { residentModels.remove(name) }
                else { residentModels.removeAll() }
            }
            let allowed = install || unload || request.first == "GET" || path == "/api/generate/placement-preview" || patchCollection
            let body = install ? Data(#"{"id":"fixture-download"}"#.utf8) : unload ? Data("{}".utf8) : patchCollection ? collection() : allowed ? response(path) : Data(#"{"error":"Fixture is read-only"}"#.utf8)
            let status = allowed ? "200 OK" : "405 Method Not Allowed"
            let contentType = path.hasPrefix("/api/gallery/image/") || path.hasPrefix("/api/gallery/thumbnail/") || path.hasSuffix("/input-thumbnail") || (retainedMediaFixture && path.hasSuffix("/fixture-source"))
                ? "image/png" : "application/json"
            var reply = Data("HTTP/1.1 \(status)\r\nContent-Type: \(contentType)\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n".utf8)
            reply.append(body)
            connection.send(content: reply, completion: .contentProcessed { _ in connection.cancel() })
        }
    }

    private func collection() -> Data {
        Data(#"{"id":"fixture-collection","name":"UAT Drafts","slug":"uat-drafts","count":1,"hidden":\#(collectionHidden)}"#.utf8)
    }

    private func response(_ path: String) -> Data {
        // A visible coastal illustration makes queue screenshots useful for
        // visual acceptance, rather than a white one-pixel placeholder.
        if path.hasSuffix("/input-thumbnail") {
            return Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAGAAAABACAIAAABqVuVZAAABE0lEQVR4nO3QPQ0CQRRF4fWDDxos0CIACxjAACpQQE+FA4yQUNBB2J2defPz7jvJqW9uvul8f9JMU/cHgwcQQAABBNDAAQQQQAABZNTrtv8NoL80JkwKQIs6JUYAqQMl6mQbAQQQQAABBBBADozyxgEKAJRilL0sAjTDVLgpBVQjgAACCCCABm56XE+r6v54dKBoTJlAcZiKgD5tjhfhDIC0mcyAVJmMgfSYqgApMVUE0mCqDuSdqRGQX6amQB6ZOgD5YuoG5IVp2h52X8G0AARTEhBMSUAwJQFFZloBFJNpNVA0pkygOExFQBGYDIC0mcyAVJmMgfSYqgApMVUE0mCqDuSdqRGQX6amQB6ZOgD5YnoDLYwrtRN2YTcAAAAASUVORK5CYII=")!
        }
        // A valid tiny PNG lets previews decode and source selection exercise
        // real image import; no generation or external machine is involved.
        if (path.hasPrefix("/api/gallery/image/fixture-") && path.hasSuffix(".png"))
            || path.hasPrefix("/api/gallery/thumbnail/fixture-") || path.hasSuffix("/input-thumbnail") || (retainedMediaFixture && path.hasSuffix("/fixture-source")) {
            return Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=")!
        }
        let json: String
        switch path {
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
        case "/api/status": json = #"{"version":"0.32.0","busy":false,"uptime_secs":1}"#
        case "/api/capabilities": json = collectionFixture
            ? #"{"max_batch_outputs":4,"gallery":{"organize":true}}"#
            : #"{"max_batch_outputs":4}"#
        case "/api/queue": json = queueFixture
            ? #"{"entries":[{"id":"fixture-video","state":"queued","model":"ltx-2.5-22b-distilled:bf16","model_display_name":"LTX-2.5 Distilled BF16","position":0,"durable":true,"metadata":{"prompt":"A coastal path at sunrise","model":"ltx-2.5-22b-distilled:bf16"}}]}"#
            : #"{"entries":[]}"#
        case "/api/gallery": return gallery
        case "/api/gallery/collections":
            if collectionFixture {
                var data = Data("[".utf8); data.append(collection()); data.append(Data("]".utf8)); return data
            }
            json = "[]"
        case "/api/gallery/tags": json = "[]"
        case "/api/history": json = #"{"entries":[]}"#
        default: json = "{}"
        }
        return Data(json.utf8)
    }
}
