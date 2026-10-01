import Foundation
import Network

/// A loopback-only machine for UI regression tests. It serves real generated
/// profiles and cannot generate or download. Optional collection mutations
/// update only the fixture’s own in-memory state.
final class FixtureMachine: @unchecked Sendable {
    private let listener: NWListener
    private let queue = DispatchQueue(label: "iphone-ui-fixture")
    private let models: Data
    private let gallery: Data
    private let collectionFixture: Bool
    private var collectionHidden = false

    init(galleryPrints: Int = 0, galleryFavorites: Int = 0, collectionFixture: Bool = false, mixedMedia: Bool = false) throws {
        self.collectionFixture = collectionFixture
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
                    "description": name, "downloaded": true, "generation_profile": row["profile"]!]
        })
        gallery = try JSONSerialization.data(withJSONObject: (0..<galleryPrints).map { index in
            ["filename": "fixture-\(index).\(mixedMedia ? ["png", "mp4", "glb"][index % 3] : "png")", "timestamp": 1_790_000_000 - index,
             "favorite": index < galleryFavorites,
             "collections": collectionFixture && index == 0 ? ["fixture-collection"] : [],
             "metadata": ["prompt": "Fixture \(index)"]] as [String: Any]
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
            let allowed = request.first == "GET" || path == "/api/generate/placement-preview" || patchCollection
            let body = patchCollection ? collection() : allowed ? response(path) : Data(#"{"error":"Fixture is read-only"}"#.utf8)
            let status = allowed ? "200 OK" : "405 Method Not Allowed"
            var reply = Data("HTTP/1.1 \(status)\r\nContent-Type: application/json\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n".utf8)
            reply.append(body)
            connection.send(content: reply, completion: .contentProcessed { _ in connection.cancel() })
        }
    }

    private func collection() -> Data {
        Data(#"{"id":"fixture-collection","name":"UAT Drafts","slug":"uat-drafts","count":1,"hidden":\#(collectionHidden)}"#.utf8)
    }

    private func response(_ path: String) -> Data {
        let json: String
        switch path {
        case "/api/models": return models
        case "/api/status": json = #"{"version":"0.32.0","busy":false,"uptime_secs":1}"#
        case "/api/capabilities": json = collectionFixture
            ? #"{"max_batch_outputs":4,"gallery":{"organize":true}}"#
            : #"{"max_batch_outputs":4}"#
        case "/api/queue": json = #"{"entries":[]}"#
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
