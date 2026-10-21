import Foundation

extension HTTPBackend {
    public func videoArtifactBytes(filename: String, role: VideoArtifactRole, offset: UInt64,
                                   sourceInstance: String, archive: String) async throws -> Data {
        let route = artifactPath(filename) + "/\(role.rawValue)?offset_bytes=\(offset)"
        let http = try artifactRequest(route, method: "GET", instance: sourceInstance, archive: archive, source: true)
        let (stream, response) = try await self.response(http)
        do {
            guard response.statusCode == 200 else {
                try HTTPRefusal.check(response, await RefusalBody.read(stream))
                throw MoldClientError.malformedResponse
            }
            let limit = 4 * 1024 * 1024
            if let length = response.value(forHTTPHeaderField: "Content-Length"), let count = UInt64(length), count > UInt64(limit) {
                throw MoldClientError.malformedResponse
            }
            var result = Data()
            for try await chunk in stream.chunks {
                try Task.checkCancellation()
                guard chunk.count <= limit - result.count else { throw MoldClientError.malformedResponse }
                result.append(chunk)
            }
            return result
        } catch { await stream.cancel(); throw error }
    }
    public func beginVideoArtifacts(filename: String, artifacts: VideoJobArtifacts,
                                    destinationInstance: String, archive: String) async throws -> VideoArtifactTransferStatus {
        struct Begin: Encodable { let archiveIdentitySha256: String; let artifacts: VideoJobArtifacts }
        try artifacts.validate()
        let route = artifactPath(filename) + "/transfer"
        var http = try artifactRequest(route, method: "POST", instance: destinationInstance, archive: archive)
        http.setValue("application/json", forHTTPHeaderField: "Content-Type")
        http.httpBody = try MoldJSON.encoder.encode(Begin(archiveIdentitySha256: archive, artifacts: artifacts))
        return try decoded(VideoArtifactTransferStatus.self, from: try await bytes(for: http), route: route)
    }
    public func videoArtifactsStatus(filename: String, handle: String, destinationInstance: String,
                                     archive: String) async throws -> VideoArtifactTransferStatus {
        let route = artifactPath(filename) + "/transfer/\(escaped(handle))"
        let http = try artifactRequest(route, method: "GET", instance: destinationInstance, archive: archive)
        return try decoded(VideoArtifactTransferStatus.self, from: try await bytes(for: http), route: route)
    }
    public func appendVideoArtifact(filename: String, handle: String, role: VideoArtifactRole, offset: UInt64, bytes: Data,
                                    destinationInstance: String, archive: String) async throws -> VideoArtifactTransferStatus {
        guard !bytes.isEmpty, bytes.count <= 4 * 1024 * 1024 else { throw MoldClientError.malformedResponse }
        let route = artifactPath(filename) + "/transfer/\(escaped(handle))/\(role.rawValue)?offset_bytes=\(offset)"
        var http = try artifactRequest(route, method: "PUT", instance: destinationInstance, archive: archive)
        http.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        http.httpBody = bytes
        return try decoded(VideoArtifactTransferStatus.self, from: try await self.bytes(for: http), route: route)
    }
    public func completeVideoArtifacts(filename: String, handle: String, destinationInstance: String,
                                       archive: String) async throws -> VideoArtifactTransferStatus {
        let route = artifactPath(filename) + "/transfer/\(escaped(handle))/complete"
        let http = try artifactRequest(route, method: "POST", instance: destinationInstance, archive: archive)
        return try decoded(VideoArtifactTransferStatus.self, from: try await bytes(for: http), route: route)
    }
    private func artifactPath(_ filename: String) -> String { "/api/gallery/video-artifacts/\(escaped(filename))" }
    private func artifactRequest(_ route: String, method: String, instance: String, archive: String,
                                 source: Bool = false) throws -> URLRequest {
        guard UUID(uuidString: instance) != nil, RetainedSourceMedia.validTransferDigest(archive) else {
            throw MoldClientError.malformedResponse
        }
        var http = request(route)
        http.httpMethod = method
        http.setValue(instance, forHTTPHeaderField: source ? "x-mold-source-instance" : "x-mold-destination-instance")
        http.setValue(archive, forHTTPHeaderField: "x-mold-gallery-archive-identity")
        return http
    }
}
