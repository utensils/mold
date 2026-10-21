import Foundation

extension HTTPBackend {
    public func videoRestartOptions(id: String, sourceInstanceId: String) async throws -> ChainRestartOptions {
        let route = "/api/video-jobs/\(escaped(id))/restart-options"
        let http = try restartRequest(route, method: "GET", instance: sourceInstanceId)
        return try decoded(ChainRestartOptions.self, from: try await bytes(for: http), route: route)
    }
    public func exportVideoRestart(id: String, request: ChainRestartExportRequest) async throws
        -> VideoParentRestartExport
    {
        let route = "/api/video-jobs/\(escaped(id))/restart-export"
        var http = try body(route, method: "POST", request)
        try videoDestination(request.sourceInstanceId, request: &http)
        return try decoded(VideoParentRestartExport.self, from: try await bytes(for: http), route: route)
    }
    public func videoRestartSource(id: String, offsetBytes: UInt64, sourceInstanceId: String, sourceRef: String)
        async throws -> Data
    {
        guard !sourceRef.isEmpty, !sourceInstanceId.isEmpty else { throw MoldClientError.malformedResponse }
        let route =
            "/api/video-jobs/\(escaped(id))/restart-source?offset_bytes=\(RouteEscaping.escapedQueryValue(String(offsetBytes)))"
        var http = try restartRequest(route, method: "GET", instance: sourceInstanceId)
        http.setValue(sourceInstanceId, forHTTPHeaderField: "x-mold-source-instance")
        http.setValue(sourceRef, forHTTPHeaderField: "x-mold-video-source-ref")
        let (stream, response) = try await self.response(http)
        do {
            guard response.statusCode == 200 else {
                let failure = await RefusalBody.read(stream)
                try HTTPRefusal.check(response, failure)
                throw MoldClientError.malformedResponse
            }
            let limit = 4 * 1024 * 1024
            if let length = response.value(forHTTPHeaderField: "Content-Length"),
                let value = UInt64(length), value > UInt64(limit)
            {
                throw MoldClientError.malformedResponse
            }
            var result = Data()
            result.reserveCapacity(limit)
            for try await chunk in stream.chunks {
                try Task.checkCancellation()
                guard chunk.count <= limit - result.count else { throw MoldClientError.malformedResponse }
                result.append(chunk)
            }
            return result
        } catch {
            await stream.cancel()
            throw error
        }
    }
    public func beginVideoSourceUpload(id: String, specJSON: String, destinationInstanceId: String) async throws
        -> VideoSourceUploadStatus
    {
        let route = "/api/video-source-uploads/\(escaped(id))"
        let http = try restartRequest(route, method: "POST", instance: destinationInstanceId, raw: specJSON)
        return try decoded(VideoSourceUploadStatus.self, from: try await bytes(for: http), route: route)
    }
    public func appendVideoSourceUpload(id: String, offsetBytes: UInt64, bytes: Data, destinationInstanceId: String)
        async throws -> VideoSourceUploadStatus
    {
        guard !bytes.isEmpty, bytes.count <= 4 * 1024 * 1024 else { throw MoldClientError.malformedResponse }
        let route =
            "/api/video-source-uploads/\(escaped(id))/chunks?offset_bytes=\(RouteEscaping.escapedQueryValue(String(offsetBytes)))"
        var http = try restartRequest(route, method: "PUT", instance: destinationInstanceId)
        http.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        http.httpBody = bytes
        return try decoded(VideoSourceUploadStatus.self, from: try await self.bytes(for: http), route: route)
    }
    public func completeVideoSourceUpload(id: String, destinationInstanceId: String) async throws
        -> VideoSourceUploadRef
    {
        let route = "/api/video-source-uploads/\(escaped(id))/complete"
        let http = try restartRequest(route, method: "POST", instance: destinationInstanceId)
        return try decoded(VideoSourceUploadRef.self, from: try await bytes(for: http), route: route)
    }
    public func validateVideoRestart(
        requestJSON: String, destinationInstanceId: String, sourceUpload: VideoSourceUploadRef?
    ) async throws -> VideoJobPlan {
        let route = "/api/video-jobs/validate"
        let http = try restartRequest(
            route, method: "POST", instance: destinationInstanceId, raw: requestJSON, source: sourceUpload)
        return try decoded(VideoJobPlan.self, from: try await bytes(for: http), route: route)
    }
    public func createVideoJob(
        requestJSON: String, validationRevision: String, operationId: String, destinationInstanceId: String,
        sourceUpload: VideoSourceUploadRef?
    ) async throws -> VideoJob {
        guard !validationRevision.isEmpty, !operationId.isEmpty else { throw MoldClientError.malformedResponse }
        let route = "/api/video-jobs"
        var http = try restartRequest(
            route, method: "POST", instance: destinationInstanceId, raw: requestJSON, source: sourceUpload)
        http.setValue(validationRevision, forHTTPHeaderField: "x-mold-video-validation-revision")
        http.setValue(operationId, forHTTPHeaderField: "x-mold-operation-id")
        return try decoded(VideoJob.self, from: try await bytes(for: http), route: route)
    }
    private func restartRequest(
        _ route: String, method: String, instance: String, raw: String? = nil, source: VideoSourceUploadRef? = nil
    ) throws -> URLRequest {
        var http = request(route, method: method)
        try videoDestination(instance, request: &http)
        if let raw {
            guard !raw.isEmpty, raw.utf8.count <= 64 * 1024 * 1024 else { throw MoldClientError.malformedResponse }
            http.setValue("application/json", forHTTPHeaderField: "Content-Type")
            http.httpBody = Data(raw.utf8)
        }
        if let source {
            let bytes = try MoldJSON.encoder.encode(source)
            guard bytes.count <= 1024, let value = String(data: bytes, encoding: .utf8) else {
                throw MoldClientError.malformedResponse
            }
            http.setValue(value, forHTTPHeaderField: "x-mold-video-source-upload")
        }
        return http
    }
}
