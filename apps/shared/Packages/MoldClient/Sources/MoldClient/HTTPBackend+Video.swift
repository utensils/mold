import Foundation

public extension HTTPBackend {
    func planVideoJob(_ request: VideoJobRequest) async throws -> VideoJobPlan {
        let plan: VideoJobPlan = try await post("/api/video-jobs/validate?planning_only=true", body: request.planningProjection)
        guard plan.validationRevision.isEmpty else { throw MoldClientError.malformedResponse }
        return plan
    }

    func validateVideoJob(_ request: VideoJobRequest) async throws -> VideoJobPlan {
        try await post("/api/video-jobs/validate", body: request)
    }

    func createVideoJob(_ request: VideoJobRequest, operationId: String, destinationInstanceId: String) async throws -> VideoJob {
        guard request.validationRevision?.isEmpty == false else {
            throw SequenceDraftError.invalid("Check this video's length before generating it.")
        }
        let route = "/api/video-jobs"
        var http = try body(route, method: "POST", request)
        try videoDestination(destinationInstanceId, request: &http)
        guard !operationId.isEmpty else { throw SequenceDraftError.invalid("A video operation identifier is required.") }
        http.setValue(operationId, forHTTPHeaderField: "x-mold-operation-id")
        return try decoded(VideoJob.self, from: try await bytes(for: http), route: route)
    }

    func videoJob(id: String) async throws -> VideoJob { try await get("/api/video-jobs/\(escaped(id))") }

    func listVideoJobs(limit: Int, offset: Int) async throws -> VideoJobListing {
        guard (1...256).contains(limit), offset >= 0 else {
            throw SequenceDraftError.invalid("A video list requires a limit from 1 to 256 and a nonnegative offset.")
        }
        return try await get("/api/video-jobs?limit=\(RouteEscaping.escapedQueryValue(String(limit)))&offset=\(RouteEscaping.escapedQueryValue(String(offset)))")
    }

    /// Also fences a known operation whose admission reply was lost. A 410 on
    /// subsequent GET confirms it was cancelled before a parent existed.
    func cancelVideoJob(id: String, destinationInstanceId: String) async throws {
        _ = try await videoMutation(id: id, action: "/cancel", method: "POST", instance: destinationInstanceId)
    }

    func resumeVideoJob(id: String, destinationInstanceId: String) async throws -> VideoJob {
        let route = "/api/video-jobs/\(escaped(id))/resume"
        return try decoded(VideoJob.self,
            from: try await videoMutation(id: id, action: "/resume", method: "POST", instance: destinationInstanceId), route: route)
    }

    func deleteVideoJob(id: String, destinationInstanceId: String) async throws {
        _ = try await videoMutation(id: id, action: "", method: "DELETE", instance: destinationInstanceId)
    }

    func videoJobEvents(id: String) -> AsyncThrowingStream<VideoJob, Error> {
        AsyncThrowingStream(bufferingPolicy: .bufferingNewest(StreamBuffering.latestOnly)) { continuation in
            let task = Task {
                do {
                    for try await frame in stream("/api/video-jobs/\(escaped(id))/events", timeout: 3_600) {
                        guard frame.name == "snapshot", let data = frame.data.data(using: .utf8) else { continue }
                        let value = try MoldJSON.decoder.decode(VideoJob.self, from: data)
                        continuation.yield(value)
                        if value.isSettled { break }
                    }
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func videoMutation(id: String, action: String, method: String, instance: String) async throws -> Data {
        var http = request("/api/video-jobs/\(escaped(id))" + action)
        http.httpMethod = method
        try videoDestination(instance, request: &http)
        return try await bytes(for: http)
    }

    private func videoDestination(_ instance: String, request: inout URLRequest) throws {
        guard !instance.isEmpty else { throw SequenceDraftError.invalid("Reconnect to verify this machine before changing a video job.") }
        request.setValue(instance, forHTTPHeaderField: "x-mold-destination-instance")
    }
}
