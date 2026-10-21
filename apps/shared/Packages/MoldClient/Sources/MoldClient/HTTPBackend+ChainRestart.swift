import Foundation

public extension HTTPBackend {
    func cancelChainOperation(id: String, operationId: String, destinationInstanceId: String) async throws {
        guard !destinationInstanceId.isEmpty else { throw SequenceDraftError.invalid("Reconnect to verify the destination machine.") }
        var http = request("/api/chain-jobs/\(escaped(id))/operations/\(escaped(operationId))/cancel")
        http.httpMethod = "POST"
        http.setValue(destinationInstanceId, forHTTPHeaderField: "x-mold-destination-instance")
        _ = try await bytes(for: http)
    }

    func chainRestartOptions(id: String) async throws -> ChainRestartOptions {
        try await get("/api/chain-jobs/\(escaped(id))/restart-options")
    }

    func exportChainRestart(id: String, request: ChainRestartExportRequest) async throws -> ChainRestartExport {
        try await post("/api/chain-jobs/\(escaped(id))/restart-export", body: request)
    }

    func createChainJob(requestJSON: String, operationId: String, destinationInstanceId: String) async throws -> CreateChainJobResponse {
        var http = request("/api/chain-jobs", method: "POST")
        http.setValue("application/json", forHTTPHeaderField: "Content-Type")
        http.setValue(operationId, forHTTPHeaderField: "x-mold-operation-id")
        http.setValue(destinationInstanceId, forHTTPHeaderField: "x-mold-destination-instance")
        // The server already protected literal backslashes. Re-encoding would
        // also risk full-range u64 seed authority and duplicate large media.
        http.httpBody = Data(requestJSON.utf8)
        return try decoded(CreateChainJobResponse.self, from: try await bytes(for: http), route: "/api/chain-jobs")
    }

    func discardChainJob(id: String, request: ChainDiscardRequest) async throws {
        try await send("/api/chain-jobs/\(escaped(id))/discard", method: "POST", body: request)
    }
}
