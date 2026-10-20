import Foundation

public struct ChainRestartPermission: Codable, Hashable, Sendable {
    public let supported: Bool
    public let reason: String?
}

public struct ChainRestartOptions: Codable, Hashable, Sendable {
    public let sourceInstanceId: String
    public let sourceRevision: String
    public let restart: ChainRestartPermission
    public let send: ChainRestartPermission
    public let discard: ChainRestartPermission?

    public func permission(destinationInstanceId: String) -> ChainRestartPermission {
        destinationInstanceId == sourceInstanceId ? restart : send
    }
}

public struct ChainRestartExportRequest: Codable, Hashable, Sendable {
    public let sourceInstanceId: String
    public let sourceRevision: String
    public let destinationInstanceId: String

    public init(sourceInstanceId: String, sourceRevision: String, destinationInstanceId: String) {
        self.sourceInstanceId = sourceInstanceId
        self.sourceRevision = sourceRevision
        self.destinationInstanceId = destinationInstanceId
    }
}

/// Private wire-ready request. Do not decode/re-encode, normalize prompts again,
/// log, or persist it: it contains source media and full-range numeric seeds.
public struct ChainRestartExport: Decodable, Hashable, Sendable {
    public let sourceInstanceId: String
    public let sourceRevision: String
    public let destinationInstanceId: String
    public let requestJSON: String

    private enum CodingKeys: String, CodingKey {
        case sourceInstanceId, sourceRevision, destinationInstanceId
        case requestJSON = "requestJson"
    }

    public func validated(for request: ChainRestartExportRequest) throws -> String {
        guard sourceInstanceId == request.sourceInstanceId, sourceRevision == request.sourceRevision,
              destinationInstanceId == request.destinationInstanceId,
              !requestJSON.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw SequenceDraftError.invalid("This video or its destination changed. Reopen the job before restarting it.")
        }
        return requestJSON
    }
}


public struct ChainDiscardRequest: Codable, Hashable, Sendable {
    public let sourceInstanceId: String
    public let sourceRevision: String
    public init(sourceInstanceId: String, sourceRevision: String) {
        self.sourceInstanceId = sourceInstanceId; self.sourceRevision = sourceRevision
    }
}
