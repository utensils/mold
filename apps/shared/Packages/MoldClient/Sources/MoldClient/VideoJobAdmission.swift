import Foundation

public struct VideoJobOwner: Hashable, Sendable {
    public let host: MoldHost
    public let instanceId: String
    public init(host: MoldHost, instanceId: String) { self.host = host; self.instanceId = instanceId }
}

/// Known parent operation, written before admission. No media or credentials.
public struct VideoJobAdmission: Codable, Hashable, Sendable, Identifiable {
    public let operationId: String
    public var id: String { operationId }
    public let hostId: UUID
    public let origin: String
    public let instanceId: String
    public let connectionMode: HostConnectionMode
    public var cancelRequested = false
    public var cancelAcknowledged = false
    public init(operationId: String, owner: VideoJobOwner) {
        self.operationId = operationId; hostId = owner.host.id
        origin = owner.host.baseURL.absoluteString; instanceId = owner.instanceId
        connectionMode = owner.host.connectionMode
    }
    public func matches(_ owner: VideoJobOwner) -> Bool {
        hostId == owner.host.id && origin == owner.host.baseURL.absoluteString
            && instanceId == owner.instanceId && connectionMode == owner.host.connectionMode
    }
}

public struct ObservedVideoJob: Identifiable, Hashable, Sendable {
    public let operationId: String
    public var id: String { owner.host.id.uuidString + ":" + operationId }
    public let owner: VideoJobOwner
    public var detail: VideoJob?
    public var message: String?
    public var cancelledBeforeAdmission = false
    public var outputAvailable: Bool?
    public var outputMessage: String?
}

public struct VideoJobPage: Hashable, Sendable {
    public let owner: VideoJobOwner
    public var hasMore: Bool
    public var nextCursor: String?
    public var isLoading: Bool
    public var error: String?
    /// Exact failed page request, including nil for the first page.
    public var retryCursor: String? = nil
}
