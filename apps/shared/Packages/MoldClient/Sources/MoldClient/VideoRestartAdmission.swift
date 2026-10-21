import Foundation

public enum VideoRestartProtocol: String, Codable, Hashable, Sendable { case chain, videoJob }

public struct VideoRestartTarget: Hashable, Sendable {
    public let host: MoldHost
    public let instanceId: String
    public init(host: MoldHost, instanceId: String) { self.host = host; self.instanceId = instanceId }
}

/// Scalar recovery identity only. Request JSON, source media and credentials
/// remain in memory and never enter this journal.
public struct VideoRestartAdmission: Codable, Hashable, Sendable, Identifiable {
    public var id: String { operationId }
    public let operationId: String
    public let destinationHostId: UUID
    public let destinationOrigin: String
    public let destinationInstanceId: String
    public let destinationMode: HostConnectionMode
    public let sourceHostId: UUID
    public let sourceJobId: String
    public let sourceInstanceId: String
    public let sourceRevision: String
    public var protocolKind: VideoRestartProtocol? = nil
    public var cancelRequested = false
    public var cancelAcknowledged = false

    public func matches(_ target: VideoRestartTarget) -> Bool {
        destinationHostId == target.host.id && destinationOrigin == target.host.baseURL.absoluteString
            && destinationInstanceId == target.instanceId && destinationMode == target.host.connectionMode
    }
}
