import Foundation

/// A discovery trigger, separate from progress. Credentials and installation
/// belong to the owner; only exact parent ID membership changes relist jobs.
public struct VideoJobDiscoveryMembership: Hashable, Sendable {
    public let owner: VideoJobOwner
    public let activeIDs: [String]
    public init(owner: VideoJobOwner, activity: ActivityHostSnapshot?) {
        self.owner = owner
        guard let activity, activity.instanceId == owner.instanceId,
              activity.routeURL == owner.host.baseURL.absoluteString else {
            activeIDs = []; return
        }
        activeIDs = Set(activity.items.filter { $0.kind == "video_job" }.map(\.id)).sorted()
    }
}
