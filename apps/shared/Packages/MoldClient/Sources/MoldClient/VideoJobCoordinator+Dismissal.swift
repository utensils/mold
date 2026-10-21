import Foundation

extension VideoJobCoordinator {
    struct Removal: Codable, Hashable {
        let hostId: UUID
        let instanceId: String
        let operationId: String
        init(_ id: String, owner: VideoJobOwner) {
            hostId = owner.host.id; instanceId = owner.instanceId; operationId = id
        }
        init(_ record: VideoJobAdmission) {
            hostId = record.hostId; instanceId = record.instanceId; operationId = record.operationId
        }
    }
    static let removalsKey = "nativeDurationQueueRemovalsV1"
    static let removalBudget = 4 * 1024 * 1024

    func isRemoved(_ id: String, owner: VideoJobOwner) -> Bool {
        removals.contains(Removal(id, owner: owner))
    }
    func recordRemoval(_ id: String, owner: VideoJobOwner) throws {
        if let removalError { throw SequenceDraftError.invalid(removalError) }
        var next = removals; next.insert(Removal(id, owner: owner))
        let data = try MoldJSON.localEncoder.encode(next)
        guard data.count <= Self.removalBudget else {
            throw SequenceDraftError.invalid("Saved Queue dismissal history is full. Keep this device’s data before repairing storage.")
        }
        defaults.set(data, forKey: Self.removalsKey)
        guard defaults.synchronize() else { throw CocoaError(.fileWriteUnknown) }
        removals = next
    }
}
