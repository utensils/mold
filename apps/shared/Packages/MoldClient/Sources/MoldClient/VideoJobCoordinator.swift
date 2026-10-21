import Foundation
import Observation

/// Native duration parents share durable admission and recovery semantics.
/// Adapters capture owner and credential revisions separately from draft edits.
@MainActor @Observable
public final class VideoJobCoordinator {
    public nonisolated static let journalKey = "nativeDurationVideoJobsV1"
    public internal(set) var unavailableRecoveryCount = 0
    public internal(set) var pending: [VideoJobAdmission] = []
    public internal(set) var jobs: [ObservedVideoJob] = []
    public internal(set) var pagination: [UUID: VideoJobPage] = [:]
    @ObservationIgnored var pageTokens: [UUID: UUID] = [:]
    public internal(set) var isSubmitting = false
    public internal(set) var error: String?
    @ObservationIgnored let defaults: UserDefaults
    @ObservationIgnored var journalError: String?
    @ObservationIgnored var submissionToken = UUID()
    @ObservationIgnored var followers: [String: Task<Void, Never>] = [:]
    var mutating: Set<String> = []
    @ObservationIgnored var readTokens: [String: UUID] = [:]
    struct TransportCleanup {
        let admission: VideoJobAdmission
        let release: @Sendable () async -> Void
    }
    @ObservationIgnored var transportCleanups: [String: TransportCleanup] = [:]
    @ObservationIgnored var followTokens: [String: UUID] = [:]
    @ObservationIgnored var followOwners: [String: VideoJobOwner] = [:]
    @ObservationIgnored var removals: Set<Removal> = []
    @ObservationIgnored var removalError: String?
    var deleting: Set<String> = []
    var outputReads: [String: VideoJobOutputRead] = [:]

    public init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        do {
            if let data = defaults.data(forKey: Self.journalKey) {
                pending = try MoldJSON.localDecoder.decode([VideoJobAdmission].self, from: data)
                guard pending.count <= 128 else { throw CocoaError(.fileReadCorruptFile) }
            }
        } catch {
            journalError = "This device’s video recovery record is unreadable. Keep it before repairing storage."
            self.error = journalError
        }
        do {
            if let data = defaults.data(forKey: Self.removalsKey) {
                guard data.count <= Self.removalBudget else { throw CocoaError(.fileReadCorruptFile) }
                removals = try MoldJSON.localDecoder.decode(Set<Removal>.self, from: data)
            }
        } catch {
            removalError = "Saved Queue dismissals are unreadable. Keep this device’s data before repairing storage."
            self.error = removalError
        }
    }

    func save(_ record: VideoJobAdmission) throws {
        guard journalError == nil else { throw CocoaError(.fileReadCorruptFile) }
        var next = pending.filter { $0.operationId != record.operationId || $0.hostId != record.hostId }
        guard next.count < 128 else { throw SequenceDraftError.invalid("Resolve earlier video attempts before starting another.") }
        next.append(record)
        defaults.set(try MoldJSON.localEncoder.encode(next), forKey: Self.journalKey)
        guard defaults.synchronize() else { throw CocoaError(.fileWriteUnknown) }
        pending = next
    }

    func forget(_ id: String, owner: VideoJobOwner) throws {
        let next = pending.filter { $0.operationId != id || !$0.matches(owner) }
        guard next != pending else { return }
        defaults.set(try MoldJSON.localEncoder.encode(next), forKey: Self.journalKey)
        guard defaults.synchronize() else { throw CocoaError(.fileWriteUnknown) }
        pending = next
    }

    func scoped(_ backend: any MoldBackend, owner: VideoJobOwner) -> any MoldBackend {
        if let http = backend as? HTTPBackend { return http.pinningVideoDestination(owner.instanceId) }
        return backend
    }

    func verify(_ owner: VideoJobOwner, backend: any MoldBackend, current: @MainActor () -> Bool) async throws {
        guard !Task.isCancelled, current(), backend.host == owner.host, !owner.instanceId.isEmpty else { throw changed() }
        let status = try await backend.status()
        guard !Task.isCancelled, current(), status.instanceId == owner.instanceId else { throw changed() }
        let capabilities = try await backend.capabilities()
        guard !Task.isCancelled, current() else { throw changed() }
        guard capabilities.canCreateVideoJob else { throw SequenceDraftError.invalid("Update this machine before creating duration-based videos.") }
    }

    func changed() -> SequenceDraftError { .invalid("The video, machine, or connection changed. Check it and try again.") }
    func identity(_ id: String, owner: VideoJobOwner) -> String { owner.host.id.uuidString + ":" + id }
    public func hasUncertainAdmission(on owner: VideoJobOwner) -> Bool {
        pending.contains { record in
            record.matches(owner) && !isRemoved(record.operationId, owner: owner) && !jobs.contains { $0.operationId == record.operationId && $0.owner == owner && $0.detail != nil }
        }
    }
    func releaseTransport(_ id: String, owner: VideoJobOwner) async {
        let key = identity(id, owner: owner)
        guard let cleanup = transportCleanups[key], cleanup.admission.matches(owner) else { return }
        transportCleanups[key] = nil
        await cleanup.release()
    }
    /// Both native shells must retain an otherwise-empty Queue for a visible
    /// per-machine retry or another page, as well as admission/recovery errors.
    public var hasQueueContent: Bool {
        !jobs.isEmpty || !pending.isEmpty || error != nil
            || pagination.values.contains { $0.error != nil || $0.hasMore || $0.isLoading }
    }

    public func isDeleting(_ job: ObservedVideoJob) -> Bool { deleting.contains(job.id) }
    public func isMutating(_ job: ObservedVideoJob) -> Bool { mutating.contains(job.id) }
    public func isCancellationRequested(_ job: ObservedVideoJob) -> Bool {
        pending.contains { $0.operationId == job.operationId && $0.matches(job.owner) && $0.cancelRequested }
    }
    func index(_ id: String, owner: VideoJobOwner) -> Int? {
        jobs.firstIndex { $0.operationId == id && $0.owner == owner }
    }
    func ensure(_ id: String, owner: VideoJobOwner) {
        guard !isRemoved(id, owner: owner) else { return }
        if !jobs.contains(where: { $0.operationId == id && $0.owner.host.id == owner.host.id }) {
            jobs.append(.init(operationId: id, owner: owner))
        }
    }
    func message(_ text: String, id: String, owner: VideoJobOwner) {
        ensure(id, owner: owner)
        if let index = index(id, owner: owner) { jobs[index].message = text }
    }
    func publish(_ value: VideoJob, owner: VideoJobOwner) {
        guard !isRemoved(value.id, owner: owner) else { return }
        if let stale = jobs.firstIndex(where: { $0.operationId == value.id && $0.owner.host.id == owner.host.id && $0.owner != owner }) {
            jobs[stale] = .init(operationId: value.id, owner: owner)
        }
        ensure(value.id, owner: owner)
        if let index = index(value.id, owner: owner), jobs[index].detail?.outputFilename != value.outputFilename {
            outputReads[jobs[index].id]?.cancel(); outputReads[jobs[index].id] = nil
            jobs[index].outputAvailable = nil; jobs[index].outputMessage = nil
        }
        if let index = index(value.id, owner: owner) { jobs[index].detail = value; jobs[index].message = value.state == .completed && value.outputFilename == nil
            ? "Completed, but no saved Library video was reported." : nil }
        let completed = jobs.filter { $0.detail?.state == .completed || $0.cancelledBeforeAdmission }
        if completed.count > 32 {
            let old = Set(completed.prefix(completed.count - 32).map(\.id))
            old.forEach { retireReads($0) }
            jobs.removeAll { old.contains($0.id) }
        }
    }
    func retireReads(_ key: String) {
        readTokens[key] = nil; outputReads[key]?.cancel(); outputReads[key] = nil
        followers[key]?.cancel(); followers[key] = nil; followTokens[key] = nil; followOwners[key] = nil
    }
    public func cancelPreparation() { submissionToken = UUID(); isSubmitting = false }
    public func suspend() {
        followers.values.forEach { $0.cancel() }; followers.removeAll(); followTokens.removeAll(); followOwners.removeAll(); readTokens.removeAll(); outputReads.values.forEach { $0.cancel() }; outputReads.removeAll(); pageTokens.removeAll(); pagination.removeAll()
    }
}
