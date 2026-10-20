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
            record.matches(owner) && !jobs.contains { $0.operationId == record.operationId && $0.owner == owner && $0.detail != nil }
        }
    }
    func releaseTransport(_ id: String, owner: VideoJobOwner) async {
        let key = identity(id, owner: owner)
        guard let cleanup = transportCleanups[key], cleanup.admission.matches(owner) else { return }
        transportCleanups[key] = nil
        await cleanup.release()
    }
    public func isMutating(_ job: ObservedVideoJob) -> Bool { mutating.contains(job.id) }
    public func isCancellationRequested(_ job: ObservedVideoJob) -> Bool {
        pending.contains { $0.operationId == job.operationId && $0.matches(job.owner) && $0.cancelRequested }
    }
    func index(_ id: String, owner: VideoJobOwner) -> Int? {
        jobs.firstIndex { $0.operationId == id && $0.owner == owner }
    }
    func ensure(_ id: String, owner: VideoJobOwner) {
        if !jobs.contains(where: { $0.operationId == id && $0.owner.host.id == owner.host.id }) {
            jobs.append(.init(operationId: id, owner: owner))
        }
    }
    func message(_ text: String, id: String, owner: VideoJobOwner) {
        ensure(id, owner: owner)
        if let index = index(id, owner: owner) { jobs[index].message = text }
    }
    func publish(_ value: VideoJob, owner: VideoJobOwner) {
        if let stale = jobs.firstIndex(where: { $0.operationId == value.id && $0.owner.host.id == owner.host.id && $0.owner != owner }) {
            jobs[stale] = .init(operationId: value.id, owner: owner)
        }
        ensure(value.id, owner: owner)
        if let index = index(value.id, owner: owner) { jobs[index].detail = value; jobs[index].message = value.state == .completed && value.outputFilename == nil
            ? "Completed, but no saved Library video was reported." : nil }
        let completed = jobs.filter { $0.detail?.state == .completed || $0.detail?.state == .cancelled || $0.cancelledBeforeAdmission }
        if completed.count > 32 {
            let old = Set(completed.prefix(completed.count - 32).map(\.id))
            old.forEach { retireReads($0) }
            jobs.removeAll { old.contains($0.id) }
        }
    }
    func retireReads(_ key: String) {
        readTokens[key] = nil
        followers[key]?.cancel(); followers[key] = nil; followTokens[key] = nil
    }
    public func cancelPreparation() { submissionToken = UUID(); isSubmitting = false }
    public func suspend() {
        followers.values.forEach { $0.cancel() }; followers.removeAll(); followTokens.removeAll(); readTokens.removeAll(); pageTokens.removeAll(); pagination.removeAll()
    }
}
