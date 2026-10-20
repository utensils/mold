import Foundation
import Observation

/// One explicit new attempt, sharing a scalar recovery ledger across both apps.
/// Closing a presentation does not cancel admitted work or replay an unknown POST.
@MainActor @Observable
public final class VideoRestartController {
    public nonisolated static let journalKey = "nativeVideoRestartsV1"
    public private(set) var pending: [VideoRestartAdmission] = []
    public internal(set) var admission: VideoRestartAdmission?
    public internal(set) var detail: ChainJobDetail?
    public internal(set) var options: ChainRestartOptions?
    public internal(set) var error: String?
    public internal(set) var notice: String?
    public internal(set) var isLoading = false
    public internal(set) var isBusy = false
    public internal(set) var isCancelling = false
    public internal(set) var cancelledBeforeAdmission = false
    public internal(set) var source: VideoRestartTarget?
    public internal(set) var sourceJobId: String?
    @ObservationIgnored var sourceBackend: (any MoldBackend)?
    @ObservationIgnored var sourceCurrent: @MainActor () -> Bool = { false }
    @ObservationIgnored var revision = 0
    @ObservationIgnored let defaults: UserDefaults
    @ObservationIgnored var journalError: String?

    public init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        do {
            if let data = defaults.data(forKey: Self.journalKey) {
                pending = try MoldJSON.decoder.decode([VideoRestartAdmission].self, from: data)
                guard pending.count <= 128 else { throw CocoaError(.fileReadCorruptFile) }
            }
        } catch {
            journalError = "This device’s video recovery record is unreadable. Keep it before repairing storage."
            self.error = journalError
        }
    }

    public func open(source: VideoRestartTarget, jobId: String, backend: any MoldBackend,
                     current: @escaping @MainActor () -> Bool) async {
        revision += 1
        let mine = revision
        self.source = source; sourceJobId = jobId; sourceBackend = backend; sourceCurrent = current
        isLoading = false
        options = nil; detail = nil; error = journalError; notice = nil; admission = nil
        cancelledBeforeAdmission = false
        guard journalError == nil else { return }
        if let saved = pending.last(where: { $0.sourceHostId == source.host.id && $0.sourceJobId == jobId
            && $0.sourceInstanceId == source.instanceId }) {
            admission = saved
            return
        }
        isLoading = true
        defer { if mine == revision { isLoading = false } }
        do {
            try await verify(source, backend: backend, current: current)
            guard mine == revision, current() else { return }
            let loaded = try await backend.chainRestartOptions(id: jobId)
            guard mine == revision, current() else { return }
            guard loaded.sourceInstanceId == source.instanceId else { throw changed() }
            options = loaded
        } catch {
            if mine == revision { self.error = UserFacingError.describe(error) }
        }
    }

    public func close() { revision += 1; isLoading = false; options = nil }

    func verify(_ target: VideoRestartTarget, backend: any MoldBackend,
                current: @MainActor () -> Bool) async throws {
        guard current(), backend.host == target.host, !target.instanceId.isEmpty else { throw changed() }
        let status = try await backend.status()
        guard current(), status.instanceId == target.instanceId else { throw changed() }
        let capabilities = try await backend.capabilities()
        guard current() else { throw changed() }
        guard capabilities.canRestartChain else {
            throw SequenceDraftError.invalid("Update this machine before restarting a saved video job.")
        }
    }

    func changed() -> SequenceDraftError {
        .invalid("The source or destination machine changed. Reopen this video job.")
    }

    func save(_ record: VideoRestartAdmission) throws {
        guard journalError == nil else { throw CocoaError(.fileReadCorruptFile) }
        var next = pending.filter { $0.operationId != record.operationId }
        guard next.count < 128 else { throw SequenceDraftError.invalid("Resolve earlier video attempts before starting another.") }
        next.append(record)
        defaults.set(try MoldJSON.encoder.encode(next), forKey: Self.journalKey)
        guard defaults.synchronize() else { throw CocoaError(.fileWriteUnknown) }
        pending = next
    }

    func forget(_ id: String) throws {
        let next = pending.filter { $0.operationId != id }
        defaults.set(try MoldJSON.encoder.encode(next), forKey: Self.journalKey)
        guard defaults.synchronize() else { throw CocoaError(.fileWriteUnknown) }
        pending = next
    }

    static func definitive(_ error: Error) -> Bool {
        guard let error = error as? MoldClientError else { return false }
        if case .unauthorized = error { return true }
        guard case let .http(status, code, _) = error else { return false }
        return [400, 401, 403, 404, 413, 415, 422, 451].contains(status)
            || (status == 409 && code == "CHAIN_DESTINATION_CHANGED")
    }
}
