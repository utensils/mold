import Foundation
import Observation

/// Settlement belongs to the exact tickets returned by this acquisition, including companions.
public enum QueueDownloadSettlement: Equatable, Sendable {
    case waiting, ready, failed(String)

    public static func progress(jobs: [DownloadJob]) -> DownloadProgress {
        let total = jobs.reduce(Int64(0)) { $0 + $1.bytesTotal }
        let done = jobs.reduce(Int64(0)) { $0 + $1.bytesDone }
        let known = !jobs.isEmpty && jobs.allSatisfy { $0.bytesTotal > 0 }
        return DownloadProgress(model: "", fraction: known ? min(1, max(0, Double(done) / Double(total))) : nil,
                                bytesDone: done, bytesTotal: known ? total : nil)
    }

    public static func resolve(ids: [String], jobs: [DownloadJob]) -> Self {
        guard !ids.isEmpty else { return .waiting }
        let matching = ids.compactMap { id in jobs.first { $0.id == id } }
        if let failure = matching.first(where: { $0.status == .failed || $0.status == .cancelled }) {
            return .failed(failure.error ?? (failure.status == .cancelled ? "Download cancelled." : "Download failed."))
        }
        return matching.count == Set(ids).count && matching.allSatisfy { $0.status == .completed } ? .ready : .waiting
    }
}

/// Native queue recovery outlives a row/detail presentation. Downloads remain server-owned.
@MainActor @Observable
public final class QueueDownloadRecovery {
    public enum Phase: Sendable { case starting, license, queued, downloading, reconnecting, retrying, failed, cancelled, complete }
    public struct State: Sendable {
        public var phase: Phase
        public var message: String
        public var fraction: Double?
        public var isBusy: Bool { ![.failed, .cancelled, .complete].contains(phase) }
    }
    private struct Key: Hashable { let host: UUID; let job: String }
    private var states: [Key: State] = [:]
    @ObservationIgnored private var attempts: [Key: UUID] = [:]
    @ObservationIgnored private var tasks: [Key: Task<Void, Never>] = [:]
    @ObservationIgnored private var boards: [UUID: [String: DownloadProgress]] = [:]
    @ObservationIgnored private var settled: [UUID: [String: DownloadJob]] = [:]

    public init() {}
    public func state(host: UUID, job: String) -> State? { states[Key(host: host, job: job)] }

    public func cancel(host: UUID, job: String) {
        let key = Key(host: host, job: job)
        attempts[key] = UUID()
        tasks.removeValue(forKey: key)?.cancel()
        states[key] = State(phase: .cancelled, message: "Download recovery cancelled. The job remains held.")
    }

    public func approveLicense(host: UUID, job: String) {
        let key = Key(host: host, job: job)
        guard states[key]?.phase == .license else { return }
        states[key] = State(phase: .starting, message: "Starting download…")
    }

    public func observe(_ event: DownloadEvent, on host: UUID) {
        if let listing = event.listing, event.type == "snapshot" { adopt(listing, on: host); return }
        var board = boards[host] ?? [:]
        if let job = DownloadBoard.apply(event, to: &board) { settled[host, default: [:]][job.id] = job }
        boards[host] = board
    }

    private func adopt(_ listing: DownloadsListing, on host: UUID) {
        boards[host] = DownloadBoard.adopt(listing)
        for job in listing.history { settled[host, default: [:]][job.id] = job }
    }

    /// Start feedback is synchronous; every network boundary is fenced before a mutation.
    public func start(
        entry: QueueEntry, host: UUID, authority: QueueAuthority, backend: any MoldBackend,
        hostName: String = "this machine", every interval: Duration = .seconds(1),
        within budget: Duration = .seconds(3600),
        isCurrent: @escaping @MainActor () -> Bool = { true },
        started: @escaping @MainActor ([String], String) -> Void = { _, _ in },
        license: @escaping @MainActor (LicenseRefusal, Bool, @escaping @MainActor () -> Void) -> Bool = { _, _, _ in false },
        refreshed: @escaping @MainActor () async -> Void = {}
    ) {
        let key = Key(host: host, job: entry.id)
        guard states[key]?.isBusy != true, let model = entry.model else { return }
                states[key] = State(phase: .starting, message: "Starting download on \(hostName)…")
        let attempt = UUID()
        attempts[key] = attempt
        tasks[key] = Task {
            defer { if attempts[key] == attempt { tasks[key] = nil } }
            do {
                try await validate(entry, authority: authority, backend: backend, isCurrent: isCurrent)
                let deadline = ContinuousClock.now.advanced(by: budget)
                let ids = try await acquire(model, entry: entry, authority: authority, key: key, backend: backend, hostName: hostName,
                                            interval: interval, deadline: deadline, isCurrent: isCurrent, license: license, attempt: attempt)
                try Task.checkCancellation()
                started(ids, model)
                guard !ids.isEmpty else { throw RecoveryError("The machine returned no download ticket. Refresh Models before retrying.") }
                while ContinuousClock.now < deadline {
                    try Task.checkCancellation()
                    guard isCurrent() else { throw RecoveryError("This job or machine changed. Download recovery stopped.") }
                    do {
                        let status = try await backend.status()
                        guard status.instanceId == authority.instanceId else { throw RecoveryError("The machine identity changed. Reopen the job before retrying.") }
                        let listing = try await backend.downloads()
                        try Task.checkCancellation()
                        adopt(listing, on: host)
                        let jobs = Array((settled[host] ?? [:]).values)
                        switch QueueDownloadSettlement.resolve(ids: ids, jobs: jobs) {
                        case .failed(let message): throw RecoveryError(message)
                        case .ready:
                            try await validate(entry, authority: authority, backend: backend, isCurrent: isCurrent)
                            states[key] = State(phase: .retrying, message: "Model ready — retrying job…")
                            do { try await backend.retryJob(authority) } catch {
                                await refreshed()
                                throw RecoveryError("Retry outcome could not be confirmed. Refresh Queue before trying again. \(error.localizedDescription)")
                            }
                            try Task.checkCancellation()
                            await refreshed()
                            try Task.checkCancellation()
                            states[key] = State(phase: .complete, message: "Retry accepted on \(hostName).")
                            return
                        case .waiting: showProgress(ids, key: key, hostName: hostName)
                        }
                    } catch let error as RecoveryError { throw error } catch is CancellationError { throw CancellationError() } catch {
                        try Task.checkCancellation()
                        guard attempts[key] == attempt else { throw CancellationError() }
                        states[key] = State(phase: .reconnecting, message: "Reconnecting — download status unavailable.", fraction: states[key]?.fraction)
                    }
                    try await Task.sleep(for: interval)
                }
                throw RecoveryError("Download status could not be confirmed. Refresh the machine before retrying.")
            } catch is CancellationError {
                // Explicit cancellation has already supplied its visible outcome.
            } catch {
                if attempts[key] == attempt, !Task.isCancelled {
                    states[key] = State(phase: .failed, message: UserFacingError.message(error.localizedDescription))
                }
            }
        }
    }

    private func acquire(_ model: String, entry: QueueEntry, authority: QueueAuthority, key: Key, backend: any MoldBackend, hostName: String,
                         interval: Duration, deadline: ContinuousClock.Instant, isCurrent: @MainActor () -> Bool,
                         license: @MainActor (LicenseRefusal, Bool, @escaping @MainActor () -> Void) -> Bool, attempt: UUID) async throws -> [String] {
        while true {
            try Task.checkCancellation()
            guard isCurrent() else { throw RecoveryError("This job or machine changed. Download recovery stopped.") }
            do { return try await DownloadAcquisition.start(model, backend: backend) } catch let MoldClientError.licenseRequired(refusal, mismatch) {
                try Task.checkCancellation()
                guard attempts[key] == attempt else { throw CancellationError() }
                states[key] = State(phase: .license, message: "Awaiting license acceptance on \(hostName).")
                guard license(refusal, mismatch, { [weak self] in
                    guard self?.attempts[key] == attempt else { return }
                    self?.approveLicense(host: key.host, job: key.job)
                }) else {
                    throw RecoveryError("Finish the other license review before starting this download.")
                }
                while states[key]?.phase == .license {
                    guard isCurrent(), ContinuousClock.now < deadline else { throw RecoveryError("License review ended or the job changed. Download recovery stopped.") }
                    try await Task.sleep(for: interval)
                }
                try Task.checkCancellation()
                // Acceptance may have waited minutes: refence the server before acquiring anything.
                try await validate(entry, authority: authority, backend: backend, isCurrent: isCurrent)
            }
        }
    }

    private func validate(_ entry: QueueEntry, authority: QueueAuthority, backend: any MoldBackend,
                          isCurrent: @MainActor () -> Bool) async throws {
        try Task.checkCancellation()
        guard isCurrent(), try await backend.status().instanceId == authority.instanceId else {
            throw RecoveryError("The machine identity changed. Reopen the job before retrying.")
        }
        let current = try await backend.queueJob(id: entry.id).job
        try Task.checkCancellation()
        guard isCurrent(), current.state == .held, current.retryable != false,
              current.authority(instanceId: authority.instanceId) == authority, current.model == entry.model,
              try await backend.status().instanceId == authority.instanceId else {
            throw RecoveryError("This job changed or is no longer held. Download recovery stopped.")
        }
        try Task.checkCancellation()
        guard isCurrent() else { throw RecoveryError("This job or machine changed. Download recovery stopped.") }
    }

    private func showProgress(_ ids: [String], key: Key, hostName: String) {
        let rows = ids.compactMap { id -> DownloadJob? in
            if let terminal = settled[key.host]?[id] { return terminal }
            guard let row = boards[key.host]?[id] else { return nil }
            return DownloadJob(id: id, model: row.model, status: row.status,
                               bytesDone: row.bytesDone ?? 0, bytesTotal: row.bytesTotal ?? 0)
        }
        guard rows.count == ids.count else {
            states[key] = State(phase: .reconnecting, message: "Waiting for the machine to confirm the download outcome.", fraction: states[key]?.fraction); return
        }
        let queued = rows.filter { $0.status != .completed }.allSatisfy { $0.status == .queued }
        let progress = QueueDownloadSettlement.progress(jobs: rows)
        states[key] = State(phase: queued ? .queued : .downloading,
                            message: queued ? "Download queued on \(hostName)." : "Downloading on \(hostName) · \(progress.sentence(bytesPerSecond: nil))",
                            fraction: progress.fraction)
    }

    private struct RecoveryError: LocalizedError {
        let message: String
        init(_ message: String) { self.message = message }
        var errorDescription: String? { message }
    }
}

/// One dispatch rule for Models and Queue; catalog acquisitions can return several tickets.
public enum DownloadAcquisition {
    public static func start(_ name: String, backend: any MoldBackend) async throws -> [String] {
        if Model.isCatalogName(name) { return try await backend.installCatalogEntry(id: name).jobIDs }
        return [try await backend.startDownload(DownloadRequest(model: name)).id]
    }
}
