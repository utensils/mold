import Foundation
import Synchronization

final class VideoJobOutputRead: Sendable {
    let owner: VideoJobOwner
    private let cancelled = Mutex(false)
    init(owner: VideoJobOwner) { self.owner = owner }
    var isCancelled: Bool { cancelled.withLock { $0 } }
    func cancel() { cancelled.withLock { $0 = true } }
}

public extension VideoJobCoordinator {
    func isCheckingOutput(_ job: ObservedVideoJob) -> Bool {
        outputReads[job.id].map { $0.owner == job.owner && !$0.isCancelled } ?? false
    }

    /// A finished parent names an output, but only its current Gallery owns
    /// navigation. A failed read is unknown, never evidence of removal.
    @discardableResult
    func checkOutput(_ job: ObservedVideoJob, backend: any MoldBackend,
                     current: @escaping @MainActor () -> Bool, force: Bool = false) async -> Bool? {
        guard !isRemoved(job.operationId, owner: job.owner), current(), !Task.isCancelled,
              let row = index(job.operationId, owner: job.owner),
              jobs[row].detail?.state == .completed, let filename = jobs[row].detail?.outputFilename else { return nil }
        if let existing = outputReads[job.id], existing.owner == job.owner, !existing.isCancelled { return nil }
        if !force, let known = jobs[row].outputAvailable { return known }
        outputReads[job.id]?.cancel()
        let read = VideoJobOutputRead(owner: job.owner); outputReads[job.id] = read
        let fresh = { @MainActor in !Task.isCancelled && !read.isCancelled && current()
            && !self.isRemoved(job.operationId, owner: job.owner)
            && self.outputReads[job.id] === read
            && self.jobs.first(where: { $0.id == job.id && $0.owner == job.owner })?.detail?.outputFilename == filename }
        defer { if outputReads[job.id] === read { outputReads[job.id] = nil } }
        return await withTaskCancellationHandler {
            do {
                try await verify(job.owner, backend: backend, current: fresh)
                let print = try await backend.galleryPrint(filename)
                guard fresh() else { return nil }
                let status = try await backend.status()
                guard status.instanceId == job.owner.instanceId else { throw changed() }
                guard fresh(), let index = index(job.operationId, owner: job.owner) else { return nil }
                guard print == nil || print?.filename == filename else { throw MoldClientError.malformedResponse }
                let available = print != nil && print?.trashedAt == nil
                jobs[index].outputAvailable = available
                jobs[index].outputMessage = nil
                return available
            } catch {
                if fresh(), let index = index(job.operationId, owner: job.owner) {
                    jobs[index].outputMessage = "Couldn’t check the saved video. \(UserFacingError.describe(error))"
                }
                return nil
            }
        } onCancel: { read.cancel() }
    }

    /// Called only from the shells' authenticated current-host event stream.
    func galleryChanged(_ event: MoldEvent.Gallery, owner: VideoJobOwner) {
        let filename: String
        let available: Bool?
        switch event {
        case let .removed(name), let .trashed(name): filename = name; available = false
        case let .added(name, row, _, _), let .updated(name, row), let .restored(name, row):
            filename = name
            available = row.flatMap { $0.filename == name ? $0.trashedAt == nil : nil }
        case .collectionsChanged: return
        }
        for index in jobs.indices where jobs[index].owner == owner && jobs[index].detail?.outputFilename == filename {
            outputReads[jobs[index].id]?.cancel(); outputReads[jobs[index].id] = nil
            jobs[index].outputAvailable = available
            jobs[index].outputMessage = nil
        }
    }
}
