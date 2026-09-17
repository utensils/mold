import Foundation
import MoldClient

// Starting an upscale, and telling a clip job what to do next.
@MainActor
extension UpscaleStore {
    /// The verb every failure about this is keyed on, so a retry replaces
    /// its predecessor in the banner rather than piling up.
    static let startVerb = "make that print bigger"

    /// Makes this print bigger on the machine that holds it.
    ///
    /// A still is SYNCHRONOUS -- `POST /api/gallery/upscale` writes the
    /// bigger picture into that machine's Library before it answers, so
    /// there is nothing to follow and the gallery is simply re-read. A clip
    /// is a durable job, and this app follows it.
    func start(_ entry: LibraryEntry) async {
        let key = Key(host: entry.hostID, filename: entry.print.filename)
        // Pressing it twice is one request. Without this the second press
        // starts a SECOND 124-frame job against the same print, and the
        // first one's id is lost the moment the second answers.
        guard !working.contains(key), !UpscalePlan.shouldPoll(jobs[key]) else { return }
        guard let backend = hosts.backend(for: entry.hostID) else { return }

        let choices = upscalers(on: entry.hostID)
        guard choices.contains(where: \.isDownloaded) else {
            hosts.report(NoUpscalerInstalled(), on: entry.hostID, doing: Self.startVerb)
            return
        }
        let model = UpscalePlan.defaultUpscaler(choices)
        let epoch = bump(key)
        working.insert(key)
        defer { working.remove(key) }

        do {
            if entry.print.kind == .clip {
                try await startClip(key, on: backend, model: model, epoch: epoch)
            } else {
                _ = try await backend.upscaleLibraryImage(
                    filename: key.filename, model: model, tileSize: nil)
                guard epochs[key] == epoch else { return }
                await library.refresh()
            }
            hosts.succeeded(on: entry.hostID, doing: Self.startVerb)
        } catch {
            guard epochs[key] == epoch else { return }
            hosts.report(error, on: entry.hostID, doing: Self.startVerb)
        }
    }

    /// The clip half, and the one sequence that is easy to get wrong: a
    /// cancel arriving while the create is still in flight.
    ///
    /// There is no id to cancel with until the host answers, so the cancel is
    /// REMEMBERED and spent the instant there is one. Dropping it instead
    /// leaves a job upscaling every frame of a clip on a machine nobody is
    /// watching, for a print the person has already given up on.
    private func startClip(
        _ key: Key, on backend: any MoldBackend, model: String, epoch: Int
    ) async throws {
        let job = try await backend.startFramewiseUpscale(
            filename: key.filename, model: model, tileSize: nil)
        guard epochs[key] == epoch else { return }
        jobs[key] = job
        guard cancelOnArrival.remove(key) == nil else {
            await transition(key, to: .cancel)
            return
        }
        poll(key)
    }

    /// Pause, resume or cancel the clip job following this print.
    ///
    /// The answer IS the job's new state, so nothing re-reads to find out
    /// what it did. Polling resumes only if the new state is still moving --
    /// a paused job moves when somebody resumes it, and that reply is the
    /// next answer.
    func transition(_ key: Key, to transition: FramewiseTransition) async {
        guard let job = jobs[key] else {
            // Nothing to act on yet. A cancel is still meaningful: it is
            // about the create that is in flight right now.
            if transition == .cancel, working.contains(key) { cancelOnArrival.insert(key) }
            return
        }
        guard let backend = hosts.backend(for: key.host) else { return }
        pollers[key]?.cancel()
        pollers[key] = nil
        do {
            let next = try await backend.transitionFramewiseUpscale(id: job.id, to: transition)
            // A newer job for the same print took the key while this was in
            // flight -- this answer is about a job nobody is following.
            guard jobs[key]?.id == job.id else { return }
            jobs[key] = next
            if UpscalePlan.shouldPoll(next) { poll(key) }
        } catch {
            hosts.report(error, on: key.host, doing: Self.followVerb)
        }
    }

    /// Stops following a settled job, so its row leaves the pane.
    func forget(_ key: Key) {
        guard let job = jobs[key], job.state.isTerminal else { return }
        pollers[key]?.cancel()
        pollers[key] = nil
        jobs[key] = nil
    }

    /// Takes the key for a new answer and abandons every older one: an epoch
    /// is the only thing that tells a reply about THIS job from a reply about
    /// the one it replaced.
    func bump(_ key: Key) -> Int {
        pollers[key]?.cancel()
        pollers[key] = nil
        let next = (epochs[key] ?? 0) + 1
        epochs[key] = next
        return next
    }
}
