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
        guard !working.contains(key), !UpscalePlan.shouldPoll(jobs[key]),
              stills[key] != .working else { return }
        guard let backend = hosts.backend(for: entry.hostID) else { return }

        // No local "is one installed" check. Nothing reads `/api/models` on
        // the way to the Library, so that cache is EMPTY there -- and a
        // machine with `real-esrgan-x4plus:fp16` ready was refused in a
        // machine-failure banner naming the wrong problem. The ported policy
        // already answers for a machine whose upscalers this app has never
        // listed: its last fallback is the manifest name (`upscale.ts:24`),
        // and the HOST is the authority on whether it has it. A real refusal
        // then arrives in the machine's own words, which name the model.
        let model = UpscalePlan.defaultUpscaler(upscalers(on: entry.hostID))
        let epoch = bump(key)
        working.insert(key)
        defer { working.remove(key) }

        do {
            if entry.print.kind == .clip {
                try await startClip(key, on: backend, model: model, epoch: epoch)
            } else {
                stills[key] = .working
                let result = try await backend.upscaleLibraryImage(
                    filename: key.filename, model: model, tileSize: nil)
                guard epochs[key] == epoch else { return }
                stills[key] = .done(filename: result.filename)
                await library.refresh(on: entry.hostID)
            }
            hosts.succeeded(on: entry.hostID, doing: Self.startVerb)
        } catch {
            guard epochs[key] == epoch else { return }
            // The row says it too, beside the print it is about -- the
            // banner names the machine and not which picture failed.
            if stills[key] == .working { stills[key] = .failed(error.reasonSentence) }
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
        // The HOST is asked first. A job started from the web UI, from a
        // second Mac, or from this one before a relaunch is still this
        // print's job, and local state cannot know about any of them --
        // desktop asks the same question before it offers Start
        // (`videoUpscale.ts:72-80`). Without it, Make Bigger... on a print
        // already being upscaled starts a SECOND pass over every frame.
        //
        // A machine too old to answer the listing at all simply does not get
        // the check: that is the same absence `recover` already tolerates.
        if let existing = try? await UpscalePlan.recoverable(
            in: backend.framewiseUpscales(), filename: key.filename) {
            guard epochs[key] == epoch else { return }
            jobs[key] = existing
            poll(key)
            return
        }
        guard epochs[key] == epoch else { return }
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
        // The epoch is bumped here for the same reason `start` bumps it: an
        // `ask` already in flight is about the job BEFORE this transition,
        // and landing it afterwards rewrites a paused job as running with
        // nothing polling it. Relying on URLSession turning the cancelled
        // request into a `CancellationError` is a property of the transport,
        // not an invariant of this store.
        let epoch = bump(key)
        do {
            let next = try await backend.transitionFramewiseUpscale(id: job.id, to: transition)
            // A newer job for the same print took the key while this was in
            // flight -- this answer is about a job nobody is following.
            guard epochs[key] == epoch, jobs[key]?.id == job.id else { return }
            jobs[key] = next
            if UpscalePlan.shouldPoll(next) { poll(key) }
        } catch {
            hosts.report(error, on: key.host, doing: Self.followVerb)
            // The transition failed; the JOB did not. `bump` stopped the
            // poll before the request went out, so without this the row
            // freezes at its last frame count and offers Pause forever --
            // and the only repair is reopening the Library.
            guard epochs[key] == epoch, UpscalePlan.shouldPoll(jobs[key]) else { return }
            poll(key)
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
