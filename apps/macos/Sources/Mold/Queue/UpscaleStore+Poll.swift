import Foundation
import MoldClient

// Following a clip job, and finding one again after a restart.
@MainActor
extension UpscaleStore {
    static let followVerb = "follow that upscale"

    /// Asks about this print's job every `interval` until it settles.
    ///
    /// Every answer is fenced twice: on the EPOCH, so a reply about a job
    /// that has since been replaced is dropped rather than written over its
    /// successor, and on the job ID, because a machine can be following a
    /// different job for the same print by the time this lands.
    func poll(_ key: Key) {
        pollers[key]?.cancel()
        let epoch = epochs[key] ?? 0
        // No `[weak self]`: this store lives as long as the app, the same
        // reasoning `QueueStore+Live`'s own coalescer uses.
        let token = (pollTokens[key] ?? 0) + 1
        pollTokens[key] = token
        following.insert(key)
        pollers[key] = Task {
            // Only THIS poll's own claim is given back -- a later poll for
            // the same key owns it by then, and a pause-then-resume replaces
            // the poll without replacing the job, so the epoch cannot tell
            // the two apart.
            defer { if pollTokens[key] == token { following.remove(key) } }
            while !Task.isCancelled {
                guard let job = jobs[key], UpscalePlan.shouldPoll(job) else { return }
                // `try?` here would swallow the cancellation and spin this at
                // full speed on the main actor.
                do { try await Task.sleep(for: interval) } catch { return }
                guard !Task.isCancelled, epochs[key] == epoch else { return }
                guard await ask(key, about: job.id, epoch: epoch) else { return }
            }
        }
    }

    /// One ask. `false` ends the loop -- the job settled, the answer was
    /// stale, or the machine could not be reached.
    private func ask(_ key: Key, about id: String, epoch: Int) async -> Bool {
        guard let backend = hosts.backend(for: key.host) else { return false }
        let next: VideoUpscaleJob
        do {
            next = try await backend.framewiseUpscale(id: id)
        } catch {
            guard epochs[key] == epoch else { return false }
            // Stop asking, and say so. A machine that has gone away answers
            // this way every 750 ms otherwise, and there is nothing to learn
            // from asking it again -- the repair is `recover()`, which runs
            // whenever the Library is opened and finds the job still running.
            hosts.report(error, on: key.host, doing: Self.followVerb)
            return false
        }
        guard epochs[key] == epoch, jobs[key]?.id == id else { return false }
        jobs[key] = next
        hosts.succeeded(on: key.host, doing: Self.followVerb)
        guard next.state == .completed else { return !next.state.isTerminal }
        // The bigger clip is a NEW row in that machine's gallery, and
        // nothing else is going to go and look.
        await library.refresh()
        return false
    }

    /// Finds the clip upscales already running on every machine, and follows
    /// them. Called when the Library opens.
    ///
    /// This is what makes a job survive a restart, a second Mac, and a poll
    /// this app gave up on. One listing per machine rather than one call per
    /// print: a library of 1,500 prints must not become 1,500 requests.
    func recover() async {
        for host in hosts.hosts where hosts.capabilities[host.id]?.canUpscaleClips == true {
            await recover(on: host.id)
        }
    }

    /// Not `private`: a test drives one machine.
    func recover(on host: MoldHost.ID) async {
        guard let backend = hosts.backend(for: host) else { return }
        let listing: [VideoUpscaleJob]
        do {
            listing = try await backend.framewiseUpscales()
        } catch {
            // An older host has no such route at all, and an unreachable one
            // is already saying so elsewhere. Neither is worth a second line
            // about a job nobody asked for yet.
            return
        }
        for job in listing {
            guard let filename = job.libraryFilename, !job.state.isTerminal else { continue }
            let key = Key(host: host, filename: filename)
            // Already being asked about on a timer, or in the middle of being
            // started -- either way this listing is older news than what the
            // store holds. Deliberately NOT "we hold this id already": a poll
            // that gave up leaves the id behind, and that is exactly the job
            // this is here to pick back up.
            guard !following.contains(key), !working.contains(key) else { continue }
            _ = bump(key)
            jobs[key] = job
            poll(key)
        }
    }
}
