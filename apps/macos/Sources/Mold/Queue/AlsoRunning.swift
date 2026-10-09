import Foundation
import MoldClient

/// One row under **Also Running** -- work a machine is doing that has no
/// queue row of its own.
struct AlsoRunningRow: Identifiable, Equatable {
    /// Where the row came from. Two sources, because what `/api/activity`
    /// says about a clip upscale is one FRAME under a fresh uuid
    /// (`video_upscale.rs:1271-1281`) -- the durable job this app follows is
    /// the only thing that names the print and counts the clip. See
    /// `AlsoRunning.rows`, which is where the machine's row gives way.
    enum Work: Equatable {
        case reported(FleetActiveWork)
        case upscale(filename: String, job: VideoUpscaleJob)
        /// A STILL, which has no durable job -- the request itself is the
        /// work, and this is the only feedback there is for it.
        case still(filename: String, state: StillUpscale)
    }

    let host: MoldHost.ID
    let work: Work

    var id: String {
        switch work {
        case let .reported(row): row.id
        case let .upscale(filename, _): "\(host):upscale:\(filename)"
        case let .still(filename, _): "\(host):still:\(filename)"
        }
    }

    /// What the work IS.
    var title: String {
        switch work {
        case let .reported(row): row.item.kindLabel
        case .upscale, .still: "Upscale"
        }
    }

    /// Which print or style it is about, when the machine says.
    var subject: String? {
        switch work {
        case let .reported(row): row.item.model
        case let .upscale(filename, _): filename
        case let .still(filename, _): filename
        }
    }

    /// Where it has got to, in the machine's own words.
    var detail: String {
        switch work {
        case let .reported(row): row.item.phaseLabel
        case let .upscale(_, job): UpscalePlan.status(of: job)
        case let .still(_, state):
            switch state {
            case .working: "Making a bigger copy"
            case let .done(filename): "Complete — \(filename)"
            case let .failed(sentence): sentence
            }
        }
    }

    /// 0...1, or nil where the machine has not counted anything yet.
    var progress: Double? {
        switch work {
        case let .reported(row):
            guard let current = row.item.current, let total = row.item.total, total > 0 else {
                return nil
            }
            return min(1, max(0, Double(current) / Double(total)))
        case let .upscale(_, job): return UpscalePlan.progress(of: job)
        // One picture, one pass: there is nothing to count.
        case .still: return nil
        }
    }

    /// Cancellable only where the machine confirmed it for this exact item
    /// Ephemeral long-clip generations use their own chain lifecycle,
    /// independently of singleton generation cancellation capabilities.
    var canCancel: Bool {
        switch work {
        case let .reported(row):
            !row.stale && !row.unavailableKind && row.item.kind == "generation" && row.item.execution == "chain"
                && row.item.canCancel && ["queued", "held", "paused", "running", "loading", "preparing"].contains(row.item.phase)
        case let .upscale(_, job): [.queued, .running, .finalizing, .paused].contains(job.state)
        // The request IS the work and the machine is already doing it;
        // there is no route that would call it off.
        case .still: false
        }
    }

    /// Why a live row offers no working Cancel, in one line -- shown on the
    /// row and behind the inert item its menu still carries. A row that
    /// offered nothing and explained nothing read as broken
    /// (2026-09-17, a queued prompt rewrite).
    var stopNote: String? {
        guard !isSettled, !canCancel else { return nil }
        switch work {
        case let .reported(row) where row.item.canCancel:
            return "Manage this work from the machine\u{2019}s web app; this app does not drive this activity type."
        case .reported:
            return "This machine doesn\u{2019}t offer a way to stop this."
        case .still:
            return "The machine is already making this; there is no way to call it off."
        case .upscale:
            return nil
        }
    }

    /// A clip upscale can be held and picked back up; nothing else here can.
    var canPause: Bool {
        guard case let .upscale(_, job) = work else { return false }
        return UpscalePlan.shouldPoll(job)
    }

    var canResume: Bool {
        switch work {
        case let .reported(row):
            !row.stale && !row.unavailableKind && row.item.kind == "generation"
                && row.item.execution == "chain" && row.item.phase == "paused"
        case let .upscale(_, job): job.state == .paused
        case .still: false
        }
    }

    /// This row is the last thing the machine said, not the current truth.
    var isStale: Bool {
        guard case let .reported(row) = work else { return false }
        return row.stale
    }

    /// Settled work this app is still holding so the answer can be read.
    var isSettled: Bool {
        switch work {
        case .reported: false
        case let .upscale(_, job): job.state.isTerminal
        case let .still(_, state): state != .working
        }
    }
}
