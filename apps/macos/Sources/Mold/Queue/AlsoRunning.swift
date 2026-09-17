import Foundation
import MoldClient

/// One row under **Also Running** -- work a machine is doing that has no
/// queue row of its own.
struct AlsoRunningRow: Identifiable, Equatable {
    /// Where the row came from. Two sources, because a clip upscale is never
    /// in `/api/activity` at all: `video_upscale.rs` drives its own engine
    /// cache rather than submitting scheduler work, so this app's own
    /// following is the only place it appears.
    enum Work: Equatable {
        case reported(FleetActiveWork)
        case upscale(filename: String, job: VideoUpscaleJob)
    }

    let host: MoldHost.ID
    let work: Work

    var id: String {
        switch work {
        case let .reported(row): row.id
        case let .upscale(filename, _): "\(host):upscale:\(filename)"
        }
    }

    /// What the work IS.
    var title: String {
        switch work {
        case let .reported(row): row.item.kindLabel
        case .upscale: "Upscale"
        }
    }

    /// Which print or style it is about, when the machine says.
    var subject: String? {
        switch work {
        case let .reported(row): row.item.model
        case let .upscale(filename, _): filename
        }
    }

    /// Where it has got to, in the machine's own words.
    var detail: String {
        switch work {
        case let .reported(row): row.item.phaseLabel
        case let .upscale(_, job): UpscalePlan.status(of: job)
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
        }
    }

    /// Cancellable only where the machine confirmed it for this exact item
    /// AND this app has a route to act through.
    ///
    /// Today that is the clip upscale it started and nothing else. Of the
    /// kinds drawn here, the scheduler-owned ones report `can_cancel: false`
    /// themselves (`routes_activity.rs:340`), and the one that reports
    /// `true` is a durable `sequence`, whose cancel is
    /// `/api/chain-jobs` -- an endpoint family this app does not speak, and
    /// chain authoring is deliberately out of its scope. Offering a Cancel
    /// that quietly does nothing would be worse than not offering one, so the
    /// server's own answer is necessary here and not sufficient.
    var canCancel: Bool {
        switch work {
        case .reported: false
        case let .upscale(_, job): !job.state.isTerminal
        }
    }

    /// A clip upscale can be held and picked back up; nothing else here can.
    var canPause: Bool {
        guard case let .upscale(_, job) = work else { return false }
        return UpscalePlan.shouldPoll(job)
    }

    var canResume: Bool {
        guard case let .upscale(_, job) = work else { return false }
        return job.state == .paused
    }

    /// This row is the last thing the machine said, not the current truth.
    var isStale: Bool {
        guard case let .reported(row) = work else { return false }
        return row.stale
    }

    /// Settled work this app is still holding so the answer can be read.
    var isSettled: Bool {
        guard case let .upscale(_, job) = work else { return false }
        return job.state.isTerminal
    }
}
