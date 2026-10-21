import MoldClient
import SwiftUI

/// A movie stays one parent row even while its child clips change machines.
struct NativeDurationJobs: View {
    @Bindable var coordinator: VideoJobCoordinator
    let backend: (VideoJobOwner) -> (any MoldBackend)?
    let current: (VideoJobOwner) -> Bool
    var open: ((MoldHost, String) -> Void)?
    var restart: ((ObservedVideoJob) -> Void)?
    var canRestart: (ObservedVideoJob) -> Bool = { _ in false }
    @State private var deleting: ObservedVideoJob?

    var body: some View {
        Group {
        if coordinator.unavailableRecoveryCount > 0 {
            Text("\(coordinator.unavailableRecoveryCount) saved video attempts belong to removed or changed connections. Restore their original machines to check them.")
                .fixedSize(horizontal: false, vertical: true)
        }
        if let error = coordinator.error {
            Text(error).fixedSize(horizontal: false, vertical: true)
        }
        ForEach(coordinator.jobs) { job in
            VStack(alignment: .leading, spacing: 8) {
                Label("Video · \(job.owner.host.name)", systemImage: "film.stack").font(.headline)
                if let detail = job.detail {
                    Text(detail.plan.model).font(.caption).foregroundStyle(.secondaryText)
                    Text("\(detail.state.rawValue.capitalized) · \(min(detail.completedClips, detail.plan.clipCount)) of \(detail.plan.clipCount) clips")
                    Text("\(VideoDurationDraft.seconds(milliseconds: detail.plan.totalDurationMs)) seconds · \(job.operationId.prefix(8))")
                        .font(.caption).foregroundStyle(.secondaryText)
                    if let error = detail.errorDetail ?? detail.error { Text(error).fixedSize(horizontal: false, vertical: true) }
                    if let filename = detail.outputFilename, let open {
                        Button("Show video in Library") { open(job.owner.host, filename) }.frame(minHeight: 44)
                    }
                } else { Text(job.cancelledBeforeAdmission ? "Cancelled before queueing" : "Checking saved video attempt…") }
                if let message = job.message { Text(message).fixedSize(horizontal: false, vertical: true) }
                actions(job)
            }.padding(.vertical, 8)
        }
        ForEach(Array(coordinator.pagination.values).sorted { $0.owner.host.name < $1.owner.host.name }, id: \.owner.host.id) { page in
            if page.hasMore || page.error != nil || page.isLoading {
                VStack(alignment: .leading, spacing: 6) {
                    if let error = page.error { Text(error).fixedSize(horizontal: false, vertical: true) }
                    if let target = backend(page.owner), current(page.owner) {
                        Button(page.isLoading ? "Loading videos…" : page.error != nil
                               ? "Retry videos on \(page.owner.host.name)" : "Load more videos on \(page.owner.host.name)") {
                            Task { await coordinator.loadMore(owner: page.owner, backend: target, current: { current(page.owner) }) }
                        }.disabled(page.isLoading).frame(minHeight: 44)
                    }
                }
            }
        }
        }
        .confirmationDialog("Delete this job and its recovery data?", isPresented: Binding(
            get: { deleting != nil }, set: { if !$0 { deleting = nil } }), titleVisibility: .visible) {
                if let job = deleting, let target = backend(job.owner) {
                    Button("Delete recovery data", role: .destructive) {
                        deleting = nil
                        Task { await coordinator.delete(job, backend: target, current: { current(job.owner) }) }
                    }
                }
                Button("Cancel", role: .cancel) { deleting = nil }
            } message: { Text("Saved Library videos remain. This job’s saved inputs and unfinished progress will be removed.") }
    }

    @ViewBuilder private func actions(_ job: ObservedVideoJob) -> some View {
        if job.cancelledBeforeAdmission || job.detail?.state == .completed {
            Button("Dismiss video") { coordinator.dismiss(job) }.frame(minHeight: 44)
        }
        if let target = backend(job.owner), current(job.owner) {
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 12) { controls(job, backend: target) }
                VStack(alignment: .leading, spacing: 8) { controls(job, backend: target) }
            }
            .disabled(coordinator.isMutating(job))
        } else {
            Text("Reconnect to this video’s original machine to check or change it.")
                .font(.caption).foregroundStyle(.secondaryText)
        }
    }

    @ViewBuilder private func controls(_ job: ObservedVideoJob, backend: any MoldBackend) -> some View {
        Button("Check queue") {
            Task { await coordinator.refresh(job.operationId, owner: job.owner, backend: backend, current: { current(job.owner) }) }
        }.frame(minHeight: 44)
        if job.detail?.canCancel == true || (job.detail == nil && !job.cancelledBeforeAdmission) {
            Button(coordinator.isCancellationRequested(job) ? "Check Stop" : "Stop", role: .destructive) {
                Task { await coordinator.cancel(job, backend: backend, current: { current(job.owner) }) }
            }.frame(minHeight: 44)
        }
        if job.detail?.canResume == true {
            Button("Resume") { Task { await coordinator.resume(job, backend: backend, current: { current(job.owner) }) } }.frame(minHeight: 44)
        }
        if let restart, canRestart(job), job.detail?.state.canRequestRestart == true {
            Button("Start again…") { restart(job) }.frame(minHeight:44)
        }
        if job.detail?.canDelete == true {
            Button("Delete job…", role: .destructive) { deleting = job }.frame(minHeight: 44)
        }
    }
}
