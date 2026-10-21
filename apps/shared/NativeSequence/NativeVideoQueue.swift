import MoldClient
import SwiftUI

/// Both native queues use the same explicit parent protocol and recovery sheet.
struct NativeVideoQueue: View {
    let sequence: NativeSequenceSession
    let videos: VideoJobCoordinator
    let restarts: VideoRestartController
    let targets: [VideoRestartTarget]
    let backend: (VideoRestartTarget) -> (any MoldBackend)?
    let current: (VideoRestartTarget) -> Bool
    let capabilities: (UUID) -> Capabilities?
    let open: (MoldHost, String) -> Void
    var changed: (() -> Void)?
    @State private var showingRecovery = false
    @State private var deleting: NativeSequenceSession.Job?

    var body: some View {
        Group {
            NativeVideoRestartReceipts(controller: restarts) { record in
                if restarts.restorePending(record) {
                    deleting = nil
                    showingRecovery = true
                }
            }
            if let error = restarts.error, restarts.pending.isEmpty {
                Text(error).fixedSize(horizontal: false, vertical: true)
            }
            NativeDurationJobs(
                coordinator: videos,
                backend: { owner in
                    let target = VideoRestartTarget(host: owner.host, instanceId: owner.instanceId)
                    return current(target) ? backend(target) : nil
                }, current: { current(.init(host: $0.host, instanceId: $0.instanceId)) }, open: open,
                restart: { job in
                    present(host: job.owner.host, instance: job.owner.instanceId, id: job.operationId, kind: .videoJob)
                },
                canRestart: { capabilities($0.owner.host.id)?.canRestartVideoJob == true })
            NativeSequenceJobs(
                session: sequence,
                backend: { job in
                    guard let target = target(job), current(target) else { return nil }
                    return backend(target)
                }, open: open,
                restart: { job in
                    present(host: job.host, instance: job.installation, id: job.jobId, kind: .chain)
                },
                discard: { job in
                    present(host: job.host, instance: job.installation, id: job.jobId, kind: .chain, discard: job)
                }, canRestart: { capabilities($0.host.id)?.canRestartChain == true })
        }
        .sheet(isPresented: $showingRecovery, onDismiss: { restarts.close() }) {
            NativeVideoRestartSheet(
                controller: restarts, targets: targets, backend: backend, current: current,
                deleting: deleting != nil,
                discarded: {
                    if let deleting, let target = target(deleting), current(target) {
                        sequence.removeDiscarded(deleting)
                    }
                    changed?()
                }, changed: changed)
        }
    }

    private func target(_ job: NativeSequenceSession.Job) -> VideoRestartTarget? {
        guard let instance = job.installation else { return nil }
        return .init(host: job.host, instanceId: instance)
    }
    private func present(
        host: MoldHost, instance: String?, id: String, kind: VideoRestartProtocol,
        discard: NativeSequenceSession.Job? = nil
    ) {
        guard let instance else { return }
        let source = VideoRestartTarget(host: host, instanceId: instance)
        guard current(source), let service = backend(source) else { return }
        deleting = discard
        showingRecovery = true
        Task {
            await restarts.open(
                source: source, jobId: id, protocolKind: kind, backend: service, current: { current(source) })
        }
    }
}
