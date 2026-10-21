import MoldClient
import SwiftUI

/// A new attempt has one destination and one durable operation identity.
struct NativeVideoRestartSheet: View {
    @Bindable var controller: VideoRestartController
    let targets: [VideoRestartTarget]
    let backend: (VideoRestartTarget) -> (any MoldBackend)?
    let current: (VideoRestartTarget) -> Bool
    var deleting = false
    var discarded: (() -> Void)?
    var changed: (() -> Void)?
    @Environment(\.dismiss) private var dismiss
    @State private var destinationId: UUID?
    @State private var confirmingDelete = false

    private var destination: VideoRestartTarget? {
        if let record = controller.admission { return targets.first(where: record.matches) }
        return targets.first(where: { $0.host.id == destinationId }) ?? controller.source
    }

    var body: some View {
        NavigationStack {
            Form {
                if let source = controller.source {
                    LabeledContent("Original machine", value: source.host.name)
                }
                if let error = controller.error { Text(error).fixedSize(horizontal: false, vertical: true) }
                if let notice = controller.notice { Text(notice).fixedSize(horizontal: false, vertical: true) }
                if controller.isLoading { ProgressView("Checking saved video…") }
                if deleting {
                    Text("Delete this job and its recovery data? Saved Library videos remain.")
                        .fixedSize(horizontal: false, vertical: true)
                    if controller.options?.discard?.supported == true {
                        Button("Delete recovery data…", role: .destructive) { confirmingDelete = true }
                            .disabled(controller.isBusy).frame(minHeight: 44)
                    } else if !controller.isLoading {
                        Text(
                            controller.options?.discard?.reason
                                ?? "This machine does not offer deletion of this saved job. Update or reconnect to its original machine."
                        )
                    }
                } else if let record = controller.admission {
                    Text("Saved attempt · \(record.operationId.prefix(8))").font(.headline)
                    if let destination {
                        LabeledContent("Destination", value: destination.host.name)
                        if let detail = controller.parentDetail {
                            Text(
                                "\(detail.state.rawValue.capitalized) · \(detail.completedClips) of \(detail.plan.clipCount) clips"
                            )
                        } else if let detail = controller.detail {
                            Text(detail.state.rawValue.capitalized)
                        } else {
                            Text("Check the destination to find out whether this attempt was accepted.")
                        }
                        receiptActions(destination)
                    } else {
                        Text(
                            "Reconnect to the saved destination machine to check this attempt. Its identity has been kept."
                        )
                    }
                } else if let options = controller.options {
                    Picker("Machine", selection: $destinationId) {
                        ForEach(targets, id: \.host.id) { target in Text(target.host.name).tag(Optional(target.host.id))
                        }
                    }
                    .disabled(controller.isBusy)
                    Text("This starts a new attempt. The original job remains on its machine.")
                        .font(.caption).foregroundStyle(.secondaryText)
                    if let destination {
                        let permission = options.permission(destinationInstanceId: destination.instanceId)
                        if !permission.supported {
                            Text(permission.reason ?? "This video cannot restart on that machine.")
                        }
                        Button(destination.instanceId == options.sourceInstanceId ? "Start again" : "Send video") {
                            guard let target = backend(destination) else { return }
                            Task {
                                await controller.start(
                                    destination: destination, backend: target, current: { current(destination) })
                                changed?()
                            }
                        }.disabled(!permission.supported || controller.isBusy || !current(destination)).frame(
                            minHeight: 44)
                    }
                }
                if let progress = controller.progress {
                    ProgressView(progress)
                } else if controller.isBusy {
                    ProgressView("Preparing video…")
                }
            }
            .navigationTitle(deleting ? "Delete video job" : "Start again")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") {
                        controller.close()
                        dismiss()
                    }
                }
            }
            .onAppear { destinationId = controller.admission?.destinationHostId ?? controller.source?.host.id }
            .confirmationDialog(
                "Delete this job and its recovery data?", isPresented: $confirmingDelete, titleVisibility: .visible
            ) {
                Button("Delete recovery data", role: .destructive) {
                    Task {
                        if await controller.discardSource() {
                            discarded?()
                            controller.close()
                            dismiss()
                        }
                    }
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Saved Library videos remain. This job’s saved inputs and unfinished progress will be removed.")
            }
        }
        #if os(macOS)
            .frame(minWidth: 420, idealWidth: 500, minHeight: 320)
        #endif
    }

    private var canStop: Bool {
        if controller.protocolKind == .videoJob { return controller.parentDetail?.canCancel ?? true }
        return controller.detail.map { ChainJobRecoveryPolicy.canCancel(state: $0.state) } ?? true
    }

    @ViewBuilder private func receiptActions(_ destination: VideoRestartTarget) -> some View {
        if let target = backend(destination), current(destination) {
            Button("Check queue") {
                Task {
                    await controller.refresh(
                        destination: destination, backend: target, current: { current(destination) })
                    changed?()
                }
            }
            .disabled(controller.isBusy || controller.isCancelling).frame(minHeight: 44)
            if !controller.cancelledBeforeAdmission, canStop {
                Button("Stop", role: .destructive) {
                    Task {
                        await controller.cancel(
                            destination: destination, backend: target, current: { current(destination) })
                        changed?()
                    }
                }
                .disabled(controller.isBusy || controller.isCancelling).frame(minHeight: 44)
            }
        }
    }
}

struct NativeVideoRestartReceipts: View {
    @Bindable var controller: VideoRestartController
    let open: (VideoRestartAdmission) -> Void
    var body: some View {
        ForEach(controller.pending) { record in
            VStack(alignment: .leading, spacing: 8) {
                Label("Saved video attempt", systemImage: "film.stack").font(.headline)
                Text(
                    "\(record.cancelRequested ? "Stop requested":"Check acceptance") · \(record.operationId.prefix(8))"
                )
                .font(.caption).foregroundStyle(.secondaryText)
                Button("Check saved attempt…") { open(record) }.disabled(controller.isBusy).frame(minHeight: 44)
            }.padding(.vertical, 8)
        }
    }
}
