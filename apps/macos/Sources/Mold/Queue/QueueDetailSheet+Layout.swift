import MoldClient
import SwiftUI

extension QueueDetailSheet {
    var header: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text("Job Details").font(.caption).foregroundStyle(.secondary)
            Text((current ?? entry).modelHeadline)
                .font(.title2.weight(.semibold))
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
            Label(activeHost.name, systemImage: "desktopcomputer")
                .font(.callout).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    var overview: some View {
        let status = QueueDetailPresentation.status(current: current, progress: progress)
        return HStack(alignment: .top, spacing: 20) {
            QueueSourceThumbnail(entry: entry, host: activeHost, size: 120, detailed: true)
            VStack(alignment: .leading, spacing: 8) {
                Text(status.title).font(.headline)
                if let message = status.message {
                    Text(message).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if let recovery = downloads.queueDownloads.state(host: host.id, job: entry.id) {
                    Text(recovery.message).foregroundStyle(.secondary)
                    if let fraction = recovery.fraction { ProgressView(value: fraction).accessibilityLabel("Model download") }
                    else if recovery.isBusy { ProgressView().controlSize(.small).accessibilityLabel(recovery.message) }
                }
                if let fraction = status.fraction {
                    ProgressView(value: fraction)
                        .accessibilityLabel("Render progress")
                } else if current?.state == .running {
                    ProgressView().controlSize(.small)
                        .accessibilityLabel("Preparing the render")
                }
                if let text = status.stepText {
                    Text(text).font(.caption).monospacedDigit().foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    var technicalDetails: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button { showsTechnicalDetails.toggle() } label: {
                HStack(spacing: 6) {
                    Image(systemName: showsTechnicalDetails ? "chevron.down" : "chevron.right")
                        .font(.caption.weight(.semibold))
                        .accessibilityHidden(true)
                    Text("Technical details")
                    Spacer(minLength: 0)
                }
                .frame(minHeight: 30)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Technical details")
            .accessibilityValue(showsTechnicalDetails ? "Expanded" : "Collapsed")
            .help(showsTechnicalDetails ? "Hide the job and model identifiers" : "Show identifiers you can copy when troubleshooting this job")
            if showsTechnicalDetails {
                VStack(alignment: .leading, spacing: 10) {
                    identifier("Job ID", value: entry.id)
                    if let model = entry.model { identifier("Model ID", value: model) }
                }
            }
        }
        .font(.callout)
    }

    private func identifier(_ title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            HStack(alignment: .top) {
                Text(value).font(.caption.monospaced()).textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 8)
                CopyButton(what: title, value: value)
            }
        }
    }

    var actions: some View {
        HStack {
            if let current {
                let actions = QueueRowActions.resolve(current, on: hosts.capabilities[host.id])
                if actions.retry, case .missingModel = queue.hold(for: current, on: host.id) {
                    Button("Download and Retry") { downloads.recover(current, on: activeHost, queue: queue) }
                        .disabled(downloads.queueDownloads.state(host: host.id, job: entry.id)?.isBusy == true)
                        .help("Download the missing model on this machine, then retry this held job")
                } else if actions.retry {
                    Button("Retry Job") { act(.retry, current) }.help("Try this failed job again on its machine")
                }
                if actions.pause {
                    Button("Pause") { act(.pause, current) }.help("Pause this job")
                }
                if actions.resume {
                    Button("Resume") { act(.resume, current) }.help("Resume this paused job")
                }
                if actions.cancel {
                    Button("Cancel Job", role: .destructive) { act(.cancel, current) }
                        .help("Cancel this job on its machine")
                }
            }
            Spacer()
            Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
                .help("Close job details; the job keeps its current state")
        }
    }

}
