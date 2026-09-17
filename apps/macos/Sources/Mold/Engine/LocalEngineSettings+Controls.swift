import MoldClient
import SwiftUI

/// What the pane SAYS the engine is doing, and what it offers about it.
/// Split from `LocalEngineSettings` only for the file-size floor.
extension LocalEngineSettings {
    @ViewBuilder var status: some View {
        switch engine.state {
        case let .unavailable(reason):
            Text(reason).foregroundStyle(.secondary)
        case .stopped:
            Text("Not running")
        case .starting:
            HStack(spacing: 6) { ProgressView().controlSize(.small); Text("Starting…") }
        case .running:
            Label("Running", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
        case let .stopping(message):
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text(message).fixedSize(horizontal: false, vertical: true)
            }
        case let .failed(failure):
            Label(failure.reason, systemImage: "exclamationmark.triangle")
                .foregroundStyle(.orange)
        }
    }

    /// Start is offered only where it can work. It used to be offered for
    /// every failure while `start()` refused anything but `.stopped`, so the
    /// button was enabled and inert -- and the engine bootstraps once per
    /// process, so for half those failures the honest answer is a relaunch
    /// (review 05-M2).
    @ViewBuilder var controls: some View {
        switch engine.state {
        case .running:
            Button("Stop Engine") { Task { await stop() } }
        case .starting, .stopping:
            Button("Start Engine") { }.disabled(true)
        case .unavailable:
            EmptyView()
        default:
            if engine.canStart {
                Button("Start Engine") { start() }
            } else {
                Button("Relaunch Mold…") { EngineRelaunch.now() }
            }
        }
    }

}
