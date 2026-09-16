import MoldClient
import SwiftUI

/// Running mold's engine on this Mac.
struct LocalEngineSettings: View {
    @Environment(MoldEngine.self) private var engine
    @Environment(HostStore.self) private var hosts

    var body: some View {
        Form {
            Section {
                LabeledContent("Status") { status }
                if case let .running(port) = engine.state {
                    LabeledContent("Address") {
                        // `Text` interpolation of an integer goes through
                        // LocalizedStringKey and gets the locale's grouping
                        // separator -- port 61440 renders as "61,440".
                        Text(verbatim: "http://127.0.0.1:\(port)")
                            .monospaced()
                            .textSelection(.enabled)
                    }
                }
            } footer: {
                Text(explanation)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Section {
                HStack {
                    Spacer()
                    controls
                }
            }
        }
        .formStyle(.grouped)
    }

    @ViewBuilder private var status: some View {
        switch engine.state {
        case let .unavailable(reason):
            Text(reason).foregroundStyle(.secondary)
        case .stopped:
            Text("Not running")
        case .starting:
            HStack(spacing: 6) { ProgressView().controlSize(.small); Text("Starting…") }
        case .running:
            Label("Running", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
        case let .failed(message):
            Label(message, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
        }
    }

    @ViewBuilder private var controls: some View {
        switch engine.state {
        case .stopped, .failed:
            Button("Start Engine") { start() }
        case .running:
            Button("Stop Engine") { Task { await stop() } }
        case .starting:
            Button("Start Engine") { }.disabled(true)
        case .unavailable:
            EmptyView()
        }
    }

    private var explanation: String {
        MoldEngine.isLinked
            ? "Runs mold's own engine inside this app, on Metal, so renders happen "
              + "here instead of on another machine. It appears in the machine list "
              + "like any other — the app talks to it over the same HTTP it uses for "
              + "a remote host. It can only be started once per launch."
            : "This build talks to remote machines only."
    }

    private func start() {
        engine.start()
        // Poll briefly: `start` returns as soon as the thread is spawned, and
        // the host only exists once a port is bound.
        Task {
            for _ in 0..<40 {
                try? await Task.sleep(for: .milliseconds(250))
                if let host = engine.host {
                    hosts.adoptLocalEngine(host)
                    return
                }
            }
        }
    }

    private func stop() async {
        await engine.stop()
        hosts.dropLocalEngine()
    }
}
