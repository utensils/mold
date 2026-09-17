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
                // The one question a local engine gets asked: which library is
                // this, and is it the same one `mold` uses in a terminal.
                LabeledContent("Home") {
                    Text(verbatim: home.url.path(percentEncoded: false))
                        .truncationMode(.head)
                        .lineLimit(1)
                        .textSelection(.enabled)
                        .help(homeExplanation)
                }
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

            // Fact 7 (design): this engine is keyless, so `auth_required`
            // is always false and it can never hold a paired client.
            // Pairing lives on the Machines destination instead, for a
            // machine the app holds an operator key for (design decision
            // 12) -- the actual pairing UI there is S5, not yet built.
            Section {
                Text("""
                     This Mac's engine has no API key, so there's nothing to pair. \
                     Pair a phone with a machine under Machines instead.
                     """)
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }

    private var home: MoldHome { MoldHome.resolve() }

    private var homeExplanation: String {
        switch home.source {
        case .environment: "From MOLD_HOME in this app's environment."
        case .saved: "The home chosen in Mold Desktop, shared with every mold on this Mac."
        case .fallback: "The default home, shared with every mold on this Mac."
        }
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
