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

            // The engine is started WITH a key (review 05-H1): loopback is
            // not a boundary a browser respects, and a keyless engine let any
            // page that found the port read and delete the library. The key
            // is this app's; a phone still cannot reach 127.0.0.1 on this
            // Mac, so pairing here would hand out a credential nothing could
            // use. Pairing lives on the Machines destination, for a machine
            // reachable over the network.
            Section {
                Text("""
                     This Mac's engine is reached over loopback with a key only this app \
                     holds, so nothing else on this Mac — or in a browser — can drive it. \
                     A phone can't reach loopback here, so pair one with a machine under \
                     Machines instead.
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
        case .stopping:
            HStack(spacing: 6) { ProgressView().controlSize(.small); Text("Finishing…") }
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
    @ViewBuilder private var controls: some View {
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
        // The host exists once the engine has ANSWERED, which on a cold home
        // with a large gallery is well past the old ten-second window: the
        // engine's own probe waits for `/api/status`, so this waits for it.
        Task {
            while true {
                if let host = engine.host {
                    hosts.adoptLocalEngine(host)
                    return
                }
                switch engine.state {
                case .starting: try? await Task.sleep(for: .milliseconds(250))
                default: return
                }
            }
        }
    }

    private func stop() async {
        await engine.stop()
        hosts.dropLocalEngine()
    }
}
