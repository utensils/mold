import MoldClient
import SwiftUI

// The sections either side of the GPUs. Split from `MachinesPane` for size;
// every one of them is the same shape -- a label, a value, and an absence
// that says something rather than leaving a gap.
extension MachinesPane {

    /// The Form's first, unlabelled section: the grouped-Form idiom for "what
    /// this thing is". `navigationTitle` and `navigationSubtitle` already
    /// carry the name and the state, so there is no header band here.
    @ViewBuilder func identity(_ host: MoldHost) -> some View {
        Section {
            if case let .up(status) = hosts.reachability(of: host) {
                LabeledContent("Version", value: "mold \(status.version)")
                LabeledContent("Up", value: uptime(status.uptimeSecs))
                if let instance = status.instanceId {
                    LabeledContent("Identity") {
                        HStack(spacing: 6) {
                            Text(instance).textSelection(.enabled).lineLimit(1)
                            CopyButton(what: "Identity", value: instance)
                        }
                    }
                }
            }
            if hosts.capabilities(of: host) == nil {
                Text("This machine hasn't said what it can do yet.")
                    .foregroundStyle(.secondary)
            }
        }
    }

    /// The machine's own memory, which is a different thing from a card's --
    /// absent entirely until a snapshot arrives, because a Memory section with
    /// no figures in it is a section that says nothing.
    @ViewBuilder func memory(_ host: MoldHost) -> some View {
        if let snapshot = machines.resource(on: host.id) {
            Section("Memory") {
                LabeledContent("System") {
                    HStack(spacing: 10) {
                        Text(DeviceWords.capacity(used: snapshot.systemRam.used,
                                                  total: snapshot.systemRam.total))
                        if let reading = MemoryReading(used: snapshot.systemRam.used,
                                                       total: snapshot.systemRam.total) {
                            MemoryBar(reading: reading)
                        }
                    }
                }
                // The aggregator needs two samples before it can report CPU
                // use at all, so an absent block is "not yet", not "zero".
                if let cpu = snapshot.cpu {
                    LabeledContent("Processors", value: processors(cpu))
                }
            }
        }
    }

    /// What this machine is carrying, counted from the stores that already
    /// own those numbers rather than fetched a second time here.
    @ViewBuilder func work(_ host: MoldHost) -> some View {
        Section {
            LabeledContent("Work here") {
                figure(workCount(host), show: "Show the queue") { destination = .queue }
            }
            LabeledContent("Models here") {
                figure(modelCount(host), show: "Show the models") { destination = .models }
            }
        }
    }

    /// A count and the way through to the pane that lists it. The Queue is
    /// already sectioned by machine, so "Show" just goes there -- threading a
    /// `ScrollViewReader` through a pane to scroll to one section is state
    /// nobody asked for.
    private func figure(_ count: String, show: String,
                        act: @escaping () -> Void) -> some View {
        HStack(spacing: 10) {
            Text(count)
            Button(show, systemImage: "chevron.right", action: act)
                .labelStyle(.iconOnly)
                .buttonStyle(.borderless)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder func address(_ host: MoldHost) -> some View {
        Section("Address") {
            LabeledContent("Address") {
                HStack(spacing: 6) {
                    Text(HostAddress.displayString(for: host.baseURL))
                        .textSelection(.enabled).lineLimit(1)
                    CopyButton(what: "Address", value: host.baseURL.absoluteString)
                }
            }
            LabeledContent("API key") {
                HStack(spacing: 10) {
                    // A keyless host is a first-class state, not a warning.
                    Text(host.apiKey?.isEmpty == false
                        ? "Set for this machine"
                        : "Not needed on this machine")
                    Button("Edit…") { editing = host }
                        .buttonStyle(.link)
                }
            }
        }
    }

    /// Mobile pairing, directly under Address -- its own type
    /// (`PairingSection.swift`) rather than a function here, since it needs
    /// its own `@State` for the sheet and the pending revoke.
    @ViewBuilder func pairing(_ host: MoldHost) -> some View {
        // ABSENT for this Mac's engine, never disabled -- `MoldEngine.isPairable`
        // carries the reason and the test (review F4).
        if MoldEngine.isPairable(host) { PairingSection(host: host) }
    }

    // MARK: - The figures

    private func uptime(_ seconds: UInt64) -> String {
        Duration.seconds(seconds).formatted(
            .units(allowed: [.days, .hours, .minutes], width: .abbreviated))
    }

    private func processors(_ cpu: CpuSample) -> String {
        "\(cpu.cores) cores · \(cpu.usagePercent.formatted(.number.precision(.fractionLength(0))))%"
    }

    private func workCount(_ host: MoldHost) -> String {
        MachineFigures.workFigure(live: queue.hasLoaded(on: host.id)
            ? queue.entries(on: host.id).filter(\.state.isLive) : nil,
            alsoRunning: fleet.alsoRunning(on: host))
    }

    private func modelCount(_ host: MoldHost) -> String {
        MachineFigures.modelFigure(ready: models.hasLoaded(on: host.id) ? models.ready(on: host.id) : nil)
    }
}
