import MoldClient
import SwiftUI

/// One machine: what it is made of, what it is doing with it, and the switch
/// per card that changes that.
///
/// A grouped `Form`, not a `List` and not a `Table`. Every row here is a label
/// beside a value, which is exactly what `LabeledContent` in a grouped Form
/// draws -- with the inset card and the section headings for free. A `Table`
/// would want columns and sorting nobody asked for, over four rows.
struct MachinesPane: View {
    // Not `private`: `MachinesPane+Sections` reads all of these, and `private`
    // does not cross a file boundary even within one type.
    @Environment(HostStore.self) var hosts
    @Environment(MachineStore.self) var machines
    @Environment(QueueStore.self) var queue
    @Environment(ModelStore.self) var models
    /// The same key the sidebar declares, over the same suite. Two views
    /// sharing one preference by name stay in sync with no plumbing.
    @AppStorage("selectedMachine", store: AppStorageSuite.defaults) var selectedMachine = ""
    @Binding var destination: Destination
    @State var editing: MoldHost?
    /// Whether "Work here" and "Models here" have anything behind them yet.
    @State private var countsLoaded = false

    var selected: MoldHost? { hosts.machine(selected: selectedMachine) }

    var body: some View {
        Group {
            if let selected {
                machine(selected)
            } else {
                ContentUnavailableView("No machines yet", systemImage: "server.rack",
                                       description: Text("Add one in Settings. Its name or IP is enough."))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .failureBanner(hosts)
        .navigationTitle(selected?.name ?? "Machines")
        .navigationSubtitle(selected.map { hosts.reachability(of: $0).summary ?? "" } ?? "")
        .toolbar { toolbar }
        .sheet(item: $editing) { host in
            HostEditor(host: host) { name, url, key in
                hosts.update(MoldHost(id: host.id, name: name, baseURL: url, apiKey: key))
            }
        }
        // The `id:` restarts both halves when the selection moves. The store's
        // single-stream invariant means the machine we were watching is
        // already let go by the time the new one opens.
        .task(id: selected?.id) {
            guard let id = selected?.id else { return }
            await machines.refresh(id)
            await loadCounts()
            machines.watchResources(on: id)
        }
        .onDisappear { machines.stopWatchingResources() }
        .focusedSceneValue(\.refreshAction) { refresh() }
    }

    @ViewBuilder private func machine(_ host: MoldHost) -> some View {
        if hosts.isUp(host) {
            Form {
                identity(host)
                gpus(host)
                memory(host)
                work(host)
                address(host)
                PeerSection(host: host)
            }
            .formStyle(.grouped)
        } else {
            unreachable(host)
        }
    }

    /// A machine that is not answering has nothing to show but its own state
    /// and the two things that might fix it. The failure banner above says
    /// what it could not do; this says what it IS.
    private func unreachable(_ host: MoldHost) -> some View {
        ContentUnavailableView {
            Label(host.name, systemImage: "server.rack")
        } description: {
            Text(hosts.reachability(of: host).sentence ?? "This machine hasn't answered yet.")
        } actions: {
            Button("Check Now") { Task { await hosts.refresh(host) } }
            Button("Edit…") { editing = host }
        }
        // Fills, so the failure line above it stays at the TOP of the pane:
        // `failureBanner` is a VStack, and a view with an intrinsic height
        // lets the whole stack centre itself.
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ToolbarContentBuilder private var toolbar: some ToolbarContent {
        ToolbarItem {
            Button { refresh() } label: {
                Label("Refresh", systemImage: "arrow.clockwise")
            }
            .disabled(selected == nil)
        }
    }

    private func refresh() {
        guard let host = selected else { return }
        Task {
            await hosts.refresh(host)
            await machines.refresh(host.id)
            countsLoaded = false
            await loadCounts()
        }
    }

    /// "Work here" and "Models here" read the stores the Queue and Models
    /// panes fill. Landing here FIRST -- a cold launch on ⌘5 -- would
    /// otherwise say "Nothing queued · None installed" about a machine
    /// holding eighty models, which is a false answer rather than a missing
    /// one. Once per pane rather than per selection, because both listings
    /// are fleet-wide and answer for every machine at once.
    private func loadCounts() async {
        guard !countsLoaded else { return }
        countsLoaded = true
        await queue.refresh()
        await models.refresh()
    }

    /// The GPUs, or the difference between a machine that does not report
    /// them and one that has none. Neither is an error and neither is a
    /// missing section.
    @ViewBuilder private func gpus(_ host: MoldHost) -> some View {
        Section("GPUs") {
            if hosts.capabilities(of: host)?.canSeeDevices != true {
                Text("This machine doesn't report its GPUs.")
                    .foregroundStyle(.secondary)
            } else if machines.devices(on: host.id).isEmpty {
                Text("No GPUs. This machine renders on the CPU.")
                    .foregroundStyle(.secondary)
            } else {
                ForEach(machines.devices(on: host.id)) { device in
                    DeviceRow(device: device, host: host,
                              sample: machines.sample(for: device, on: host.id))
                }
            }
        }
    }
}
