import MoldClient
import SwiftUI

/// The fleet (DESIGN.md §5.4): one card per machine, then what is on this
/// network and not in the list yet. With no machine at all, the first-run
/// explanation and the two ways in.
struct MachinesView: View {
    @Environment(HostStore.self) private var hosts
    @Environment(NearbyBrowser.self) private var nearby
    @Environment(AppRouter.self) private var router
    @ScaledMetric(relativeTo: .body) private var cardWidth = 320
    @State private var removing: MoldHost?
    @State private var editing: MoldHost?

    var body: some View {
        @Bindable var router = router
        Group {
            if hosts.hosts.isEmpty {
                EmptyState(title: String(localized: "No machines yet"), symbol: Destination.machines.symbol,
                           message: String(localized: "Mold makes pictures on a computer you own. Add one to begin.")) {
                    VStack(spacing: 12) {
                        Button("Add a Machine…") { router.showsAddMachine = true }
                            .prominentAction()
                        if !nearby.machines.isEmpty {
                            Text("\(nearby.machines.count) found on this network")
                                .foregroundStyle(.secondaryText)
                        }
                    }
                }
            } else {
                fleet
            }
        }
        .navigationDestination(for: MoldHost.ID.self) { MachineDetailView(id: $0) }
        .navigationDestination(for: ModelsRoute.self) { route in
            ModelsView(fixedHost: route.host).navigationTitle("Models")
        }
        .navigationDestination(for: QueueRoute.self) { _ in QueueView().navigationTitle("Queue") }
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button { router.showsAddMachine = true } label: {
                    Label("Add a Machine", systemImage: "plus")
                }
            }
        }
        .sheet(isPresented: $router.showsAddMachine) { AddMachineSheet() }
        .sheet(item: $editing) { EditMachineSheet(host: $0) }
        .confirmationDialog(removing.map { "Remove \($0.name)?" } ?? "",
                            isPresented: Binding(get: { removing != nil }, set: { if !$0 { removing = nil } }),
                            titleVisibility: .visible) {
            Button("Remove", role: .destructive) { removing.map { hosts.remove($0.id) } }
        } message: {
            Text("Its key is removed from this iPhone too. Its prints stay on the machine.")
        }
        .onAppear { nearby.start() }
        .refreshable { await hosts.refreshAll() }
    }

    private var fleet: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                FailureBanner()
                if let preferred = hosts.preferredHost {
                    // On iPhone Models has no tab: it belongs to a machine.
                    NavigationLink(value: ModelsRoute(host: preferred.id)) {
                        Label {
                            AdaptiveRow { Text("Models") } value: { Text(preferred.name) }
                        } icon: { Image(systemName: Destination.models.symbol) }
                        .padding(14)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(.background.secondary, in: .rect(cornerRadius: 8))
                    }
                    .buttonStyle(.plain)
                    .padding(.horizontal, 16)
                }
                LazyVGrid(columns: [GridItem(.adaptive(minimum: min(cardWidth, 600)), spacing: 16)], spacing: 16) {
                    ForEach(hosts.hosts) { host in
                        NavigationLink(value: host.id) { MachineCard(host: host) }
                            .buttonStyle(.plain)
                            .contextMenu { menu(for: host) }
                    }
                }
                .padding(.horizontal, 16)
                NearbySection()
            }
            .padding(.vertical, 8)
        }
    }

    /// The Mac card menu's items, in its order; destructive last.
    @ViewBuilder private func menu(for host: MoldHost) -> some View {
        Button { Task { await hosts.refresh(host) } } label: { Label("Check Now", systemImage: "arrow.clockwise") }
        Button { hosts.makeDefault(host.id) } label: { Label("Set as Default", systemImage: "star") }
            .disabled(hosts.defaultMachine == host.id)
        Button { UIPasteboard.general.string = HostAddress.displayString(for: host.baseURL) } label: {
            Label("Copy Address", systemImage: "doc.on.doc")
        }
        Button { editing = host } label: { Label("Edit…", systemImage: "pencil") }
        Divider()
        Button(role: .destructive) { removing = host } label: { Label("Remove…", systemImage: "trash") }
    }
}

/// Machines this network advertises that are not in the list yet.
struct NearbySection: View {
    @Environment(HostStore.self) private var hosts
    @Environment(NearbyBrowser.self) private var nearby

    var body: some View {
        let fresh = nearby.machines.filter { !hosts.knows($0) }
        if !fresh.isEmpty || nearby.problem != nil {
            VStack(alignment: .leading, spacing: 8) {
                Text("Nearby").font(.headline).padding(.horizontal, 16)
                if let problem = nearby.problem {
                    Text(problem).foregroundStyle(.secondaryText).padding(.horizontal, 16)
                }
                ForEach(fresh) { machine in NearbyRow(machine: machine) }
            }
        }
    }
}

/// Machines ▸ a machine ▸ Models (or Queue), as a navigation value.
struct ModelsRoute: Hashable { let host: MoldHost.ID }
struct QueueRoute: Hashable { let host: MoldHost.ID }
