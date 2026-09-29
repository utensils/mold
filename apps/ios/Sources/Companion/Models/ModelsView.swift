import MoldClient
import SwiftUI

/// Models, per machine (DESIGN.md §5.5): what it has installed, and what it
/// could fetch. On iPad a sidebar destination (following the chosen
/// machine); on iPhone reached from Machines.
struct ModelsView: View {
    enum Pane: String, CaseIterable, Identifiable {
        case installed, discover
        var id: Self { self }
        var title: String {
            switch self {
            case .installed: String(localized: "Installed")
            case .discover: String(localized: "Discover")
            }
        }
    }

    @Environment(HostStore.self) private var hosts
    @Environment(ModelStore.self) private var models
    @Environment(AppRouter.self) private var router
    /// Fixed when opened from one machine's detail; else the Default.
    var fixedHost: MoldHost.ID?
    @State private var chosen: MoldHost.ID?
    @SceneStorage("models.pane") private var pane: Pane = .installed

    private var hostID: MoldHost.ID? { fixedHost ?? chosen ?? hosts.preferredHost?.id }

    var body: some View {
        Group {
            if let id = hostID, let host = hosts.host(id) {
                content(host)
            } else {
                EmptyState(title: String(localized: "No machine to show"), symbol: Destination.models.symbol,
                           message: String(localized: "Models belong to a machine. Add one to see what it has installed.")) {
                    Button("Add a Machine…") { router.addMachine() }.prominentAction()
                }
            }
        }
        .sheet(item: Binding(get: { models.pendingLicense }, set: { models.pendingLicense = $0 })) { pending in
            LicenceSheet(pending: pending)
        }
    }

    @ViewBuilder private func content(_ host: MoldHost) -> some View {
        Group {
            switch pane {
            case .installed: InstalledModels(host: host)
            case .discover: DiscoverModels(host: host)
            }
        }
        .safeAreaBar(edge: .top) {
            Picker("Show models", selection: $pane) {
                ForEach(Pane.allCases) { Text($0.title).tag($0) }
            }
            .pickerStyle(.menu)
            .accessibilityIdentifier("models-pane")
            .padding(.horizontal, 16)
            .padding(.bottom, 8)
        }
        .toolbar {
            if fixedHost == nil, hosts.hosts.count > 1 {
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        Picker("Machine", selection: Binding(get: { host.id }, set: { chosen = $0 })) {
                            ForEach(hosts.hosts) { Text($0.name).tag($0.id) }
                        }
                    } label: {
                        Label(host.name, systemImage: "server.rack").labelStyle(.titleAndIcon)
                    }
                }
            }
        }
        .navigationSubtitle(fixedHost == nil ? host.name : "")
        .task(id: host.id) { await models.refresh(on: host.id) }
    }
}
