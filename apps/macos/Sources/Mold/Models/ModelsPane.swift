import MoldClient
import SwiftUI

/// What a machine has installed, as a table someone managing eighty models
/// can sort and scan -- rather than a flat list they have to read end to end
/// to find the one row that needs attention.
struct ModelsPane: View {
    // Not `private`: `ModelsPane+Grouping` and `ModelsPane+Table` read all of
    // these, and `private` does not cross a file boundary even within one
    // type.
    @Environment(HostStore.self) var hosts
    @Environment(ModelStore.self) var models
    @Environment(DownloadStore.self) var downloads
    @Environment(LicenseStore.self) var licenses
    @Environment(CatalogStore.self) var catalog

    /// The same key the sidebar and `MachinesPane` declare, over the same
    /// suite -- one notion of "the machine you are working on" rather than
    /// two, and it persists across launches.
    @AppStorage("selectedMachine", store: AppStorageSuite.defaults) var selectedMachine = ""
    @AppStorage("modelsSortColumn", store: AppStorageSuite.defaults) var sortColumnRaw =
        ModelSort.Column.model.rawValue
    @AppStorage("modelsSortAscending", store: AppStorageSuite.defaults) var sortAscending = true
    @AppStorage("modelsScope", store: AppStorageSuite.defaults) var scopeRaw = ModelScope.installed.rawValue
    @State var query = ""
    @State var selection: Model.ID?
    // Not `private`: `ModelsPane+Actions` reads and writes these too, same
    // file-boundary reason as the doc comment above.
    @State var pendingDestruction: Destruction?
    @State var componentsModel: Model?
    @State var licenseInfo: ThirdPartyLicense?
    /// A removal's own one-line report -- there is no `HostFailure`-shaped
    /// funnel for a SUCCESS, so this is a transient caption under the table
    /// rather than a new store-wide mechanism for the one caller that needs
    /// it (design S5).
    @State var removalSummary: String?

    /// `ModelSort` itself is not `@AppStorage`-able -- it is not a primitive
    /// and not `RawRepresentable` -- so this reads and writes the two scalar
    /// keys that back it, the same shape `selectedHostID` already uses for
    /// `selectedMachine`.
    var sort: Binding<ModelSort> {
        Binding(
            get: { ModelSort(column: ModelSort.Column(rawValue: sortColumnRaw) ?? .model, ascending: sortAscending) },
            set: { sortColumnRaw = $0.column.rawValue; sortAscending = $0.ascending }
        )
    }

    /// Same shape as `sort`: a scalar `@AppStorage` key resolved through the
    /// pure fallback rule, so a stored `.discover` from a machine that could
    /// browse never strands the pane on one that cannot (design S3/S6).
    var scope: Binding<ModelScope> {
        Binding(
            get: { ModelScope.resolved(stored: ModelScope(rawValue: scopeRaw) ?? .installed, available: availableScopes) },
            set: { scopeRaw = $0.rawValue }
        )
    }

    var body: some View {
        // A local binding, the way `LibraryPane+Toolbar` reads `navigation`:
        // `downloads.pendingLicense` is `internal(set)`, not `@State` here.
        @Bindable var downloads = downloads
        VStack(spacing: 0) {
            content
            if let removalSummary {
                Text(removalSummary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 12)
                    .padding(.top, 4)
            }
            // Installed-only (design S3): Discover has its own result count
            // in the subtitle, and a footer built from the Installed list
            // would say "0 installed" while a Discover search is typed
            // (design S6b).
            if scope.wrappedValue == .installed {
                ModelsFooter(count: installedCount, host: host, status: status)
            }
        }
        .failureBanner(hosts)
        .navigationTitle("Models")
        .navigationSubtitle(subtitle)
        .searchable(text: $query, prompt: "Search models")
        .toolbar { toolbar }
        .sheet(item: $downloads.pendingLicense) { pending in
            LicenseSheet(pending: pending)
        }
        .sheet(item: $componentsModel) { model in
            if let host { ComponentsSheet(model: model, host: host) }
        }
        .sheet(item: $licenseInfo) { license in
            LicenseInfoSheet(license: license)
        }
        .destructionDialog($pendingDestruction)
        .focusedSceneValue(\.modelSelection, modelSelection)
        .task { await load() }
        .onChange(of: hosts.reachability) { _, _ in adoptPreferredHost() }
    }

    @ViewBuilder private var content: some View {
        switch scope.wrappedValue {
        case .installed:
            if sections.isEmpty { empty } else { table }
        case .discover:
            if let host { DiscoverTable(host: host, searchText: $query) }
        }
    }

    private var subtitle: String {
        Self.subtitle(scope: scope.wrappedValue, hostName: host?.name, installedCount: installedCount, discoverTotal: discoverTotal)
    }

    @ViewBuilder private var empty: some View {
        if models.isLoading {
            ProgressView("Asking each machine what it has…")
        } else {
            ContentUnavailableView("No models", systemImage: "cube",
                                   description: Text("Nothing matches on this machine."))
        }
    }

    func progress(_ model: Model) -> DownloadStore.Progress? {
        guard let host else { return nil }
        return downloads.progress(for: model.name, on: host.id)
    }

    private func load() async {
        await models.refresh()
        adoptPreferredHost()
    }

    /// Land on a machine once one has answered.
    ///
    /// `preferredHost` falls back to the first row configured, which before
    /// any answer is in may well be a machine that is off -- so this waits for
    /// an `up`. Until then `host` falls back the same way for display, so the
    /// pane still shows something; what it does not do is PIN the picker to a
    /// machine nobody chose. "Nobody chose" is asked directly against the
    /// stored id -- `machine(selected:)` itself already falls back to
    /// `preferredHost`, so asking IT would never see "nothing yet".
    private func adoptPreferredHost() {
        let chosen = UUID(uuidString: selectedMachine).flatMap(hosts.host)
        guard chosen == nil, hosts.hosts.contains(where: hosts.isUp) else { return }
        selectedMachine = hosts.preferredHost?.id.uuidString ?? ""
    }
}
