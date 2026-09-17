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

    /// The same key the sidebar and `MachinesPane` declare, over the same
    /// suite -- one notion of "the machine you are working on" rather than
    /// two, and it persists across launches.
    @AppStorage("selectedMachine", store: AppStorageSuite.defaults) var selectedMachine = ""
    @AppStorage("modelsSortColumn", store: AppStorageSuite.defaults) var sortColumnRaw =
        ModelSort.Column.model.rawValue
    @AppStorage("modelsSortAscending", store: AppStorageSuite.defaults) var sortAscending = true
    @State var query = ""
    @State var selection: Model.ID?

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

    var body: some View {
        VStack(spacing: 0) {
            content
            ModelsFooter(count: candidates.count, host: host, status: status)
        }
        .failureBanner(hosts)
        .navigationTitle("Models")
        .navigationSubtitle(subtitle)
        .searchable(text: $query, prompt: "Search models")
        .toolbar { toolbar }
        .task { await load() }
        .onChange(of: hosts.reachability) { _, _ in adoptPreferredHost() }
    }

    @ViewBuilder private var content: some View {
        if sections.isEmpty {
            empty
        } else {
            table
        }
    }

    private var subtitle: String {
        guard let host else { return "No machine" }
        return "\(candidates.count) installed on \(host.name)"
    }

    @ViewBuilder private var empty: some View {
        if models.isLoading {
            ProgressView("Asking each machine what it has…")
        } else {
            ContentUnavailableView("No models", systemImage: "cube",
                                   description: Text("Nothing matches on this machine."))
        }
    }

    @ToolbarContentBuilder private var toolbar: some ToolbarContent {
        ToolbarItem {
            Picker("Machine", selection: selectedHostID) {
                ForEach(hosts.hosts) { host in
                    Text(host.name).tag(MoldHost.ID?.some(host.id))
                }
            }
        }
        // The Discover scope lands in S6 alongside the catalog browser it
        // has something to show; a one-segment picker in the meantime would
        // be a control with nothing to switch.
    }

    func progress(_ model: Model) -> DownloadStore.Progress? {
        guard let host else { return nil }
        return downloads.progress(for: model.name, on: host.id)
    }

    func install(_ model: Model) {
        guard let host else { return }
        Task { await downloads.install(model.name, on: host) }
    }

    /// The Cancel action for a row mid-download, or `nil` off it. The job id
    /// comes straight from `DownloadStore.active`'s own keys -- `progress`
    /// alone does not carry it, and this is the one place both the id and
    /// its progress are read from the same dictionary together.
    func cancel(_ model: Model) -> (() -> Void)? {
        guard let host,
              let job = downloads.active[host.id]?.first(where: { $0.value.model == model.name })
        else { return nil }
        return { Task { await downloads.cancel(jobID: job.key, on: host) } }
    }

    private func load() async {
        await models.refresh()
        adoptPreferredHost()
    }

    /// `selectedMachine` read and written the way `HostStore.machine(selected:)`
    /// expects: a `Binding<MoldHost.ID?>` over the stored `uuidString`, shared
    /// with the sidebar and `MachinesPane` rather than a picker of its own.
    private var selectedHostID: Binding<MoldHost.ID?> {
        Binding(
            get: { hosts.machine(selected: selectedMachine)?.id },
            set: { selectedMachine = $0?.uuidString ?? "" }
        )
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
