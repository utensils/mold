import MoldClient
import SwiftUI

/// What each machine can render with, written so a person can tell what a
/// model is FOR.
///
/// mold's manifest already describes every model as "Title — plain-English
/// trade-off". This groups by model and lets each variant row carry that
/// sentence, instead of listing 170 rows of quantization tags and leaving the
/// reader to work out the difference between q4 and bf16.
struct ModelsPane: View {
    // Not `private`: `ModelsPane+Grouping` reads all of these, and `private`
    // does not cross a file boundary even within one type.
    @Environment(HostStore.self) var hosts
    @Environment(ModelStore.self) var models
    @Environment(DownloadStore.self) private var downloads

    /// The same key the sidebar and `MachinesPane` declare, over the same
    /// suite. "Models here -- Show ›" then lands on the right machine with
    /// zero plumbing, and the app has one notion of "the machine you are
    /// working on" instead of two -- picking a machine here is the same act
    /// as picking it in the sidebar, and it persists across launches.
    @AppStorage("selectedMachine", store: AppStorageSuite.defaults) var selectedMachine = ""
    @State var query = ""
    @State var installedOnly = true

    var body: some View {
        Group {
            if groups.isEmpty {
                empty
            } else {
                List {
                    ForEach(groups, id: \.title) { group in
                        if group.isSolo {
                            ModelSoloRow(model: group.variants[0], title: group.title,
                                         install: install,
                                         progress: progress(group.variants[0]))
                        } else {
                            Section {
                                ForEach(group.variants) { variant in
                                    ModelVariantRow(model: variant, groupTitle: group.title,
                                                    install: install,
                                                    progress: progress(variant))
                                }
                            } header: {
                                ModelGroupHeader(title: group.title, repo: group.repo)
                            }
                        }
                    }
                }
                .listStyle(.inset)
            }
        }
        .failureBanner(hosts)
        .navigationTitle("Models")
        .navigationSubtitle(subtitle)
        .searchable(text: $query, prompt: "Search models")
        .toolbar { toolbar }
        .task { await load() }
        .onChange(of: hosts.reachability) { _, _ in adoptPreferredHost() }
    }

    private var subtitle: String {
        guard let host else { return "No machine" }
        let count = candidates.count
        return "\(count) \(installedOnly ? "installed" : "available") on \(host.name)"
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
        ToolbarItem {
            Toggle(isOn: $installedOnly) {
                Label("Installed only", systemImage: "internaldrive")
            }
            .help(installedOnly ? "Showing installed models" : "Showing everything on offer")
        }
    }

    private func progress(_ model: Model) -> DownloadStore.Progress? {
        guard let host else { return nil }
        return downloads.progress(for: model.name, on: host.id)
    }

    private func install(_ model: Model) {
        guard let host else { return }
        Task {
            await downloads.install(model, on: host)
        }
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
