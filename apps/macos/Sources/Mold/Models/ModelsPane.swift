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
    @Environment(HostStore.self) private var hosts
    @Environment(ModelStore.self) private var models
    @Environment(DownloadStore.self) private var downloads

    @State private var hostID: MoldHost.ID?
    @State private var query = ""
    @State private var installedOnly = true

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
        .navigationTitle("Models")
        .navigationSubtitle(subtitle)
        .searchable(text: $query, prompt: "Search models")
        .toolbar { toolbar }
        .task { await load() }
    }

    // MARK: - Grouping

    /// Deliberately not named `Group`: that shadows SwiftUI's own view inside
    /// `body`, and the resulting errors point everywhere but here.
    private struct VariantGroup {
        let title: String
        let repo: String?
        let variants: [Model]

        /// A model with one untagged variant is not a group of anything.
        /// Giving it a heading plus a row leaves the row with nothing to say,
        /// which reads as a rendering bug rather than as a simple model.
        var isSolo: Bool { variants.count == 1 && variants[0].tag == nil }
    }

    private var host: MoldHost? {
        hosts.hosts.first { $0.id == hostID } ?? hosts.preferredHost
    }

    private var candidates: [Model] {
        guard let host else { return [] }
        let all = installedOnly ? models.ready(on: host.id) : models.generators(on: host.id)
        guard !query.isEmpty else { return all }
        let needle = query.lowercased()
        return all.filter {
            $0.description.lowercased().contains(needle) || $0.name.lowercased().contains(needle)
        }
    }

    private var groups: [VariantGroup] {
        Dictionary(grouping: candidates, by: \.baseName)
            .map { _, variants in
                let sorted = variants.sorted { ($0.sizeGb ?? 0) < ($1.sizeGb ?? 0) }
                let lead = sorted[0]
                // The trade-off sentence describes the VARIANT, so it stays
                // on the row. Repeating it in the heading said the same thing
                // twice for every single-variant model.
                return VariantGroup(title: lead.baseTitle, repo: lead.hfRepo, variants: sorted)
            }
            .sorted { $0.title.localizedStandardCompare($1.title) == .orderedAscending }
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
            Picker("Machine", selection: $hostID) {
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
        await hosts.refreshAll()
        await models.refresh()
        if hostID == nil { hostID = hosts.preferredHost?.id }
    }
}
