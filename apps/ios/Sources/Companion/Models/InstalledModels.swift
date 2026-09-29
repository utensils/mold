import MoldClient
import SwiftUI

/// Installed, grouped by family; what is fetching at the top; the machine's
/// own disk figure in the footer.
struct InstalledModels: View {
    @Environment(HostStore.self) private var hosts
    @Environment(ModelStore.self) private var models
    let host: MoldHost
    @State private var deleting: Model?
    @State private var inspecting: Model?

    var body: some View {
        List {
            FailureBanner().listRowInsets(EdgeInsets()).listRowBackground(Color.clear)
            if let summary = models.summary {
                Text(summary).foregroundStyle(.secondaryText)
                    .task { try? await Task.sleep(for: .seconds(8)); models.summary = nil }
            }
            let fetching = (models.active[host.id] ?? [:]).sorted { $0.value.model < $1.value.model }
            if !fetching.isEmpty {
                Section {
                    ForEach(fetching, id: \.key) { job, row in
                        DownloadRow(row: row, rate: models.rate(of: job)) {
                            Task { await models.cancel(job: job, on: host.id) }
                        }
                    }
                } header: { SectionHeader(String(localized: "Downloading")) }
            }
            ForEach(models.installed(on: host.id), id: \.family) { group in
                Section {
                    ForEach(group.models) { model in
                        InstalledRow(model: model)
                            .swipeActions { Button("Delete", systemImage: "trash", role: .destructive) { deleting = model } }
                            .contextMenu { menu(model) }
                    }
                } header: { SectionHeader(group.family) }
            }
            if models.installed(on: host.id).isEmpty, fetching.isEmpty {
                Text(models.emptyInventoryMessage(on: host))
                    .foregroundStyle(.secondaryText)
            }
            if case let .up(status) = hosts.reachability(of: host), let disk = status.modelsDisk {
                Section {} footer: {
                    Text("Models use \(ByteCountFormatter.string(fromByteCount: Int64(disk.totalBytes), countStyle: .file)) · \(ByteCountFormatter.string(fromByteCount: Int64(disk.freeBytes), countStyle: .file)) free")
                        .foregroundStyle(.secondaryText)
                }
            }
        }
        .refreshable { await hosts.refresh(host); await models.refresh(on: host.id) }
        .confirmationDialog(deleting.map { String(localized: "Delete \($0.headline)?") } ?? "",
                            isPresented: Binding(get: { deleting != nil }, set: { if !$0 { deleting = nil } }),
                            titleVisibility: .visible) {
            Button("Delete", role: .destructive) {
                if let model = deleting { Task { await models.delete(model, on: host.id) } }
            }
        } message: {
            Text("Files another installed model still uses are kept.")
        }
        .sheet(item: $inspecting) { ComponentsSheet(model: $0, host: host) }
    }

    /// The Mac's model menu, in its order; Delete last.
    @ViewBuilder private func menu(_ model: Model) -> some View {
        if model.isLoaded == true {
            Button("Unload", systemImage: "eject") { Task { await models.unload(model, on: host.id) } }
        } else {
            Button("Load", systemImage: "memorychip") { Task { await models.load(model, on: host.id) } }
        }
        if case .needsRepair = model.installState {
            Button("Repair", systemImage: "wrench.and.screwdriver") { Task { await models.install(model.name, on: host.id) } }
        }
        Button("Components", systemImage: "shippingbox") { inspecting = model }
        Divider()
        Button("Delete…", systemImage: "trash", role: .destructive) { deleting = model }
    }
}

/// Name over mono id, with the state and size -- stacked at large sizes.
private struct InstalledRow: View {
    let model: Model

    var body: some View {
        AdaptiveRow {
            VStack(alignment: .leading, spacing: 2) {
                Text(model.headline)
                Text(verbatim: model.name).font(.caption.monospaced()).foregroundStyle(.secondaryText)
                if let reason = state { Text(reason).font(.caption).foregroundStyle(.secondaryText) }
            }
        } value: {
            if let bytes = model.diskUsageBytes {
                Text(ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)).monospacedDigit()
            }
        }
        .accessibilityElement(children: .combine)
    }

    private var state: String? {
        switch model.installState {
        case .loaded: String(localized: "Loaded")
        case .needsRepair: String(localized: "Some files are missing — Repair fetches them")
        default: model.runtimeAvailable == false ? model.runtimeUnavailableReason : nil
        }
    }
}

/// "2.1 / 11.8 GB · 42 MB/s" with its bar and Cancel Download.
struct DownloadRow: View {
    @Environment(\.dynamicTypeSize) private var size
    let row: DownloadProgress
    let rate: Double?
    let cancel: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(verbatim: row.model).font(.body.monospaced())
            if let failed = row.failed {
                Text(failed).foregroundStyle(.secondaryText)
            }
            if let fraction = row.fraction {
                ProgressView(value: fraction)
                    .accessibilityValue(String(localized: "\(Int(fraction * 100)) percent"))
            } else {
                ProgressView().frame(maxWidth: .infinity, alignment: .leading)
            }
            let layout = RowAxis.for(size) == .horizontal
                ? AnyLayout(HStackLayout(alignment: .firstTextBaseline)) : AnyLayout(VStackLayout(alignment: .leading, spacing: 6))
            layout {
                Text(verbatim: row.sentence(bytesPerSecond: rate)).font(.caption.monospacedDigit()).foregroundStyle(.secondaryText)
                if RowAxis.for(size) == .horizontal { Spacer(minLength: 8) }
                Button("Cancel Download", role: .destructive, action: cancel).buttonStyle(.bordered)
            }
        }
        .padding(.vertical, 2)
    }
}
