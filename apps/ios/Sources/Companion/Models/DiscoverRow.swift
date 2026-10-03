import MoldClient
import SwiftUI

struct DiscoverRow: View {
    @Environment(ModelStore.self) private var models
    @Environment(HostStore.self) private var hosts
    @Environment(\.dynamicTypeSize) private var size
    let entry: CatalogEntry
    let host: MoldHost

    var body: some View {
        let stacked = RowAxis.for(size) == .vertical
        let layout = stacked ? AnyLayout(VStackLayout(alignment: .leading, spacing: 8))
                             : AnyLayout(HStackLayout(alignment: .center, spacing: 12))
        layout {
            VStack(alignment: .leading, spacing: 2) {
                Text(entry.name)
                ModelSourceLabel(source: entry.source)
                if let author = entry.author { Text("by \(author)").font(.callout).foregroundStyle(.secondaryText) }
                Text(verbatim: detail).font(.caption.monospaced()).foregroundStyle(.secondaryText)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            action.frame(maxWidth: stacked ? .infinity : nil)
        }
        .padding(.vertical, 2)
    }

    private var detail: String {
        var parts = [entry.family]
        if let bytes = entry.sizeBytes { parts.append(ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)) }
        parts.append(String(localized: "\(entry.downloadCount.formatted(.number.notation(.compactName))) downloads"))
        return parts.joined(separator: " · ")
    }

    @ViewBuilder private var action: some View {
        if let (job, row) = models.progress(for: entry.id, on: host.id) {
            VStack(alignment: .trailing, spacing: 4) {
                if let fraction = row.fraction { ProgressView(value: fraction).frame(minWidth: 80) }
                Button("Cancel Download", role: .destructive) { Task { await models.cancel(job: job, on: host.id) } }
                    .buttonStyle(.bordered)
            }
        } else if entry.installed {
            Text("Installed").foregroundStyle(.secondaryText)
        } else if entry.supported {
            Button("Get") { Task { await models.install(entry.id, on: host.id) } }
                .buttonStyle(.bordered)
                .accessibilityLabel(String(localized: "Get \(entry.name)"))
        } else if let page = entry.pageUrl.flatMap(URL.init(string:)) {
            Link("Open Page", destination: page).buttonStyle(.bordered)
        }
    }
}
