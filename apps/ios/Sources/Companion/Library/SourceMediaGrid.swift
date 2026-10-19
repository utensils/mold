import MoldClient
import SwiftUI

/// The Library's tight, uncropped rows with source identities and oriented ratios.
struct SourceMediaGrid: View {
    @Environment(HostStore.self) private var hosts
    let sources: [SourceMediaDisplay]
    let edge: CGFloat
    let openOwner: (PrintID) -> Void
    let scrollTo: (String) -> Void
    @State private var width: CGFloat = 0
    @State private var layout = JustifiedSourceLayout()
    @State private var aspects = SourceMediaAspects()
    @State private var visible: Set<String> = []

    var body: some View {
        let rows = layout.resolve(sources.map {
            .init(id: $0.id, aspect: aspects.aspect(for: $0.id, key: key(for: $0)))
        }, width: width, targetHeight: edge)
        LazyVStack(alignment: .leading, spacing: JustifiedLayout.gap) {
            ForEach(rows) { row in
                HStack(spacing: JustifiedLayout.gap) {
                    ForEach(row.items) { item in
                        let source = sources[item.index]
                        SourceMediaTile(source: source, openOwner: openOwner) { key, ratio in
                            aspects.record(id: source.id, key: key, aspect: ratio)
                        }
                        .frame(width: item.width, height: row.height)
                    }
                }
                .frame(height: row.height)
                .id("source-row-" + row.id)
                .onScrollVisibilityChange(threshold: 0.1) { shown in
                    if shown { visible.insert(row.id) } else { visible.remove(row.id) }
                }
                .onDisappear { visible.remove(row.id) }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .onGeometryChange(for: CGFloat.self) { $0.size.width } action: { next in width = next }
        .onChange(of: width) { _, _ in preserveAnchor(in: rows) }
        .onChange(of: edge) { _, _ in preserveAnchor(in: rows) }
        .onChange(of: aspects.revision) { _, _ in preserveAnchor(in: rows) }
    }
    private func key(for source: SourceMediaDisplay) -> SourcePreviewKey {
        let row = source.preferredUsage { hosts.host($0).map(hosts.isUp) ?? false }!
        return .init(authority: .init(owner: row.owner, route: hosts.host(row.owner.hostID), instance: hosts.instanceID(of: row.owner.hostID)), member: row.member)
    }
    private func preserveAnchor(in rows: [JustifiedSourceLayout.Row]) {
        guard let anchor = sources.first(where: { visible.contains($0.id) })?.id,
              let row = rows.first(where: { $0.items.contains(where: { $0.id == anchor }) }) else { return }
        Task { @MainActor in
            await Task.yield()
            scrollTo("source-row-" + row.id)
        }
    }
}
