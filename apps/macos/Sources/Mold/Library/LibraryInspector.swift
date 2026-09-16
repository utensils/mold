import MoldClient
import MoldStyle
import SwiftUI

/// What the selection is made of, and what you can do with it.
///
/// Ordered by how often it is touched, not by how the wire format is shaped:
/// the name and the filing at the top because those are what people change,
/// provenance below in a disclosure because it is what people read, and the
/// actions last. A print's own numbers are all selectable, because the point
/// of showing a seed is that somebody copies it.
struct LibraryInspector: View {
    let entries: [LibraryEntry]
    let host: MoldHost?
    let scope: LibraryScope
    let actions: LibraryActions
    let filterByTag: (String) -> Void

    @AppStorage("inspectorShowsProvenance", store: AppStorageSuite.defaults)
    private var showsProvenance = true

    var body: some View {
        Group {
            if entries.isEmpty {
                ContentUnavailableView("Nothing selected", systemImage: "sidebar.right")
            } else {
                ScrollView { content.padding(16) }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder private var content: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            if scope.isTrash {
                TrashCountdownBlock(entries: entries)
            } else {
                organize
            }
            if entries.count == 1, let entry = entries.first {
                DisclosureGroup("Provenance", isExpanded: $showsProvenance) {
                    ProvenanceGrid(entry: entry)
                        .padding(.top, 6)
                }
                .font(.callout)
            }
            InspectorActions(entries: entries, scope: scope, actions: actions)
        }
    }

    /// The picture, then what it is called.
    @ViewBuilder private var header: some View {
        if entries.count == 1, let entry = entries.first {
            if let host {
                LibraryThumbnail(entry: entry, host: host, edge: 320)
                    .frame(maxWidth: .infinity)
            }
            if !scope.isTrash {
                TitleField(entry: entry, actions: actions)
            }
        } else {
            VStack(spacing: 6) {
                Image(systemName: "square.stack")
                    .font(.largeTitle)
                    .foregroundStyle(.tertiary)
                Text("\(entries.count) prints selected").font(.headline)
                if let span = machines {
                    Text(span).font(.caption).foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity)
        }
    }

    @ViewBuilder private var organize: some View {
        LabeledSection("Tags") {
            TagEditor(entries: entries, actions: actions, filterBy: filterByTag)
        }
        LabeledSection("Collections") {
            CollectionsField(entries: entries, actions: actions)
        }
    }

    /// Says when a selection spans machines, because the actions below will
    /// then touch more than one.
    private var machines: String? {
        let names = Set(entries.map(\.hostName)).sorted()
        return names.count > 1 ? "On \(names.joined(separator: ", "))" : names.first
    }
}

/// A heading and the thing it names. Small enough to be a shape rather than a
/// component, but every section wants the same one.
struct LabeledSection<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content

    init(_ title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            content
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
