import MoldClient
import MoldStyle
import SwiftUI

/// The Library's Filter menu, after Photos: one pick (All Items, Favourites,
/// Photos, Not in a Collection, Portrait, ...) checked at a time, a Model
/// Family submenu that combines with it, and Clear Filters. The words, the
/// groups and what a choice does are MoldClient's `LibraryFilter` and
/// `LibraryFamilies`; this only draws them. Machines keep their own toolbar
/// menu, beside it.
struct LibraryFilterMenu: View {
    @Binding var query: LibraryQuery
    /// The families the Library holds (`LibraryFamilyFacetCache`).
    let families: [LibraryFamilyFacet]
    let didChange: () -> Void

    var body: some View {
        let title = LibraryFilter.summary(for: query)
        Menu {
            ForEach(Array(LibraryFilter.sections.enumerated()), id: \.offset) { index, section in
                if index > 0 { Divider() }
                ForEach(section) { filter in
                    Toggle(isOn: Binding(get: { LibraryFilter.selected(in: query) == filter },
                                         set: { _ in apply(filter.applying(to: query)) })) {
                        Label(filter.title, systemImage: filter.symbol)
                    }
                }
            }
            if !families.isEmpty {
                Divider()
                Menu("Model Family") {
                    let chosen = LibraryFamilies.selected(in: query)
                    ForEach(families) { family in
                        Toggle(isOn: Binding(get: { chosen.contains(family.slug) },
                                             set: { _ in apply(LibraryFamilies.toggling(family.slug, label: family.label, in: query)) })) {
                            Label {
                                Text(family.label)
                            } icon: {
                                if let color = family.color { Image(nsImage: FamilySwatch.menuImage(rgb: color.rgb)) }
                            }
                        }
                    }
                    Divider()
                    Button("All Families") { apply(LibraryFamilies.clearing(query)) }
                        .disabled(chosen.isEmpty)
                }
            }
            Divider()
            Button("Clear Filters") { apply(LibraryFilter.clearing(query)) }
                .disabled(!LibraryFilter.isActive(query))
        } label: {
            LibraryFilterLabel(title: title, systemImage: LibraryFilter.isActive(query)
                ? "line.3.horizontal.decrease.circle.fill" : "line.3.horizontal.decrease.circle")
        }
        .accessibilityLabel("Filter: \(title)")
        .help("Show only some prints: favourites, a media type, prints in or not in a collection, made from a source, tagged or untagged, an orientation or a model family (\(title))")
    }

    private func apply(_ next: LibraryQuery) {
        query = next
        didChange()
    }
}
