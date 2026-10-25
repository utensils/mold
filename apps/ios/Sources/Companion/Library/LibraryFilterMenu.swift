import MoldClient
import MoldStyle
import SwiftUI

/// The Library's Filter button, after Photos: one checkmarked pick, a Model
/// Family submenu with each family's colour dot, the Machine choice, and
/// Clear Filters. The words, groups and effects are MoldClient's
/// `LibraryFilter` and `LibraryFamilies`, shared with the Mac.
struct LibraryFilterMenu: View {
    @Binding var query: LibraryQuery
    let families: [LibraryFamilyFacet]

    var body: some View {
        let title = LibraryFilter.summary(for: query)
        Menu {
            ForEach(Array(LibraryFilter.sections.enumerated()), id: \.offset) { index, section in
                Section {
                    ForEach(section) { filter in
                        Toggle(isOn: Binding(get: { LibraryFilter.selected(in: query) == filter },
                                             set: { _ in query = filter.applying(to: query) })) {
                            Label(filter.title, systemImage: filter.symbol)
                        }
                    }
                }
            }
            Section {
                if !families.isEmpty {
                    Menu {
                        let chosen = LibraryFamilies.selected(in: query)
                        ForEach(families) { family in
                            Toggle(isOn: Binding(get: { chosen.contains(family.slug) },
                                                 set: { _ in query = LibraryFamilies.toggling(family.slug, label: family.label, in: query) })) {
                                Label {
                                    Text(family.label)
                                } icon: {
                                    if let color = family.color { Image(uiImage: FamilySwatch.menuImage(rgb: color.rgb)) }
                                }
                            }
                        }
                        Button("All Families") { query = LibraryFamilies.clearing(query) }
                            .disabled(chosen.isEmpty)
                    } label: {
                        Label("Model Family", systemImage: "circle.hexagongrid")
                    }
                }
                LibraryMachinePicker().pickerStyle(.menu)
            }
            Section {
                Button("Clear Filters", systemImage: "xmark.circle") { query = LibraryFilter.clearing(query) }
                    .disabled(!LibraryFilter.isActive(query))
            }
        } label: {
            Label("Filter", systemImage: LibraryFilter.isActive(query)
                ? "line.3.horizontal.decrease.circle.fill" : "line.3.horizontal.decrease.circle")
        }
        .accessibilityLabel(String(localized: "Filter"))
        .accessibilityValue(title)
        .accessibilityIdentifier("library-filter")
    }
}

/// The phone title menu's media types, beside the shelves (the Filter
/// button carries the same picks with the rest of the menu).
struct LibraryMediaPicker: View {
    @Binding var query: LibraryQuery
    var body: some View {
        Picker("Media Type", selection: Binding(get: {
            LibraryMediaFilter.selected(in: query)
        }, set: { if let filter = $0 { query = filter.applying(to: query) } })) {
            ForEach(LibraryMediaFilter.allCases) { filter in Text(filter.title).tag(Optional(filter)) }
        }
    }
}
