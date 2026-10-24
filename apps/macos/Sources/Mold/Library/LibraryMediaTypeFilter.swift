import MoldClient
import SwiftUI

/// The media-type title Source Media's filter shows for a Library query.
/// The Library itself now uses `LibraryFilterMenu`.
enum LibraryMediaTypeFilter {
    static func title(for query: LibraryQuery) -> String {
        LibraryMediaFilter.selected(in: query)?.title ?? "Multiple Media Types"
    }
}

struct SourceMediaTypeFilter: View {
    @Binding var filter: SourceMediaFilter
    var title: String? = nil
    var body: some View {
        Menu {
            Picker("Media Type", selection: $filter) {
                ForEach(SourceMediaFilter.allCases) { choice in Text(choice.title).tag(choice) }
            }.pickerStyle(.inline)
        } label: {
            LibraryFilterLabel(title: title ?? filter.title, systemImage: "photo.on.rectangle")
        }
        .accessibilityLabel("Media Type: \(title ?? filter.title)")
        .help("Filter source media by type")
    }
}
