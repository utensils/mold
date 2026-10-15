import MoldClient
import SwiftUI

struct LibraryMediaTypeFilter: View {
    @Binding var query: LibraryQuery
    let didChange: () -> Void

    var body: some View {
        let title = Self.title(for: query)
        Menu {
            Picker("Media Type", selection: Binding(get: {
                LibraryMediaFilter.selected(in: query)
            }, set: { filter in
                guard let filter else { return }
                query = filter.applying(to: query)
                didChange()
            })) {
                ForEach(LibraryMediaFilter.allCases) { filter in
                    Text(filter.title).tag(Optional(filter))
                }
            }
            .pickerStyle(.inline)
        } label: {
            LibraryFilterLabel(title: title, systemImage: "photo.on.rectangle")
        }
        .accessibilityLabel("Media Type: \(title)")
        .help("Show all media, photos, videos or 3D in this shelf (\(title))")
    }

    static func title(for query: LibraryQuery) -> String {
        LibraryMediaFilter.selected(in: query)?.title ?? "Multiple Media Types"
    }
}
