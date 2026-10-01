import MoldClient
import SwiftUI

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
