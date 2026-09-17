import MoldClient
import MoldStyle
import SwiftUI

/// "From Library…" (M8 design, decision 5): a searchable grid of the
/// fleet's pictures, newest first, host-badged when there is more than one
/// machine. Double-click or Use hands the caller its bytes, fetched from the
/// machine that holds it through the same route Quick Look uses.
///
/// `rows` and `caption` are pure (`LibraryPicker.rows`, `caption(count:)`),
/// so the grid's own filtering and footer text are tested with no view.
struct LibraryPickerSheet: View {
    /// Handed an `ImportedPicture`, already base64'd off the main actor
    /// (`PictureImport`) -- the wells hold what will be sent, not raw bytes
    /// they would have to encode on the main thread (finding 02#10).
    let pick: (ImportedPicture) -> Void

    @Environment(\.dismiss) private var dismiss
    @Environment(LibraryStore.self) private var library
    /// Not `private`: `LibraryPickerSheet+Grid`, an extension in another
    /// file, resolves each row's machine through it.
    @Environment(HostStore.self) var hosts
    /// Not `private`, same reason -- the empty state's wording depends on it.
    @State var query = ""
    /// Not `private`, same reason -- a tile sets and reads this.
    @State var selected: PrintID?
    @State private var isFetching = false

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Choose a Picture").font(.headline)
                Spacer()
                TextField("Search", text: $query)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 220)
                    .accessibilityLabel("Search pictures")
            }
            .padding(12)
            Divider()
            grid
            Divider()
            footer
        }
        .frame(width: 680, height: 520)
        .task {
            guard library.items.isEmpty, !library.isLoading else { return }
            await library.refresh()
        }
    }

    private var footer: some View {
        HStack {
            Text(LibraryPickerSheet.caption(count: rows.count))
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            Button("Cancel") { dismiss() }
                .keyboardShortcut(.cancelAction)
            Button("Use", action: use)
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.borderedProminent)
                .disabled(selected == nil || isFetching)
        }
        .padding(12)
    }

    /// Not `private`: `LibraryPickerSheet+Grid` draws one tile per row.
    var rows: [LibraryEntry] { LibraryPicker.rows(library.items, query: query) }

    /// Not `private`: a double-clicked tile calls this directly.
    func use() {
        guard let selected, let entry = rows.first(where: { $0.id == selected }) else { return }
        isFetching = true
        Task {
            defer { isFetching = false }
            do {
                pick(try await PictureSource.bytes(
                    of: .print(selected), hosts: hosts, library: library))
                dismiss()
            } catch {
                hosts.report(error, on: selected.host, doing: "fetch that picture")
            }
        }
    }
}
