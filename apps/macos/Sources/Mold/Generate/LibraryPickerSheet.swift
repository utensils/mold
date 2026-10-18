import AppKit
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
    /// What the well that opened this sheet can read. A print is bytes mold
    /// made, which is not the same as bytes that well's path decodes -- the
    /// identity encoder reads PNG and JPEG and nothing else -- so the picture
    /// is conformed on the way out exactly as a file would be.
    var accepting: Set<String> = PictureImport.engineReadable
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
    @State var isFetching = false
    @State var machineID: MoldHost.ID?
    struct PreviewSelection: Identifiable {
        let entry: LibraryEntry
        let authority: InputMediaPreviewAuthority
        var id: PrintID { entry.id }
    }
    @State var preview: PreviewSelection?
    @State var fetchError: String?
    @State private var fetchTask: Task<Void, Never>?

    private var availableSize: CGSize {
        NSApp.keyWindow?.screen?.visibleFrame.size
            ?? NSScreen.main?.visibleFrame.size ?? CGSize(width: 1200, height: 900)
    }

    private var sheetSize: CGSize { LibraryPicker.sheetSize(available: availableSize) }
    private var minimumSize: CGSize { LibraryPicker.minimumSheetSize(available: availableSize) }

    var body: some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 10) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Choose from Library").font(.headline)
                    Text("Pictures from your connected machines").font(.caption).foregroundStyle(.secondary)
                }
                HStack {
                Picker("Machine", selection: $machineID) {
                    Text("All Machines").tag(nil as MoldHost.ID?)
                    ForEach(hosts.hosts) { host in Text(host.name).tag(Optional(host.id)) }
                }
                .frame(maxWidth: 200)
                TextField("Search pictures", text: $query)
                    .textFieldStyle(.roundedBorder)
                    .frame(minWidth: 120, maxWidth: .infinity)
                    .accessibilityLabel("Search pictures")
                }
            }
            .padding(12)
            Divider()
            grid
            if let fetchError {
                Text(fetchError).font(.callout).foregroundStyle(.red)
                    .frame(maxWidth: .infinity, alignment: .leading).padding(12)
            }
            Divider()
            footer
        }
        .frame(minWidth: minimumSize.width, idealWidth: sheetSize.width,
               maxWidth: sheetSize.width, minHeight: minimumSize.height,
               idealHeight: sheetSize.height, maxHeight: sheetSize.height)
        .onDisappear { fetchTask?.cancel() }
        .sheet(item: $preview) { selection in
            let entry = selection.entry
            let authority = selection.authority
            InputMediaViewer(title: entry.print.displayName, identity: "\(entry.id)", load: {
                guard authority.isCurrent(host: hosts.host(entry.hostID), instance: hosts.instanceID(of: entry.hostID)) else { throw CancellationError() }
                let route = authority.host
                let file = try await hosts.backend(for: route).mediaPreviewFile(entry.print.filename, trashed: false)
                guard authority.isCurrent(host: hosts.host(entry.hostID), instance: hosts.instanceID(of: entry.hostID)),
                      rows.contains(where: { $0.id == entry.id }) else { throw CancellationError() }
                return file
            }, isCurrent: { rows.contains(where: { $0.id == entry.id }) && authority.isCurrent(host: hosts.host(entry.hostID), instance: hosts.instanceID(of: entry.hostID)) })
        }
        .task {
            guard library.items.isEmpty, !library.isLoading else { return }
            await library.refresh()
        }
    }

    private var footer: some View {
        HStack {
            if isFetching { ProgressView().controlSize(.small) }
            Text(isFetching ? "Preparing picture…" : LibraryPickerSheet.caption(count: rows.count))
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            Button("Cancel") { dismiss() }
                .keyboardShortcut(.cancelAction)
            Button("Use Picture", action: use)
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.borderedProminent)
                .disabled(!rows.contains { $0.id == selected } || isFetching)
        }
        .padding(12)
    }

    /// Not `private`: `LibraryPickerSheet+Grid` draws one tile per row.
    var rows: [LibraryEntry] {
        LibraryPicker.rows(library.items, query: query, machineID: machineID,
                           availableHosts: Set(hosts.hosts.map(\.id)),
                           reachableHosts: Set(hosts.hosts.filter { hosts.isUp($0) }.map(\.id)),
                           hiddenCollectionIDs: library.hiddenCollectionIDs, privacyEvidence: library.trashed, privacyLinks: library.syncPrivacyLinks())
    }

    func showPreview(_ entry: LibraryEntry) {
        guard let route = hosts.host(entry.hostID) else { return }
        preview = PreviewSelection(entry: entry, authority: .init(host: route, instance: hosts.instanceID(of: entry.hostID)))
    }

    /// Not `private`: a double-clicked tile calls this directly.
    func use() {
        guard !isFetching, let selected, rows.contains(where: { $0.id == selected }) else { return }
        isFetching = true
        fetchError = nil
        fetchTask = Task {
            defer { isFetching = false }
            do {
                let picture = try await PictureSource.bytes(
                    of: .print(selected), accepting: accepting,
                    hosts: hosts, library: library)
                try Task.checkCancellation()
                guard rows.contains(where: { $0.id == selected }) else { return }
                pick(picture)
                dismiss()
            } catch is CancellationError {
                return
            } catch {
                fetchError = error.localizedDescription
                hosts.report(error, on: selected.host, doing: "fetch that picture")
            }
        }
    }
}
