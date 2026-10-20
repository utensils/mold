import MoldClient
import SwiftUI

/// Everything a print (or a selection) can do, in the Mac tile menu's order,
/// destructive last behind a divider (DESIGN.md §5.2). Items appear only where
/// every machine involved can do them -- organizing needs the machine's
/// `gallery.organize` capability.
struct PrintMenu: View {
    @Environment(LibraryStore.self) private var library
    @Environment(PrintActions.self) private var actions
    @Environment(AppRouter.self) private var router
    @Environment(GenerateController.self) private var generate
    let entries: [LibraryEntry]
    let trashed: Bool
    var compact = false
    var requestPermanentDelete: (([LibraryEntry]) -> Void)?

    var body: some View {
        if compact {
            Menu { items } label: { Label("More", systemImage: "ellipsis.circle") }
        } else {
            items
        }
    }

    @ViewBuilder private var items: some View {
        Button {
            UIPasteboard.general.string = entries.map { $0.print.filename }.joined(separator: "\n")
        } label: { Label(entries.count == 1 ? "Copy File Name" : "Copy File Names", systemImage: "doc.on.doc") }
        .disabled(entries.isEmpty)

        if trashed {
            Button { Task { await library.putBack(entries) } } label: {
                Label("Put Back", systemImage: "arrow.uturn.backward")
            }
            Divider()
            Button(role: .destructive) { requestPermanentDelete?(entries) } label: {
                Label("Delete Immediately", systemImage: "trash.slash")
            }
            .disabled(requestPermanentDelete == nil)
        } else {
            if single, let entry = entries.first, entry.print.kind == .picture {
                ForEach(attachmentChoices) { choice in
                    Button(choice.title) { router.useAsSource(entry, attachment: choice) }
                }
            }
            if single, let entry = entries.first, entry.print.kind != .mesh {
                Button { router.reuse(entry) } label: {
                    Label("Use These Settings", systemImage: "arrow.uturn.left.circle")
                }
            }
            if single, let entry = entries.first, entry.print.isVideo,
               generate.hosts.capabilities[entry.hostID]?.canExtendLibraryVideo == true {
                Button { router.extendVideo(entry) } label: {
                    Label("Extend video…", systemImage: "film.stack")
                }
            } else if single, let entry = entries.first, entry.print.kind == .clip,
               entry.print.metadata.chainJobId != nil {
                Button { router.continueVideo(entry) } label: {
                    Label("Continue video…", systemImage: "film.stack")
                }
            }
            if canOrganize {
                Button { library.apply(.favorite(!allFavourite), to: entries) } label: {
                    Label(allFavourite ? "Unfavourite" : "Favourite", systemImage: allFavourite ? "star.slash" : "star")
                }
                SelectionCollectionMenu(entries: entries)
                Button { actions.sheet = .tags(entries) } label: { Label("Tags…", systemImage: "tag") }
                if single, let entry = entries.first {
                    Button { actions.sheet = .rename(entry) } label: { Label("Rename…", systemImage: "pencil") }
                }
            }
            Button { actions.share(entries) } label: { Label("Share…", systemImage: "square.and.arrow.up") }
            if entries.contains(where: { $0.print.kind != .mesh }) {
                Button { actions.saveToPhotos(entries) } label: {
                    Label("Save to Photos", systemImage: "square.and.arrow.down")
                }
            }
            if single, let entry = entries.first {
                if MediaExportKind(filename: entry.print.filename, trashed: trashed) != nil {
                    Button { actions.openExport(entry) } label: { Label("Export…", systemImage: "square.and.arrow.up.on.square") }
                }
                Menu {
                    Button("Save to Files…") { actions.deliverOriginal(entry, destination: .files) }
                    Button("Save to Mold folder") { actions.deliverOriginal(entry, destination: .folder) }
                    ForEach(entry.print.assets ?? []) { asset in
                        Menu(asset.displayName) {
                            Button("Share…") { actions.deliverAsset(asset, entry: entry, destination: .share) }
                            Button("Save to Files…") { actions.deliverAsset(asset, entry: entry, destination: .files) }
                            Button("Save to Mold folder") { actions.deliverAsset(asset, entry: entry, destination: .folder) }
                        }
                    }
                } label: { Label("Save Files", systemImage: "folder") }
            }
            if single, let entry = entries.first, entry.print.kind == .picture {
                Button { actions.copy(entry) } label: { Label("Copy", systemImage: "doc.on.doc") }
            }
            Divider()
            Button(role: .destructive) { Task { await library.trash(entries) } } label: {
                Label("Delete", systemImage: "trash")
            }
        }
    }

    private var attachmentChoices: [SourceMediaAttachment] {
        SourceMediaAttachment.printChoices(recipe: generate.recipe, media: generate.draft.media)
    }

    private var single: Bool { entries.count == 1 }
    private var allFavourite: Bool { !entries.isEmpty && entries.allSatisfy(\.isFavorite) }
    private var canOrganize: Bool {
        library.canOrganize(entries)
    }
}

/// The sheets `PrintActions` asks for, presented once per window.
struct PrintSheets: ViewModifier {
    var presentsActions = true
    @Environment(PrintActions.self) private var actions

    func body(content: Content) -> some View {
        let recovery = Binding(get: {
            if case .export = actions.sheet { return nil as PermissionRecovery? }
            return presentsActions ? actions.permissionRecovery : nil
        },
                               set: { actions.permissionRecovery = $0 })
        let sheet = Binding(get: { presentsActions ? actions.sheet : nil }, set: { actions.sheet = $0 })
        content.permissionAlert(recovery).sheet(item: sheet, onDismiss: { actions.presentationDismissed() }) { sheet in
            Group { switch sheet {
            case let .share(urls): ShareSheet(items: urls).presentationDetents([.medium, .large])
            case let .export(session): MediaExportSheet(session: session)
            case let .files(urls): ExportDocumentPicker(urls: urls) { saved in
                if saved { actions.status = "Saved to Files." }
                actions.sheet = nil
            }
            case let .tags(entries): TagsSheet(entries: entries)
            case let .newCollection(entries): NewCollectionSheet(entries: entries)
            case let .rename(entry): RenameSheet(entry: entry)
            } }.onAppear { actions.presentedSheet = sheet }

        }
    }
}

extension View {
    func printSheets(presentsActions: Bool = true) -> some View { modifier(PrintSheets(presentsActions: presentsActions)) }
}
