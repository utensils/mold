import MoldClient
import SwiftUI

/// The merged library.
///
/// Which shelf is showing comes from the sidebar, so this pane is only ever
/// asked "draw what the query selects" -- the filtering itself is
/// `LibraryQuery`, which is pure and tested away from any view.
struct LibraryPane: View {
    @Environment(HostStore.self) var hosts
    @Environment(LibraryStore.self) var library
    @Environment(LibraryNavigation.self) var navigation
    @Environment(GenerateController.self) var generate
    @Environment(DraftPersistence.self) var drafts
    @Environment(ModelStore.self) var models
    /// Which print the Generate pane's draft came from, and what its own
    /// machine still holds for it. Written by `reuse(_:)` in `+Wiring`.
    @Environment(ReuseStore.self) var reuseStore
    @Environment(PrintMaterializer.self) var materializer
    /// Make Bigger… and the clip jobs it starts.
    @Environment(UpscaleStore.self) var upscales
    /// The WINDOW's undo manager. The store registers against it so Edit ▸
    /// Undo, which SwiftUI wires to the responder chain, finds our edits --
    /// and so a focused text field still keeps ⌘Z for itself.
    /// Not `private`: `LibraryPane+Chrome` hands it to the store's undo,
    /// and `private` does not cross a file boundary.
    @Environment(\.undoManager) var undoManager
    @Binding var destination: Destination

    @State var selection = LibraryCursor.Selection.empty
    @State var viewing: PrintID?
    /// Cancels an attachment whose selection moved while its bytes loaded.
    @State var attachmentVersion = 0
    /// Backs Edit ▸ Find (design S6): `.searchFocused($isSearchFocused)`
    /// below, set from `body`'s own `findAction` focused value.
    /// Deliberately not `private`: what the menu is offered is assembled in an
    /// extension in another file, and `private` does not cross that boundary.
    @FocusState var isSearchFocused: Bool
    /// Published by the inspector's Title field and "Add a tag". The search
    /// field says so through `isSearchFocused` above, which this pane owns.
    @FocusedValue(\.editingText) var editingText: Bool?
    /// Persisted, and deliberately not `private`: the toolbar button that
    /// flips it lives in an extension in another file.
    @AppStorage("libraryShowsInspector", store: AppStorageSuite.defaults)
    var showsInspector = true
    @State private var pendingDestruction: LibraryActions.Destruction?
    /// The filtered, sorted and grouped library, kept between passes. A body
    /// pass happens on every arrow key and every character typed, and re-doing
    /// all of that per pass is work proportional to the whole library for a
    /// change that moved the cursor. See `LibraryShowingCache`.
    @State private var index = LibraryShowingCache()
    /// The shelf being renamed from the MENU BAR. The sidebar row has its own;
    /// both open the same sheet.
    @State var renamingShelf: CollectionShelf?
    /// The mesh export waiting on its controls. Both doors -- the tile's menu
    /// and the viewer's -- open this one sheet.
    @State private var meshExport: MeshExportPrompt?

    var actions: LibraryActions {
        LibraryActions(hosts: hosts, library: library, reuse: reuse,
                       useAsSource: attachmentOffer.canUseAsSource
                           ? { attach($0, as: .source) } : nil,
                       addAsReference: attachmentOffer.canAddReference
                           ? { attach($0, as: .reference) } : nil,
                       confirmDestruction: { pendingDestruction = $0 },
                       materializer: materializer, upscales: upscales,
                       collectionAction: { performCollection($0) },
                       meshExport: { meshExport = $0 })
    }

    // Three stages rather than one chain: what is on screen, what dresses it,
    // and what plugs it in. `showing` is derived once per DATA or QUERY change
    // and threaded down, rather than each stage re-filtering the whole library
    // -- or this pass re-doing what the last one already worked out.
    var body: some View {
        let showing = index.showing(pool: pool, revision: library.rows.value,
                                    query: resolved, selection: selection.items)
        return watched(showing)
            .focusedSceneValue(\.refreshAction) { Task { await actions.reload() } }
            .focusedSceneValue(\.inspectorToggle, InspectorToggle(isShowing: showsInspector) {
                showsInspector.toggle()
            })
            .focusedSceneValue(\.librarySelection, menuSelection(showing))
            .focusedSceneValue(\.libraryImport, menuImport)
            .focusedSceneValue(\.libraryFile, menuFile(showing))
            .focusedSceneValue(\.findAction) { isSearchFocused = true }
            .focusedSceneValue(\.thumbnailScale, ThumbnailScaleAction(edge: navigation.edge) { delta in
                navigation.edge = ThumbnailStep.apply(navigation.edge, delta: delta)
                navigation.rememberEdge()
            })
            .destructionDialog($pendingDestruction)
            .sheet(item: $renamingShelf) { ShelfNameSheet(shelf: $0) }
            .sheet(item: $meshExport) { prompt in
                MeshExportSheet(prompt: prompt) { request in
                    actions.export(prompt.entry, request: request)
                }
            }
    }
}
