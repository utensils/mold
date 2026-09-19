import AVKit
import AppKit
import MoldClient
import SwiftUI

/// One print, large.
///
/// Replaces the grid in place rather than opening a sheet: a sheet would put a
/// dimmed backdrop and a title bar between you and the picture, and arrow keys
/// would stop meaning "next".
struct LibraryViewer: View {
    let entry: LibraryEntry
    let host: MoldHost?
    let actions: LibraryActions
    let scope: LibraryScope
    let shelves: [CollectionShelf]
    let enclosingShelf: CollectionShelf?
    let trashCount: Int
    let onClose: () -> Void
    let onStep: (Int) -> Void

    @Environment(ThumbnailCache.self) private var cache
    @State private var full: NSImage?
    /// Not `private`: the mesh arm lives in `+Mesh` for size.
    @State var placeholder: NSImage?
    @State private var player: AVPlayer?
    /// Anything being typed into keeps the keyboard for its caret. A key
    /// equivalent is checked BEFORE the focused field sees the key, so every
    /// shortcut this viewer binds stands down while text is being edited --
    /// the search field says so through the environment, the inspector's
    /// fields through `editingText`.
    @Environment(\.isSearching) private var isSearching
    @FocusedValue(\.editingText) private var editingText: Bool?
    /// And so does a focused 3-D view, which ORBITS with the arrows and which
    /// only AppKit can answer for (`ArrowKeyClaim`). Re-read when a mesh view
    /// TAKES or GIVES UP first responder -- the two moments the answer can
    /// change -- and written only when it differs, so the viewer's body is not
    /// churned. The window notification stays as a backstop for the responder
    /// changes no mesh view announces (clicking into the grid, or a sheet).
    @State private var responderClaimsArrows = false

    var body: some View {
        ZStack {
            Color.clear
            if entry.print.isVideo {
                video
            } else if entry.print.isMesh {
                mesh
            } else if let image = full ?? placeholder {
                Image(nsImage: image)
                    .resizable()
                    .interpolation(full == nil ? .low : .high)
                    .aspectRatio(contentMode: .fit)
                    // The thumbnail stands in at the right aspect while the
                    // full picture loads, so nothing jumps when it arrives.
                    .opacity(full == nil ? 0.55 : 1)
            } else {
                ProgressView()
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.background)
        .overlay(alignment: .top) { bar }
        .libraryMenu(LibraryMenu(targets: [entry], scope: scope, actions: actions,
                                 shelves: shelves, enclosingShelf: enclosingShelf,
                                 trashCount: trashCount, open: nil))
        .task(id: entry.id) { await load() }
        .onReceive(NotificationCenter.default.publisher(for: MeshMetalView.claimChanged)) { _ in
            readArrowClaim()
        }
        .onReceive(NotificationCenter.default.publisher(for: NSWindow.didUpdateNotification)) { _ in
            readArrowClaim()
        }
    }

    /// Video plays in place rather than as a poster you have to export to see.
    ///
    /// `AVPlayer` builds its own requests and cannot carry `X-Api-Key`, so the
    /// URL is minted with a media ticket on a keyed host and is the plain URL
    /// on a keyless one.
    @ViewBuilder private var video: some View {
        if let player {
            VideoPlayer(player: player)
                .onDisappear { player.pause() }
        } else {
            ProgressView()
        }
    }

    /// Written once, because Escape and the two arrows all answer to it: a
    /// caret has the better claim on an unmodified key, so every shortcut here
    /// stands down and the field answers instead. How long that lasts is the
    /// FIELD's to decide -- the search field releases focus on its own Escape,
    /// so the next one leaves the viewer; a title field reverts and keeps
    /// typing, so you leave it before Escape means "back" again.
    /// Not `private`: the bar lives in `+Bar` for size, and `private` does
    /// not cross files for the same type.
    var isEditing: Bool { isSearching || editingText == true }

    /// The answer is AppKit's, asked at the moments it can change. Written
    /// only when it differs: both publishers fire often.
    private func readArrowClaim() {
        let claimed = ArrowKeyClaim.isClaimedNow
        if claimed != responderClaimsArrows { responderClaimsArrows = claimed }
    }

    /// Escape still means "back" while a 3-D view has focus -- it is the arrows
    /// the mesh claims, and a viewer you cannot leave would be worse than one
    /// whose arrows do two things.
    func stepping(_ key: KeyEquivalent) -> KeyboardShortcut? {
        isEditing || responderClaimsArrows ? nil : KeyboardShortcut(key, modifiers: [])
    }

    private func load() async {
        full = nil
        player?.pause()
        player = nil

        if entry.print.isVideo {
            // Streamed, not downloaded: a clip can be hundreds of megabytes
            // and waiting for all of it before the first frame is not playback.
            guard let url = await actions.playableURL(for: entry) else { return }
            let player = AVPlayer(url: url)
            player.play()
            self.player = player
            return
        }

        guard let host else { return }
        placeholder = await cache.image(for: entry, host: host, size: MediaURL.largestThumbnail)
        // A mesh stops at its poster: the stored bytes are a GLB, which
        // `NSImage(data:)` cannot read, so fetching them would be megabytes
        // downloaded to produce a nil. The interactive viewer is what will
        // want them (see `mesh`).
        guard !entry.print.isMesh, let data = await actions.data(for: entry) else { return }
        full = NSImage(data: data)
    }
}
