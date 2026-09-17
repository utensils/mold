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
    let onClose: () -> Void
    let onStep: (Int) -> Void

    @Environment(ThumbnailCache.self) private var cache
    @State private var full: NSImage?
    @State private var placeholder: NSImage?
    @State private var player: AVPlayer?
    /// Anything being typed into keeps the keyboard for its caret. A key
    /// equivalent is checked BEFORE the focused field sees the key, so every
    /// shortcut this viewer binds stands down while text is being edited --
    /// the search field says so through the environment, the inspector's
    /// fields through `editingText`.
    @Environment(\.isSearching) private var isSearching
    @FocusedValue(\.editingText) private var editingText: Bool?

    var body: some View {
        ZStack {
            Color.clear
            if entry.print.isVideo {
                video
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
        .task(id: entry.id) { await load() }
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

    private var bar: some View {
        HStack(spacing: 12) {
            // A grid, not a third chevron: the icon says WHERE back goes, and
            // the bar no longer reads as three arrows in a row.
            Button { onClose() } label: { Label("Library", systemImage: "square.grid.2x2") }
                // Every key this viewer answers is bound to the control that
                // performs it, never to a focus the viewer holds: SwiftUI
                // hands the grid's focus to the search field the moment the
                // viewer replaces it, so `.onKeyPress` here reached nothing.
                // A key equivalent is window-scoped and needs no focus, and
                // these controls exist only while a print is showing.
                .keyboardShortcut(isEditing ? nil : KeyboardShortcut.cancelAction)
                .help("Back to the library (esc)")
            Divider().frame(height: 14)
            Button { onStep(-1) } label: { Label("Previous", systemImage: "chevron.left") }
                .keyboardShortcut(stepping(.leftArrow))
                .help("Previous print (←)")
            Button { onStep(1) } label: { Label("Next", systemImage: "chevron.right") }
                .keyboardShortcut(stepping(.rightArrow))
                .help("Next print (→)")
            Spacer()
            Text(entry.print.metadata.prompt ?? entry.print.filename)
                .lineLimit(1)
                .foregroundStyle(.secondary)
            Spacer()
            Button { actions.toggleFavorite([entry]) } label: {
                Label("Favorite",
                      systemImage: entry.print.isFavorite ? "star.fill" : "star")
            }
            .help(entry.print.isFavorite ? "Remove from favorites" : "Add to favorites")
            Button { actions.save([entry]) } label: {
                Label("Save", systemImage: "square.and.arrow.down")
            }
            .help("Save a copy")
        }
        .buttonStyle(.accessoryBar)
        .labelStyle(.iconOnly)
        .padding(10)
        .background(.bar)
    }

    /// Written once, because Escape and the two arrows all answer to it: a
    /// caret has the better claim on an unmodified key, so every shortcut here
    /// stands down and the field answers instead. How long that lasts is the
    /// FIELD's to decide -- the search field releases focus on its own Escape,
    /// so the next one leaves the viewer; a title field reverts and keeps
    /// typing, so you leave it before Escape means "back" again.
    private var isEditing: Bool { isSearching || editingText == true }

    private func stepping(_ key: KeyEquivalent) -> KeyboardShortcut? {
        isEditing ? nil : KeyboardShortcut(key, modifiers: [])
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
        placeholder = await cache.image(for: entry, host: host, size: 512)
        guard let data = await actions.data(for: entry) else { return }
        full = NSImage(data: data)
    }
}
