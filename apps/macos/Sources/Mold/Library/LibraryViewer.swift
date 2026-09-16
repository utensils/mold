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

    var body: some View {
        ZStack {
            Color.clear
            if let image = full ?? placeholder {
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
        .onKeyPress(.escape) { onClose(); return .handled }
        .onKeyPress(.leftArrow) { onStep(-1); return .handled }
        .onKeyPress(.rightArrow) { onStep(1); return .handled }
        .focusable()
        .focusEffectDisabled()
    }

    private var bar: some View {
        HStack(spacing: 12) {
            Button { onClose() } label: { Label("Back", systemImage: "chevron.left") }
                .help("Back to the library (esc)")
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

    private func load() async {
        full = nil
        if let host {
            placeholder = await cache.image(for: entry, host: host, size: 512)
        }
        guard let data = await actions.data(for: entry) else { return }
        full = NSImage(data: data)
    }
}
