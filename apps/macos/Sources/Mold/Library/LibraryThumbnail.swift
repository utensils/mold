import AppKit
import MoldClient
import MoldStyle
import SwiftUI

/// One tile's picture.
///
/// The metadata-sized placeholder is drawn immediately and the image is overlaid when it arrives,
/// so the grid's geometry never depends on decode state -- tiles don't reflow
/// under a scroll as pictures land.
struct LibraryThumbnail: View {
    let entry: LibraryEntry
    let host: MoldHost
    let edge: CGFloat
    var continuous = false

    @Environment(ThumbnailCache.self) private var cache
    @State private var image: NSImage?

    var body: some View {
        Rectangle()
            .fill(.quaternary)
            .aspectRatio(continuous ? JustifiedLayout.aspect(width: entry.print.metadata.width, height: entry.print.metadata.height) : 1, contentMode: .fit)
            .overlay {
                if let image {
                    // Preserve Library ratios; standalone square previews crop.
                    ZStack {
                        if entry.print.metadata.showsAlphaBed { AlphaBed() }
                        Image(nsImage: image)
                            .resizable()
                            .interpolation(.medium)
                            .aspectRatio(contentMode: continuous ? .fit : .fill)
                    }
                } else {
                    // a11y: placeholder inside LibraryCell, which carries the label
                    // The kind's own glyph: `LibraryToken` already names one
                    // per kind, and a mesh drawn as `photo` said the wrong
                    // thing about a print nothing else in the app could open.
                    Image(systemName: LibraryToken.kind(entry.print.kind).symbol)
                        .font(.title3)
                        .foregroundStyle(.tertiary)
                }
            }
            .clipShape(.rect(cornerRadius: continuous ? 0 : Chrome.thumbnailRadius))
            .task(id: taskKey) { await load() }
    }

    /// Re-fetch when the print's bytes change or the requested size crosses a
    /// bucket -- not on every pixel of a zoom slider.
    private var taskKey: String {
        "\(entry.id.filename)|\(entry.print.mediaVersion ?? "")|\(bucket)"
    }

    /// Only two renditions are ever asked for, however smooth the zoom.
    private var bucket: Int { edge > 160 ? 512 : 256 }

    private func load() async {
        image = await cache.image(for: entry, host: host, size: bucket)
    }
}
