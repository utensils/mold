import AppKit
import MoldClient
import MoldStyle
import SwiftUI

/// One tile's picture.
///
/// The square is drawn immediately and the image is overlaid when it arrives,
/// so the grid's geometry never depends on decode state -- tiles don't reflow
/// under a scroll as pictures land.
struct LibraryThumbnail: View {
    let item: LibraryEntry
    let host: MoldHost
    let edge: CGFloat

    @Environment(ThumbnailCache.self) private var cache
    @State private var image: NSImage?

    var body: some View {
        Rectangle()
            .fill(.quaternary)
            .aspectRatio(1, contentMode: .fit)
            .overlay {
                if let image {
                    Image(nsImage: image)
                        .resizable()
                        .interpolation(.medium)
                        .aspectRatio(contentMode: .fill)
                } else {
                    Image(systemName: item.print.isVideo ? "film" : "photo")
                        .font(.title3)
                        .foregroundStyle(.tertiary)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: Chrome.thumbnailRadius, style: .continuous))
            .task(id: taskKey) { await load() }
    }

    /// Re-fetch when the print's bytes change or the requested size crosses a
    /// bucket -- not on every pixel of a zoom slider.
    private var taskKey: String {
        "\(item.id.filename)|\(item.print.mediaVersion ?? "")|\(bucket)"
    }

    /// Only two renditions are ever asked for, however smooth the zoom.
    private var bucket: Int { edge > 160 ? 512 : 256 }

    private func load() async {
        image = await cache.image(for: item, host: host, size: bucket)
    }
}
