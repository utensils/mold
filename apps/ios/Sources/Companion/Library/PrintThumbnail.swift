import MoldClient
import SwiftUI

/// A print's picture at the size it is drawn, from the loader; a quiet
/// placeholder until it arrives (never a spinner or animation per tile).
struct PrintThumbnail: View {
    @Environment(ThumbnailLoader.self) private var loader
    @Environment(\.displayScale) private var scale
    let entry: LibraryEntry
    let points: CGFloat
    var trashed = false
    var contentMode: ContentMode = .fill
    @State private var image: UIImage?

    var body: some View {
        ZStack {
            Rectangle().fill(.fill.tertiary)
            if let image {
                Image(uiImage: image).resizable().aspectRatio(contentMode: contentMode)
            }
        }
        .task(id: "\(entry.hostID)|\(entry.id.filename)|\(ThumbnailLoader.version(entry.print))|\(ThumbnailLoader.bucket(Int(points * scale)))|\(trashed)") {
            image = loader.cachedThumbnail(for: entry)
            let loaded = await loader.image(for: entry, pixels: Int(points * scale), trashed: trashed)
            guard !Task.isCancelled else { return }
            image = loaded
        }
    }
}
