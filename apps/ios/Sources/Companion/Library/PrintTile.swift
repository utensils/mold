import MoldClient
import SwiftUI

/// One print in the grid: the picture, square, radius 5, with the Mac's
/// badges -- a star, a clip's length, a 3-D cube, the machine when there is
/// more than one, and a countdown in Recently Deleted. VoiceOver reads the
/// whole tile as one sentence (`spokenDescription`).
struct PrintTile: View {
    @Environment(\.dynamicTypeSize) private var size
    let entry: LibraryEntry
    let points: CGFloat
    let trashed: Bool
    let selecting: Bool
    let selected: Bool
    let showsHost: Bool

    var body: some View {
        Color.clear
            .aspectRatio(1, contentMode: .fit)
            .overlay { PrintThumbnail(entry: entry, points: points, trashed: trashed) }
            .overlay(alignment: .topTrailing) { topBadge }
            .overlay(alignment: .bottomLeading) { hostBadge }
            .overlay(alignment: .bottomTrailing) { kindBadge }
            .overlay(alignment: .topLeading) { selectMark }
            .clipShape(.rect(cornerRadius: 5))
            .contentShape(.rect)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(entry.spokenDescription(showsHost: showsHost))
            .accessibilityAddTraits(selected ? [.isButton, .isSelected] : .isButton)
    }

    @ViewBuilder private var topBadge: some View {
        if trashed, let days = TrashCountdown.days(until: entry.print.purgeAt) {
            Badge(text: String(localized: "\(days) d"), mono: true)
        } else if entry.print.isFavorite {
            Badge(symbol: "star.fill")
        }
    }

    /// Hidden at accessibility sizes, where it would cover the picture; the
    /// spoken label and the Info sheet still say it.
    @ViewBuilder private var hostBadge: some View {
        if showsHost, !size.isAccessibilitySize {
            Badge(text: entry.hostBadge(compact: points < 150))
        }
    }

    @ViewBuilder private var kindBadge: some View {
        switch entry.print.kind {
        case .clip: Badge(symbol: "play.fill", text: duration)
        case .mesh: Badge(symbol: "cube")
        default: EmptyView()
        }
    }

    private var duration: String? {
        guard let frames = entry.print.metadata.frames, let fps = entry.print.metadata.fps, fps > 0 else { return nil }
        let seconds = Int((Double(frames) / Double(fps)).rounded())
        return String(format: "%d:%02d", seconds / 60, seconds % 60)
    }

    @ViewBuilder private var selectMark: some View {
        if selecting {
            Image(systemName: selected ? "checkmark.circle.fill" : "circle")
                .font(.title3)
                .symbolRenderingMode(.palette)
                .foregroundStyle(.white, selected ? Color.accentColor : .black.opacity(0.35))
                .padding(6)
                .accessibilityHidden(true)
        }
    }
}

/// A small label over a picture: legible on any pixels because it carries its
/// own dark backing, the one place a fixed dark fill is right.
struct Badge: View {
    var symbol: String?
    var text: String?
    var mono = false

    var body: some View {
        HStack(spacing: 3) {
            // a11y: decorative -- the tile's spoken label says what each badge says.
            if let symbol { Image(systemName: symbol).accessibilityHidden(true) }
            if let text { Text(text).font(mono ? .caption2.monospacedDigit() : .caption2) }
        }
        .font(.caption2.weight(.semibold))
        .foregroundStyle(.white)
        .padding(.horizontal, 5)
        .padding(.vertical, 2)
        .background(.black.opacity(0.6), in: .rect(cornerRadius: 5))
        .padding(5)
    }
}
