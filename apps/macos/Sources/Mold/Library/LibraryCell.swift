import MoldClient
import MoldStyle
import SwiftUI

struct LibraryCell: View {
    let item: LibraryEntry
    let host: MoldHost
    let edge: CGFloat
    let isSelected: Bool
    let showsHostBadge: Bool

    var body: some View {
        LibraryThumbnail(item: item, host: host, edge: edge)
            .overlay(alignment: .bottomTrailing) { badges }
            .overlay(alignment: .topLeading) { hostBadge }
            .overlay { selectionRing }
            .contentShape(Rectangle())
            .help(item.print.metadata.prompt ?? item.print.filename)
            .accessibilityLabel(item.print.metadata.prompt ?? item.print.filename)
    }

    @ViewBuilder private var selectionRing: some View {
        if isSelected {
            RoundedRectangle(cornerRadius: Chrome.thumbnailRadius, style: .continuous)
                .strokeBorder(Color.accentColor, lineWidth: 2)
        }
    }

    @ViewBuilder private var badges: some View {
        HStack(spacing: 4) {
            if item.print.isFavorite {
                Image(systemName: "star.fill")
            }
            if item.print.isVideo {
                Image(systemName: "play.fill")
            }
        }
        .font(.caption2)
        // Badges sit over arbitrary pixels, where a semantic label color could
        // land white-on-white. A backdrop is what makes them legible.
        .foregroundStyle(.white)
        .padding(4)
        .background(Chrome.badgeBackdrop, in: Capsule())
        .padding(5)
    }

    /// Which machine made it. Only shown when more than one is in the list --
    /// on a single-host library it would be noise on every tile.
    @ViewBuilder private var hostBadge: some View {
        if showsHostBadge {
            Text(item.hostName)
                .font(.caption2)
                .foregroundStyle(.white)
                .padding(.horizontal, 5)
                .padding(.vertical, 2)
                .background(Chrome.badgeBackdrop, in: Capsule())
                .padding(5)
        }
    }
}
