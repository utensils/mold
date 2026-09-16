import MoldClient
import MoldStyle
import SwiftUI

struct LibraryCell: View {
    let entry: LibraryEntry
    let host: MoldHost
    let edge: CGFloat
    let isSelected: Bool
    let isLead: Bool
    let showsHostBadge: Bool

    var body: some View {
        LibraryThumbnail(entry: entry, host: host, edge: edge)
            .overlay(alignment: .bottomTrailing) { badges }
            .overlay(alignment: .topTrailing) { trashCountdown }
            .overlay(alignment: .topLeading) { hostBadge }
            .overlay { selectionRing }
            .contentShape(Rectangle())
            .help(entry.print.metadata.prompt ?? entry.print.filename)
            .accessibilityLabel(entry.print.metadata.prompt ?? entry.print.filename)
    }

    @ViewBuilder private var selectionRing: some View {
        if isSelected {
            // The lead is drawn heavier: with several selected, the arrow keys
            // move from ONE of them and you need to see which.
            RoundedRectangle(cornerRadius: Chrome.thumbnailRadius, style: .continuous)
                .strokeBorder(Color.accentColor, lineWidth: isLead ? 3 : 2)
                .opacity(isLead ? 1 : 0.6)
        }
    }

    @ViewBuilder private var badges: some View {
        HStack(spacing: 4) {
            if entry.print.isFavorite {
                Image(systemName: "star.fill")
            }
            if entry.print.isVideo {
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

    /// How long the machine will keep a trashed print. Each one carries its
    /// own countdown, which is why the trash is never collapsed or grouped.
    @ViewBuilder private var trashCountdown: some View {
        if let purge = entry.print.purgeAt {
            let days = max(0, Int((Double(purge) - Date.now.timeIntervalSince1970) / 86_400))
            Text(days == 0 ? "today" : "\(days)d")
                .font(.caption2)
                .monospacedDigit()
                .foregroundStyle(.white)
                .padding(.horizontal, 5)
                .padding(.vertical, 2)
                .background(Chrome.badgeBackdrop, in: Capsule())
                .padding(5)
                .help("Purged in \(days) days")
        }
    }

    /// Which machine made it. Only shown when more than one is in the list --
    /// on a single-host library it would be noise on every tile.
    @ViewBuilder private var hostBadge: some View {
        if showsHostBadge {
            Text(entry.hostName)
                .font(.caption2)
                .foregroundStyle(.white)
                .padding(.horizontal, 5)
                .padding(.vertical, 2)
                .background(Chrome.badgeBackdrop, in: Capsule())
                .padding(5)
        }
    }
}
