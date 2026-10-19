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
    var fresh = false

    var body: some View {
        LibraryThumbnail(entry: entry, host: host, edge: edge, continuous: true)
            .overlay(alignment: .bottomTrailing) { badges }
            .overlay(alignment: .topTrailing) { trashCountdown }
            .overlay(alignment: .topLeading) {
                VStack(alignment: .leading, spacing: 0) {
                    if fresh, entry.print.purgeAt == nil { newBadge }
                    hostBadge
                }
            }
            .overlay { selectionRing }
            .clipped()
            .contentShape(Rectangle())
            // One element, not five: the badges are facts ABOUT the print and
            // belong in its sentence, not as separate stops on the way past it.
            .accessibilityElement(children: .ignore)
            .accessibilityLabel((fresh ? String(localized: "New") + ", " : "") + entry.spokenDescription(showsHost: showsHostBadge))
            .accessibilityAddTraits(isSelected ? [.isSelected, .isImage] : .isImage)
    }

    private var newBadge: some View {
        Text("New")
            .font(.caption2.weight(.semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, 5).padding(.vertical, 2)
            .background(Color.accentColor.mix(with: .black, by: 0.4), in: .rect(cornerRadius: 5))
            .padding(5)
    }

    @ViewBuilder private var selectionRing: some View {
        if isSelected {
            Rectangle().fill(Color.accentColor.opacity(0.12))
            Rectangle()
                .strokeBorder(Color.accentColor, lineWidth: isLead ? 6 : 4)
            // A contrasting inner edge stays visible on blue or busy media.
            Rectangle().inset(by: isLead ? 6 : 4)
                .strokeBorder(.white, lineWidth: 1)
            VStack {
                Spacer(minLength: 0)
                HStack {
                    Image(systemName: "checkmark.circle.fill")
                        .symbolRenderingMode(.palette)
                        .foregroundStyle(.white, Color.accentColor)
                        .font(.title3.weight(.semibold))
                        .background(.white, in: Circle())
                        .padding(8)
                    Spacer(minLength: 0)
                }
            }
            .allowsHitTesting(false)
        }
    }

    var mediaSymbol: String? {
        switch entry.print.kind {
        case .clip: "play.fill"
        case .mesh: "cube"
        case .picture: nil
        }
    }

    @ViewBuilder private var badges: some View {
        if entry.isFavorite || mediaSymbol != nil {
        HStack(spacing: 4) {
            if entry.isFavorite {
                Image(systemName: "star.fill")
            }
            if let mediaSymbol { Image(systemName: mediaSymbol) }
        }
        .font(.caption2)
        // Badges sit over arbitrary pixels, where a semantic label color could
        // land white-on-white. A backdrop is what makes them legible.
        .foregroundStyle(.white)
        .padding(4)
        .background(Chrome.badgeBackdrop, in: Capsule())
        .padding(5)
        }
    }

    /// How long the machine will keep a trashed print. Each one carries its
    /// own countdown, which is why the trash is never collapsed or grouped.
    @ViewBuilder private var trashCountdown: some View {
        if let left = TrashCountdown.days(until: entry.print.purgeAt) {
            let days = max(0, left)
            Text(days == 0 ? "today" : "\(days)d")
                .font(.caption2)
                .monospacedDigit()
                .foregroundStyle(.white)
                .padding(.horizontal, 5)
                .padding(.vertical, 2)
                .background(Chrome.badgeBackdrop, in: Capsule())
                .padding(5)
                .help(days == 0 ? "Permanently deleted today" : "Permanently deleted in \(days) days")
        }
    }

    /// Every machine holding it -- a print saved to This Mac is ONE tile,
    /// and the badge is where it says it is on both. Only shown when more
    /// than one machine is in the list; on a single-host library it would be
    /// noise on every tile.
    @ViewBuilder private var hostBadge: some View {
        if showsHostBadge {
            ViewThatFits(in: .horizontal) {
                badge(entry.hostBadge(compact: false))
                badge(entry.hostBadge(compact: true))
            }
        }
    }

    private func badge(_ text: String) -> some View {
        Text(text)
            .font(.caption2)
            .lineLimit(1)
            .fixedSize()
            .foregroundStyle(.white)
            .padding(.horizontal, 5)
            .padding(.vertical, 2)
            .background(Chrome.badgeBackdrop, in: Capsule())
            .padding(5)
    }
}
