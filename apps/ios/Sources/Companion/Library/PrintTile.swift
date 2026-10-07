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

    /// `false` when the grid draws the badges itself, OUTSIDE the tile's
    /// button: SwiftUI lists every view inside a button's label for testing
    /// and auditing even when it is hidden from VoiceOver, so tiny badge
    /// text inside the label was audited as if it were the tile's content.
    var drawsBadges = true
    var fresh = false

    var body: some View {
        Color.clear
            .aspectRatio(1, contentMode: .fit)
            .overlay { PrintThumbnail(entry: entry, points: points, trashed: trashed) }
            .overlay { if drawsBadges { badges } }
            .overlay(alignment: .topLeading) { selectMark }
            .clipShape(.rect(cornerRadius: 5))
            .contentShape(.rect)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel((fresh && !trashed ? String(localized: "New") + ", " : "") + entry.spokenDescription(showsHost: showsHost))
            .accessibilityAddTraits(selected ? [.isButton, .isSelected] : .isButton)
    }

    /// The star, the clip's length, the cube, the machine, the countdown --
    /// all said in the tile's spoken label, so hidden from VoiceOver.
    var badges: some View {
        Color.clear
            .overlay(alignment: .topTrailing) { topBadge }
            .overlay(alignment: .bottom) {
                HStack(spacing: 0) {
                    hostBadge.frame(minWidth: 0, maxWidth: .infinity, alignment: .leading)
                    kindBadge.layoutPriority(1)
                }
            }
            .allowsHitTesting(false)
            .accessibilityHidden(true)
    }

    @ViewBuilder private var topBadge: some View {
        if trashed, let days = TrashCountdown.days(until: entry.print.purgeAt) {
            Badge(text: String(localized: "\(days) d"), mono: true)
        } else {
            VStack(alignment: .trailing, spacing: 0) {
                if fresh { Badge(text: String(localized: "New"), accent: true) }
                if entry.print.isFavorite { Badge(symbol: "star.fill") }
            }
        }
    }

    /// Hidden at accessibility sizes, where it would cover the picture; the
    /// spoken label and the Info sheet still say it.
    @ViewBuilder private var hostBadge: some View {
        if showsHost, !size.isAccessibilitySize, !isCompact {
            Badge(text: entry.hostBadge(compact: points < 150))
        }
    }

    @ViewBuilder private var kindBadge: some View {
        switch entry.print.kind {
        // On the smallest tiles the length would cover the picture: the
        // play mark alone says it is a clip (and VoiceOver says its length).
        case .clip: Badge(symbol: "play.fill", text: isCompact ? nil : duration)
        case .mesh: Badge(symbol: "cube")
        default: EmptyView()
        }
    }

    /// The two smallest sizes: badges shrink to a symbol.
    private var isCompact: Bool { points < 90 }

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
    var accent = false

    /// XCUITest still lists hidden views; the accessibility audit keys its
    /// badge exemption on this (ShellAccessibilityTests).
    static let identifier = "tile-badge"

    var body: some View {
        Group {
            if let symbol, text != nil {
                // Native fitting keeps a crowded combined badge useful
                // without shrinking its font or the tile's full spoken label.
                ViewThatFits(in: .horizontal) {
                    line
                    Image(systemName: symbol)
                        .accessibilityHidden(true)
                        .accessibilityIdentifier(Self.identifier)
                }
            } else {
                line
            }
        }
        .font(.caption2.weight(.semibold))
        // Never wraps onto the picture: a badge is one short line or none.
        .fixedSize(horizontal: false, vertical: true)
        .foregroundStyle(.white)
        .padding(.horizontal, 5)
        .padding(.vertical, 2)
        // Dark enough for white text over any picture: over pure white it
        // still leaves 5:1.
        .background(accent ? Color.accentColor.mix(with: .black, by: 0.4) : .black.opacity(0.85), in: .rect(cornerRadius: 5))
        .padding(5)
        // a11y: the tile is one element whose spoken label already says what
        // each badge shows; a second, tiny copy would only be noise.
        .accessibilityHidden(true)
    }

    private var line: some View {
        HStack(spacing: 3) {
            // Decorative: the tile's spoken label contains the full metadata.
            if let symbol { Image(systemName: symbol).accessibilityHidden(true).accessibilityIdentifier(Self.identifier) }
            if let text {
                Text(text).font(mono ? .caption2.monospacedDigit() : .caption2)
                    .lineLimit(1)
                    .accessibilityIdentifier(Self.identifier)
            }
        }
    }
}
