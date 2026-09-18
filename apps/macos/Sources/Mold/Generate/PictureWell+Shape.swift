import AppKit
import MoldStyle
import SwiftUI

// The square itself: the fill, the picture or its placeholder glyph, and the
// badge that says a click opens a menu. Split from the well's own wiring
// purely for size.
//
// a11y: both glyphs here are decoration on a control that IS named -- the
// well's `label` reaches VoiceOver and the tooltip from `PictureWell.square`,
// which composes this, and naming the chevron separately would announce a
// second element for the badge on a menu that already says what it opens.
extension PictureWell {
    @ViewBuilder var clickable: some View {
        if opensOnClick {
            Menu {
                RowActionMenu(actions: rows, perform: route)
            } label: {
                fill
            }
            .menuStyle(.button)
            .buttonStyle(.plain)
            .menuIndicator(.hidden)
        } else {
            fill
        }
    }

    private var fill: some View {
        ZStack {
            RoundedRectangle(cornerRadius: Chrome.wellRadius, style: .continuous)
                .fill(targeted ? Chrome.wellFillTargeted : Chrome.wellFill)
            if let preview {
                Image(nsImage: preview)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else {
                Image(systemName: placeholder).foregroundStyle(.tertiary)
            }
        }
        .frame(width: size, height: size)
        .clipShape(RoundedRectangle(cornerRadius: Chrome.wellRadius, style: .continuous))
        .overlay(alignment: .bottomTrailing) { if opensOnClick { menuBadge } }
    }

    /// Reads as a menu without a full disclosure triangle taking up room in a
    /// 64pt well.
    private var menuBadge: some View {
        Image(systemName: "chevron.down")
            .font(.caption2)
            .foregroundStyle(.white)
            .padding(4)
            .background(Chrome.badgeBackdrop, in: Circle())
            .padding(3)
    }
}
