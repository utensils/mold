import SwiftUI

/// Shape and spacing tokens.
///
/// Radii encode hierarchy, and the order is the point: a tile sits inside a
/// card, a card inside a well, a well inside the floating panel. Reading a
/// radius should tell you how deep in the stack you are.
public enum Chrome {
    /// The floating panel over the canvas -- the outermost surface.
    public static let panelRadius: CGFloat = 16
    /// An inset well that receives something (a dropped image, a value).
    public static let wellRadius: CGFloat = 10
    /// A card in a list or grid.
    public static let cardRadius: CGFloat = 8
    /// A library thumbnail. Matches `cardRadius` -- a thumbnail *is* a card.
    public static let thumbnailRadius: CGFloat = 8
    /// The smallest enclosed thing: a swatch, a badge backing.
    public static let tileRadius: CGFloat = 5
    /// Half of a 22pt chip, so a chip is a stadium.
    public static let chipRadius: CGFloat = 11
    public static let fieldRadius: CGFloat = 6

    /// One control row. Sliders, menus and steppers all sit on this height so
    /// a row of mixed controls has one baseline.
    public static let fieldHeight: CGFloat = 26
    public static let barHeight: CGFloat = 42
    public static let sidebarRowHeight: CGFloat = 28

    public static let shadowRadius: CGFloat = 22
    public static let shadowY: CGFloat = 8
    public static let shadowOpacity: Double = 0.28
}

public extension Chrome {
    /// Hairline separator. `separatorColor` already resolves for light and dark.
    static let hairline = Color(nsColor: .separatorColor)

    /// Washes are built from `.primary` and `.accentColor` on purpose: they
    /// inherit the system appearance and the user's own accent, so there is
    /// no palette to maintain and nothing to re-tune for dark mode.
    static let hoverWash = Color.primary.opacity(0.12)
    static let wellFill = Color.primary.opacity(0.06)
    static let wellFillHovered = Color.primary.opacity(0.10)
    static let wellFillTargeted = Color.accentColor.opacity(0.14)
    /// Backs a badge drawn over a picture, where a semantic color would be
    /// illegible against arbitrary pixels.
    static let badgeBackdrop = Color.black.opacity(0.6)
}
