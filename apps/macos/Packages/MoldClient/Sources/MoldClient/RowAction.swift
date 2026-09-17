import Foundation

/// One entry in a menu: a command, a submenu, or a separator.
///
/// THE menu model. Every contextual menu, every menu-bar menu and every
/// click-to-open `Menu` in this app is a `[RowAction]` and nothing else --
/// named once so a row's right-click menu is built from the SAME list its
/// inline controls and its menu-bar twin are, rather than a second opinion
/// that drifts. It lives here, beside the wire types, because it is pure: a
/// test asks what a row offers without rendering anything.
///
/// It grew `children` and `separator` because three lanes each invented their
/// own menu type in the same week -- the Library's modelled submenus (Move to
/// Collection ▸, Export ▸) and explicit grouping, which this one lacked, and
/// so could not be folded in without them.
/// DELIBERATELY NOT `Identifiable`: a drawn menu can repeat itself. Every
/// separator is `RowAction(title: "")`, and two submenus can share a title,
/// so nothing in the VALUE tells one row from another. A rendered row is
/// identified by its POSITION in the list it was rendered into, which is
/// what `RowActionMenu` keys its `ForEach` on -- a content-derived `id` here
/// handed SwiftUI the same identity for every divider in the menu.
public struct RowAction<Kind: Hashable>: Equatable {
    /// What performing this row MEANS. `nil` on a submenu and on a separator,
    /// neither of which is something to do.
    public let kind: Kind?
    public let title: String
    /// Drawn last, behind a separator: anything that cannot be taken back.
    public var isDestructive: Bool
    /// Present but inert -- a row that cannot do this, said plainly rather
    /// than by the item quietly not being there.
    public var isDisabled: Bool
    /// A submenu's own items. Empty on an ordinary command.
    public var children: [RowAction]

    public init(kind: Kind? = nil, title: String, isDestructive: Bool = false,
                isDisabled: Bool = false, children: [RowAction] = []) {
        self.kind = kind
        self.title = title
        self.isDestructive = isDestructive
        self.isDisabled = isDisabled
        self.children = children
    }

    /// A rule the menu draws as a divider rather than a row.
    public static var separator: RowAction { RowAction(title: "") }

    public var isSeparator: Bool { kind == nil && children.isEmpty && title.isEmpty }
    public var isSubmenu: Bool { !children.isEmpty }

    /// Whether a submenu is worth opening: one with nothing enabled in it is
    /// a dead end, and `rendered` drops it.
    var leadsSomewhere: Bool { children.contains { !$0.isDisabled } }

    /// The same row, its kind read as something wider -- what a surface that
    /// mixes a declared list with items of its own needs to end up with ONE
    /// list rather than two menus stacked.
    public func mapKind<Other: Hashable>(_ transform: (Kind) -> Other) -> RowAction<Other> {
        RowAction<Other>(kind: kind.map(transform), title: title,
                         isDestructive: isDestructive, isDisabled: isDisabled,
                         children: children.map { $0.mapKind(transform) })
    }
}

extension RowAction: Sendable where Kind: Sendable {}
