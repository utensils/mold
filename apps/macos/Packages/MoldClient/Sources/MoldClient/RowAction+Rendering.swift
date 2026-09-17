import Foundation

// The house rules, in the one place every menu in the app goes through. Pure,
// so a surface's order, wording and gating are a test rather than something
// you check by right-clicking.
public extension RowAction {
    /// Everything ordinary, then everything destructive -- whatever order a
    /// surface declared them in, so a menu's bottom item is always the one
    /// that cannot be taken back.
    ///
    /// For a FLAT surface. A surface that groups its own list says so with
    /// `.separator`s, and `rendered` then leaves its order alone.
    static func ordered(_ actions: [RowAction]) -> [RowAction] {
        actions.filter { !$0.isDestructive } + actions.filter(\.isDestructive)
    }

    /// The list a menu actually draws, in draw order.
    ///
    /// Three rules, pinned here rather than at each of the fifteen surfaces:
    /// a submenu with no enabled entry in it is a dead end and is dropped; a
    /// list that carries no separator of its own gets the house one, before
    /// the first destructive item; and no menu opens, closes or doubles on a
    /// divider, whatever the gating above left out.
    static func rendered(_ actions: [RowAction]) -> [RowAction] {
        let live = actions.filter { $0.isSeparator || $0.kind != nil || $0.leadsSomewhere }
        return trimmingSeparators(live.contains(where: \.isSeparator) ? live : grouped(live))
    }

    /// Whether the row carries a menu AT ALL.
    ///
    /// A right-click that opens an empty menu is worse than one that opens
    /// nothing: it says there is something here and then does not say what. A
    /// disabled placeholder is the same lie with an extra row. So a row with
    /// no applicable action gets no menu attached -- and that belongs here
    /// rather than at one call site, because every caller has the case.
    static func offersMenu(_ actions: [RowAction]) -> Bool {
        rendered(actions).contains { !$0.isSeparator }
    }

    /// The house rule for a flat list: ordinary, then a divider, then what
    /// cannot be taken back.
    ///
    /// Public for the one surface that puts a group of its OWN ahead of a
    /// declared list -- an adapter row leads with the adapter's trained words,
    /// which are its vocabulary rather than actions on it. Declaring the
    /// separator between the two groups makes the whole list explicitly
    /// grouped, so it has to carry this one too.
    static func grouped(_ actions: [RowAction]) -> [RowAction] {
        let ordinary = actions.filter { !$0.isDestructive }
        let destructive = actions.filter(\.isDestructive)
        guard !ordinary.isEmpty, !destructive.isEmpty else { return ordinary + destructive }
        return ordinary + [.separator] + destructive
    }

    private static func trimmingSeparators(_ actions: [RowAction]) -> [RowAction] {
        var kept: [RowAction] = []
        for action in actions
            where !(action.isSeparator && (kept.isEmpty || kept.last?.isSeparator == true)) {
            kept.append(action)
        }
        while kept.last?.isSeparator == true { kept.removeLast() }
        return kept
    }
}
