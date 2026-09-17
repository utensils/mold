import MoldClient

/// What one config row can do, in Advanced's table and on a curated pane's
/// `SettingRow` alike.
///
/// Advanced carried a Reset button in its own column and nothing else; a
/// curated row carried no action at all, so a key's name -- the thing you need
/// to search the docs or write a `mold config set` -- could only be retyped by
/// hand from the screen.
enum ConfigRowActions {
    enum Kind: Hashable {
        case copyKey, copyValue, copyVariable, reset
    }

    /// A secret's value is never offered. The field shows a mask, not the
    /// credential (`ConfigEntry.editableText`), and a menu item that put a
    /// machine's API key on the pasteboard because somebody right-clicked
    /// near it is not a convenience.
    static func offered(for entry: ConfigEntry) -> [RowAction<Kind>] {
        var actions = [RowAction<Kind>(kind: .copyKey, title: "Copy Key")]
        if entry.editor != .secret, !entry.editableText.isEmpty {
            actions.append(RowAction(kind: .copyValue, title: "Copy Value"))
        }
        if let envVar = entry.envVar {
            actions.append(RowAction(kind: .copyVariable, title: "Copy \(envVar)"))
        }
        if entry.canReset {
            // Destructive, so it lands last behind a divider: the machine
            // forgets the saved value and falls back, and this pane cannot
            // put it back.
            actions.append(RowAction(kind: .reset, title: "Reset to Default", isDestructive: true))
        }
        return RowAction.ordered(actions)
    }

    /// What each copy action puts on the pasteboard, or nothing.
    static func copied(_ kind: Kind, from entry: ConfigEntry) -> String? {
        switch kind {
        case .copyKey: entry.key
        case .copyValue: entry.editor == .secret ? nil : entry.editableText
        case .copyVariable: entry.envVar
        case .reset: nil
        }
    }
}
