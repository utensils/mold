import MoldClient
import SwiftUI

/// The Model menu.
///
/// `LibraryCommands`'s own reason (decision 22, M5): the menu bar is what
/// Help ▸ Search searches and VoiceOver reads, and Delete existing only in a
/// contextual menu would be unreachable from the keyboard. Every item here is
/// the same call the row's contextual menu makes -- both draw from the exact
/// `ModelActions.Item` list `ModelActions.menu(for:...)` resolved, so there is
/// nothing here to fall out of step with the row.
struct ModelCommands: Commands {
    @FocusedValue(\.modelSelection) private var selection

    var body: some Commands {
        CommandMenu("Model") {
            RowActionMenu(actions: selection?.items ?? []) { selection?.perform($0) }
        }
    }
}

/// What the Models pane's current selection can do, and how to do it --
/// resolved once per body pass the same way `LibrarySelection` is.
///
/// Equatable on the target and items, never `perform`: a closure is never
/// equal to itself, but two models with the same actions still need distinct
/// focused values or the menu can keep acting on the previous row.
struct ModelSelection: Equatable {
    struct Target: Equatable {
        let host: MoldHost.ID
        let model: Model.ID
    }

    let target: Target
    let items: [ModelActions.Item]
    let perform: (ModelActions.Kind) -> Void

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.target == rhs.target && lhs.items == rhs.items
    }
}

struct ModelSelectionKey: FocusedValueKey {
    typealias Value = ModelSelection
}

extension FocusedValues {
    var modelSelection: ModelSelection? {
        get { self[ModelSelectionKey.self] }
        set { self[ModelSelectionKey.self] = newValue }
    }
}
