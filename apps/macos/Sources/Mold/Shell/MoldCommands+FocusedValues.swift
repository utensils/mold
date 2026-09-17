import SwiftUI

// Every focused value `MoldCommands` reads, moved out of that file purely
// for size -- a pure move, so nothing here changes what already worked.

/// Lets whichever pane is showing say how it refreshes, so ⌘R means the right
/// thing in each without the menu knowing about any of them.
struct RefreshActionKey: FocusedValueKey {
    typealias Value = () -> Void
}

/// Whether the Generate pane's prompt capsule is tucked away, and how to
/// change that. Equatable on the state alone -- a closure never is, and the
/// menu only needs to redraw when the word on the item changes.
struct PromptTuckAction: Equatable {
    let isTucked: Bool
    let toggle: () -> Void

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.isTucked == rhs.isTucked }
}

struct PromptTuckKey: FocusedValueKey {
    typealias Value = PromptTuckAction
}

/// Whether the showing pane has an inspector open, and how to change that.
/// Equatable on the state alone, for the same reason `PromptTuckAction` is.
struct InspectorToggle: Equatable {
    let isShowing: Bool
    let toggle: () -> Void

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.isShowing == rhs.isShowing }
}

struct InspectorToggleKey: FocusedValueKey {
    typealias Value = InspectorToggle
}

/// Whether a text field somewhere is being typed into.
///
/// A key equivalent is checked BEFORE the focused field sees the key, so an
/// unmodified Escape or arrow bound to a control will take the key off a
/// caret. Fields that can be focused while such a control exists publish this
/// so the control can stand down; ABSENT means nothing is being typed, which
/// is why it is published as nil rather than as `false`.
struct EditingTextKey: FocusedValueKey {
    typealias Value = Bool
}

/// What File ▸ Export… and File ▸ Save a Copy… act on -- the Library's own
/// selection, reached through `LibraryMenu.swift`'s own `actions.save` and
/// `actions.export`, so the menu bar item and the right-click item are the
/// same call (design S6).
struct LibraryFile: Equatable {
    let count: Int
    /// What the SINGLE selected print can also be saved as -- empty when
    /// more than one print is selected or it has no other form
    /// (`LibraryMenu.swift`'s own `exportMenu` gate).
    let exportFormats: [String]
    let save: () -> Void
    let export: (String) -> Void

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.count == rhs.count && lhs.exportFormats == rhs.exportFormats
    }
}

struct LibraryFileKey: FocusedValueKey {
    typealias Value = LibraryFile
}

/// Edit ▸ Find, focusing the pane's own `.searchable` field
/// (`.searchFocused(_:)`) rather than opening a bespoke find bar.
struct FindActionKey: FocusedValueKey {
    typealias Value = () -> Void
}

/// View ▸ Larger/Smaller Thumbnails, stepping the Library's own slider range
/// (`LibraryPane+Toolbar.swift`'s `88...260`) through `ThumbnailStep`.
struct ThumbnailScaleAction: Equatable {
    let edge: CGFloat
    let step: (CGFloat) -> Void

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.edge == rhs.edge }
}

struct ThumbnailScaleKey: FocusedValueKey {
    typealias Value = ThumbnailScaleAction
}

extension FocusedValues {
    var refreshAction: RefreshActionKey.Value? {
        get { self[RefreshActionKey.self] }
        set { self[RefreshActionKey.self] = newValue }
    }

    var promptTuck: PromptTuckAction? {
        get { self[PromptTuckKey.self] }
        set { self[PromptTuckKey.self] = newValue }
    }

    var inspectorToggle: InspectorToggle? {
        get { self[InspectorToggleKey.self] }
        set { self[InspectorToggleKey.self] = newValue }
    }

    var editingText: Bool? {
        get { self[EditingTextKey.self] }
        set { self[EditingTextKey.self] = newValue }
    }

    var libraryFile: LibraryFile? {
        get { self[LibraryFileKey.self] }
        set { self[LibraryFileKey.self] = newValue }
    }


    var findAction: FindActionKey.Value? {
        get { self[FindActionKey.self] }
        set { self[FindActionKey.self] = newValue }
    }

    var thumbnailScale: ThumbnailScaleAction? {
        get { self[ThumbnailScaleKey.self] }
        set { self[ThumbnailScaleKey.self] = newValue }
    }
}

/// The `88...260` range `LibraryPane+Toolbar.swift`'s own slider already
/// uses -- pure, so ⌘+/⌘− land on exactly the same range the drag does,
/// without a view to press them through (design S6).
enum ThumbnailStep {
    static let delta: CGFloat = 24
    static let range: ClosedRange<CGFloat> = 88...260

    static func apply(_ edge: CGFloat, delta: CGFloat) -> CGFloat {
        min(max(edge + delta, range.lowerBound), range.upperBound)
    }
}
