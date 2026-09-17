import Foundation

/// One thing a Generate surface can do, declared ONCE so the inline control,
/// the contextual menu and (where one exists) the menu-bar item all render the
/// same list in the same order, and a test can ask what a row offers without a
/// view.
///
/// Nothing here is invented: every case is either already reachable from an
/// inline control in this pane or a trivial composition of one. Anything that
/// would need a verb the app does not have -- per-entry prompt-history delete,
/// mask invert -- is deliberately absent rather than shown and broken.
enum GenerateAction: Hashable, CaseIterable {
    // A finished result.
    case saveACopy
    case copyResult
    case showInLibrary
    case useAsSourceImage
    case addAsReference

    // The source well.
    case chooseFile
    case chooseFromLibrary
    case paste
    case editMask
    case removeSource

    // A reference-strip item, and its background.
    case moveLeft
    case moveRight
    case replacePicture
    case removeReference
    case addReference
    case removeAllReferences

    // An identity photograph.
    case replacePhoto
    case removePhoto

    // The mask row.
    case clearMask

    // An adapter row.
    case resetStrength
    case removeAdapter

    // A recent prompt.
    case usePrompt
    case copyPrompt

    var title: String {
        switch self {
        case .saveACopy: "Save a Copy…"
        case .copyResult, .copyPrompt: "Copy"
        case .showInLibrary: "Show in Library"
        case .useAsSourceImage: "Use as Source Image"
        case .addAsReference: "Add as Reference"
        case .chooseFile, .replacePicture, .replacePhoto: "Choose File…"
        case .chooseFromLibrary: "Choose from Library…"
        case .paste: "Paste"
        case .editMask: "Edit Mask…"
        case .removeSource, .removeReference, .removePhoto, .removeAdapter: "Remove"
        case .moveLeft: "Move Left"
        case .moveRight: "Move Right"
        case .addReference: "Add…"
        case .removeAllReferences: "Remove All"
        case .clearMask: "Clear Mask"
        case .resetStrength: "Reset Strength"
        case .usePrompt: "Use Prompt"
        }
    }

    /// Destructive items are shown last, after a divider, with the system's
    /// destructive role -- the house rule everywhere in this app.
    var isDestructive: Bool {
        switch self {
        case .removeSource, .removeReference, .removePhoto, .removeAdapter,
             .removeAllReferences, .clearMask:
            true
        default:
            false
        }
    }
}

/// A rendered menu: the ordinary items, then the destructive ones behind a
/// divider. EMPTY means no menu at all -- a row with nothing applicable must
/// not sprout an empty contextual menu.
struct GenerateMenuItems: Equatable {
    let ordinary: [GenerateAction]
    let destructive: [GenerateAction]

    var isEmpty: Bool { ordinary.isEmpty && destructive.isEmpty }
    /// Flat, in render order -- what a test asserts against.
    var all: [GenerateAction] { ordinary + destructive }

    /// The one place the ordering rule lives.
    init(_ actions: [GenerateAction]) {
        ordinary = actions.filter { !$0.isDestructive }
        destructive = actions.filter(\.isDestructive)
    }
}
