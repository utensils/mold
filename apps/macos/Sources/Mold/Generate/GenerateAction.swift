import MoldClient

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

    // A numeric control whose default is the recipe's own.
    case resetReferenceWeight

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
        case .resetReferenceWeight: "Reset Weight"
        case .usePrompt: "Use Prompt"
        }
    }

    /// Destructive items are shown last, after a divider, with the system's
    /// destructive role -- `RowAction.rendered`'s rule, which is every menu's.
    var isDestructive: Bool {
        switch self {
        case .removeSource, .removeReference, .removePhoto, .removeAdapter,
             .removeAllReferences, .clearMask:
            true
        default:
            false
        }
    }

    /// This action as a menu row. Generate declares its lists as cases and
    /// turns them into rows HERE, so a pane's contextual menu and the click
    /// menu beside it are one list drawn by the app's one renderer.
    var row: RowAction<GenerateAction> {
        RowAction(kind: self, title: title, isDestructive: isDestructive)
    }
}
