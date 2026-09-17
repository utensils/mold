import AppKit
import MoldClient
import SwiftUI

/// What a finished result can do, as closures the pane supplies once.
///
/// `RunCanvas` is a view over a `RunState` and holds no draft; reusing a
/// result as a source picture or a reference is a draft mutation. Passing the
/// two in keeps the canvas view-only and keeps ONE definition of each verb,
/// shared by `ResultBar`'s buttons and the contextual menu beside them.
struct ResultActions {
    let save: (BatchResult) -> Void
    let copy: (BatchResult) -> Void
    let showInLibrary: () -> Void
    /// `nil` where the recipe has no source well at all.
    let useAsSource: ((BatchResult) -> Void)?
    /// `nil` where the recipe takes no references, or the strip is full.
    let addAsReference: ((BatchResult) -> Void)?

    var menu: [GenerateMenus.Row] {
        GenerateMenus.result(canUseAsSource: useAsSource != nil,
                             canAddReference: addAsReference != nil)
    }

    func perform(_ action: GenerateAction, on result: BatchResult) {
        switch action {
        case .saveACopy: save(result)
        case .copyResult: copy(result)
        case .showInLibrary: showInLibrary()
        case .useAsSourceImage: useAsSource?(result)
        case .addAsReference: addAsReference?(result)
        default: break
        }
    }
}

/// The result menu where the result may not exist yet -- the canvas draws its
/// media before it knows which child is selected in a settled outcome.
struct OptionalResultMenu: ViewModifier {
    let result: BatchResult?
    let actions: ResultActions

    func body(content: Content) -> some View {
        if let result {
            content.resultContextMenu(result, actions: actions)
        } else {
            content
        }
    }
}

extension View {
    /// The result menu, rendered from the ONE list so the canvas's big
    /// picture and every strip tile offer the same things in the same order.
    func resultContextMenu(_ result: BatchResult, actions: ResultActions) -> some View {
        rowActionMenu(actions.menu) { actions.perform($0, on: result) }
    }
}
