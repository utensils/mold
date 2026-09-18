import AppKit
import MoldClient
import SwiftUI

// Where the pane meets the file: a restore on first appearance, and a
// debounced write on every change after it.
struct PersistedDraft: ViewModifier {
    let controller: GenerateController
    let drafts: DraftPersistence

    func body(content: Content) -> some View {
        content
            .task { restore() }
            // The MODEL and the machine are part of what was being authored,
            // so a change to either is a change worth writing -- and the
            // draft's own value covers everything else.
            .onChange(of: descriptor) { _, next in drafts.schedule(next) }
            // A quit does not wait for a debounce -- and `onDisappear` is NOT
            // a quit hook: on macOS it is not reliably delivered for the key
            // window's content on Cmd-Q, so the last word of a prompt typed
            // and immediately quit on was lost inside the 400 ms debounce.
            // `willTerminate` is the one that always arrives; `onDisappear`
            // stays for the ordinary case of the pane going away.
            .onDisappear { drafts.flush(descriptor) }
            .onReceive(NotificationCenter.default.publisher(
                for: NSApplication.willTerminateNotification)) { _ in
                drafts.flush(descriptor)
            }
    }

    private var descriptor: DraftDescriptor {
        DraftDescriptor(controller.draft, model: controller.modelName,
                        family: controller.modelFamily, recipeID: controller.recipeID)
    }

    /// Puts the last draft back, ONCE, and only over an untouched pane.
    ///
    /// A restore that ran after somebody had started typing would throw their
    /// prompt away -- so an already-edited draft wins, exactly as a live
    /// conditioning value beats a parked one.
    private func restore() {
        guard let descriptor = drafts.restore() else { return }
        guard controller.draft == RenderDraft() else { return }
        var draft = controller.draft
        descriptor.apply(to: &draft)
        controller.draft = draft
        // The model and machine are RECORDED, not selected here: the model
        // list arrives asynchronously, and `GeneratePane+Models` adopts one
        // through the ordinary path so the recipe reconciles the draft it
        // just restored rather than a default one.
        controller.modelFamily = descriptor.family ?? controller.modelFamily
        controller.recipeID = descriptor.recipeID ?? controller.recipeID
    }
}

extension View {
    /// Keeps the Generate draft across launches (`DraftPersistence`).
    func persistingDraft(
        _ controller: GenerateController, in drafts: DraftPersistence
    ) -> some View {
        modifier(PersistedDraft(controller: controller, drafts: drafts))
    }
}
