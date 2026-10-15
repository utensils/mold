import MoldClient

extension ReuseStore {
    /// A durable local snapshot no longer needs an archive locator. This is
    /// a completion receipt, never permission to hydrate remote descriptors.
    func restoreLocalInputsIfComplete(_ inputs: DraftInputSnapshot, saved: SavedReuse) -> Bool {
        guard saved.version == 1, inputs.version == 1, saved.localInputsComplete == true,
              inputs.retainedReuseFingerprint == nil, Self.hasOnlyLocalReferences(inputs) else { return false }
        clear()
        return true
    }

    /// Record completion only after the live authoritative inventory has no
    /// hidden roles left. A populated picture alone cannot prove this.
    func rememberLocalInputsComplete(in draft: RenderDraft) {
        guard authority == nil, !restorationFailed, var saved = savedRecipe,
              Self.hasOnlyLocalReferences(DraftInputSnapshot(draft.media)) else { return }
        saved.localInputsComplete = true
        savedRecipe = saved
        savedFile?.save(saved)
    }

    private static func hasOnlyLocalReferences(_ inputs: DraftInputSnapshot) -> Bool {
        (inputs.active.generationReferences + inputs.parked.generationReferences).allSatisfy {
            $0.media.authority == "inline" && $0.media.data != nil
        }
    }
}
