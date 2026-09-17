import Foundation

// The corollary of `PromptTransform.swift`'s own rule, on the one type that
// carries decoded open enums back OUT to a server.
public extension PromptTransformProvenance {
    /// This provenance as the wire can carry it -- or `nil`, because some of
    /// it cannot.
    ///
    /// Every enum in the Rust `PromptTransformProvenance`
    /// (`types.rs:711-727`) is STRICT: none of them has an `unknown` variant.
    /// A host that names a task or a dimension this build predates decodes to
    /// `.unknown` (which is the point of `OpenWireEnum`), but re-encoding that
    /// literal makes serde refuse the entire `POST /api/generation-batches`
    /// body. The failure is sticky and opaque: the poisoned block lives on in
    /// the draft, `revertExpansion()` only reaches it while the prompt is
    /// still byte-identical, and every later press of Generate fails naming
    /// `prompt_transform` rather than the wand that wrote it.
    ///
    /// Provenance is a record of how a prompt was written, never a reason to
    /// refuse a render -- so each field degrades as far as it honestly can.
    /// `operation` and `task` are required with no server-side default, so an
    /// unspellable one drops the block. `source_kind` is `#[serde(default)]`
    /// `Direct` (`types.rs:698, 724`), so an unspellable one becomes that.
    /// A dimension is dropped on its own; losing the whole record over one
    /// word nobody reads would be the same overreaction in reverse.
    var wireSafe: PromptTransformProvenance? {
        guard operation != .unknown, task != .unknown else { return nil }
        return PromptTransformProvenance(
            operation: operation,
            rootPrompt: rootPrompt,
            sourcePrompt: sourcePrompt,
            sourceKind: sourceKind == .unknown ? .direct : sourceKind,
            task: task,
            dimensions: dimensions.filter { $0 != .unknown })
    }
}
