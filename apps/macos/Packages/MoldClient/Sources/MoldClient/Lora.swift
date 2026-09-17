import Foundation

/// LoRA adapter constants shared by the reading extensions.
///
/// A stub today -- S2 grows this into the adapter stack the Adapters group
/// renders. It exists now because `RecipeCapabilities.loraStack` needs a
/// fallback cap for a host that advertises `supports_lora: true` with no
/// `lora` block at all (an older host that predates the block).
public enum Lora {
    /// `generation_profile.rs:2316` -- every recipe but LTX-2's `ic-lora`
    /// pipeline gets four slots; `ic-lora` itself advertises its own
    /// narrower `lora.max_count` via the block, so this is only the
    /// no-block fallback.
    public static let defaultMaxStack = 4
}
