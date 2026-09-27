import Foundation

/// One control on a curated Settings pane: the key it edits, how to draw it,
/// and what the engine will refuse outside of.
///
/// Nothing on the wire carries any of this (`types.rs:12833-12852`), so it is
/// authored here and pinned to `crates/mold-core/src/config_keys.rs` by
/// `SettingKeysContractTests` -- the same arrangement
/// `studio/lib/settingsSchema.ts` has with the same file, and the same
/// arrangement `ModelFamilyContractTests` has with `manifest.rs`.
public struct SettingKey: Hashable, Sendable, Identifiable {
    public enum Editor: Hashable, Sendable {
        case toggle
        case text
        case number(min: Double, max: Double, step: Double?)
        case choice([String])
    }

    public let key: String
    public let label: String
    public let help: String
    public let editor: Editor

    public init(key: String, label: String, help: String, editor: Editor) {
        self.key = key
        self.label = label
        self.help = help
        self.editor = editor
    }

    public var id: String { key }
}

/// Namespace for the curated per-pane key arrays: one static array per pane
/// (S4a: `generation`, `expansion`; S4b adds `library`, `performance`).
/// General has no `SettingKey` array of its own -- its two notification
/// toggles and its media-cache cap are client-only `AppStorage`, never a
/// server row -- so a future pane is a one-line addition to `all` rather
/// than a change to every consumer.
public enum SettingKeys {
    /// Every curated key, pane by pane. `SettingKeysContractTests` flattens
    /// this for its "is every curated key real" and "no key curated twice"
    /// checks; a view reads one pane's own array directly.
    public static var all: [[SettingKey]] { [generation, expansion, library, performance] }
}
