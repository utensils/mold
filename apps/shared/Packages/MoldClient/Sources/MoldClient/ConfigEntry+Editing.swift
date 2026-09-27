import Foundation

/// The Advanced renderer's whole inference, derived from one row and nothing
/// else -- nothing on the wire declares a type (`types.rs:12833-12852`), so
/// this IS the type.
public extension ConfigEntry {
    enum Editor: Hashable, Sendable { case toggle, number, text, secret, unset }

    /// How to draw this row, from the VALUE's JSON type. A secret key wins
    /// over its value's own type -- `runpod.api_key` at `null` is still a
    /// secret, not an unset text field.
    var editor: Editor {
        if Self.secretKeys.contains(key) { return .secret }
        switch value {
        case .bool: return .toggle
        case .number: return .number
        case .string: return .text
        case .null: return .unset
        }
    }

    /// The two keys whose getter returns the literal `"<set>"` rather than
    /// the value (`config_keys.rs:529-531`, `:552-555`). There is no generic
    /// secret contract on the wire; the studio hard-codes the same two
    /// (`studio/api/config.ts:170-177`).
    static let secretKeys: Set<String> = ["runpod.api_key", "lambda.api_key"]

    /// What a text field starts with. EMPTY for a secret, always: `"<set>"`
    /// is a mask, and writing it back would set the key to the literal
    /// string `<set>`.
    var editableText: String {
        if editor == .secret { return "" }
        switch value {
        case let .string(text): return text
        case let .number(number): return Self.numberText(number)
        case let .bool(bool): return bool ? "true" : "false"
        case .null: return ""
        }
    }

    /// Words for a secret's state, since its field is blank either way.
    var secretState: String { value == .null ? "Not set" : "Set" }

    /// `source == "env"`: PUT answers 403 `ENV_OVERRIDDEN`
    /// (`routes_config.rs:192-197`), so the row is read-only before it is
    /// tried.
    var isEnvOwned: Bool { source == "env" }

    /// `source == "db"`: exactly DELETE's own gate
    /// (`routes_config.rs:270-276`), so Reset is never offered where it
    /// refuses.
    var canReset: Bool { source == "db" }

    /// Absent means false. Only the three `scheduler.*` keys ever carry it
    /// (`routes_config.rs:63`).
    var needsRestart: Bool { restartRequired ?? false }

    /// Typed text back to a scalar, by editor. An emptied field sends
    /// `.null` rather than an empty string -- a null body is how an optional
    /// key clears (`routes_config.rs:143`). `nil` means the text does not
    /// parse for this editor.
    static func scalar(from text: String, editor: Editor) -> ConfigScalar? {
        switch editor {
        case .toggle:
            if text == "true" { return .bool(true) }
            if text == "false" { return .bool(false) }
            return nil
        case .number:
            if text.isEmpty { return .null }
            guard let number = Double(text) else { return nil }
            return .number(number)
        case .text, .secret, .unset:
            return text.isEmpty ? .null : .string(text)
        }
    }

    private static func numberText(_ value: Double) -> String {
        value.truncatingRemainder(dividingBy: 1) == 0 && abs(value) < 1e15
            ? String(Int64(value))
            : String(value)
    }
}
