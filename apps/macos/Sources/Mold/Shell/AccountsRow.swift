import MoldClient

/// What one provider's row in `AccountsSettings` says, boiled down to three
/// cases -- pure so `AccountsTests` can check the words without a view.
enum AccountsRow: Equatable {
    case unset
    case environment(masked: String)
    case stored(masked: String)

    static func resolve(_ state: CatalogCredentialState) -> AccountsRow {
        guard state.configured, let masked = state.masked else { return .unset }
        return state.isFromEnvironment ? .environment(masked: masked) : .stored(masked: masked)
    }

    var summary: String {
        switch self {
        case .unset:
            "Not set."
        case let .environment(masked):
            "\(masked) — set by this machine's environment. Saving a token here overrides it, "
                + "and clearing it falls back."
        case let .stored(masked):
            "\(masked) — stored on this machine."
        }
    }

    /// A Clear button only removes something THIS machine stored -- an
    /// environment token has nothing here to clear, and offering the button
    /// anyway would be a control that lies.
    var offersClear: Bool {
        if case .stored = self { true } else { false }
    }

    /// A blank field has nothing to send -- the server 400s an empty token
    /// (`catalog_credentials.rs:293-295`), so Save simply does not ask.
    static func canSave(token: String) -> Bool {
        !token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}
