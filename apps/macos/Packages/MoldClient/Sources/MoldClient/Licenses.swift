import Foundation

/// One third-party licence and THIS machine's acceptance of it.
///
/// `GET /api/licenses` (`types.rs:12517-12541`). Acceptance is per Mold data
/// root, so a fleet holds one of these per machine and they never merge: a
/// licence accepted on workstation is not accepted on hal9000.
public struct ThirdPartyLicense: Codable, Hashable, Sendable, Identifiable {
    public let id: String
    public let name: String
    /// The immutable, commit-pinned text. With `sha256` this is the identity
    /// an acceptance is bound to -- not a link to show someone.
    public let url: String
    /// The browsable page. Presentation only, and deliberately NOT part of
    /// the accepted identity, because its contents move (`types.rs:12530`).
    public let canonical: String
    public let sha256: String
    public let summary: String
    /// A record bound to superseded terms reads as false (`types.rs:12536`).
    public let accepted: Bool
    /// Manifest names this licence gates. Present on every server.
    public let requiredBy: [String]
    /// The same models in the registry's own words. ADDITIVE -- an older host
    /// omits it and the row falls back to `requiredBy`.
    public let requiredByStyles: [LicensedStyle]?

    public init(
        id: String, name: String, url: String, canonical: String, sha256: String,
        summary: String, accepted: Bool, requiredBy: [String] = [],
        requiredByStyles: [LicensedStyle]? = nil
    ) {
        self.id = id
        self.name = name
        self.url = url
        self.canonical = canonical
        self.sha256 = sha256
        self.summary = summary
        self.accepted = accepted
        self.requiredBy = requiredBy
        self.requiredByStyles = requiredByStyles
    }

    /// What to send back. There is no `accepted` field on the wire: naming
    /// the terms IS the acceptance (`types.rs:12560-12583`).
    public var acceptance: LicenseAcceptance {
        LicenseAcceptance(id: id, url: url, sha256: sha256)
    }
}

public struct LicensedStyle: Codable, Hashable, Sendable {
    public let name: String
    public let description: String

    public init(name: String, description: String) {
        self.name = name
        self.description = description
    }
}

public struct LicenseListing: Codable, Hashable, Sendable {
    public let licenses: [ThirdPartyLicense]

    public init(licenses: [ThirdPartyLicense]) {
        self.licenses = licenses
    }
}

/// Consent, carrying the exact terms that were shown.
///
/// `types.rs:12573-12583`. Every field is required and there is no bare-id
/// form: the server verifies `(url, sha256)` against what IT pins before
/// writing ANY entry in the array (`routes.rs:1193-1208`), so a mismatch
/// anywhere writes nothing at all.
public struct LicenseAcceptance: Codable, Hashable, Sendable {
    public let id: String
    public let url: String
    public let sha256: String

    public init(id: String, url: String, sha256: String) {
        self.id = id
        self.url = url
        self.sha256 = sha256
    }
}

/// The machine-readable half of a refusal. `types.rs:12604-12618`.
public struct LicenseRefusal: Codable, Hashable, Sendable, Identifiable {
    public let id: String
    public let name: String
    public let url: String
    public let canonical: String
    public let sha256: String
    public let summary: String

    public init(id: String, name: String, url: String, canonical: String, sha256: String, summary: String) {
        self.id = id
        self.name = name
        self.url = url
        self.canonical = canonical
        self.sha256 = sha256
        self.summary = summary
    }

    public var acceptance: LicenseAcceptance { .init(id: id, url: url, sha256: sha256) }
}

public enum LicenseCode {
    /// 403 -- nothing is accepted yet (`types.rs:12621`).
    public static let notAccepted = "LICENSE_NOT_ACCEPTED"
    /// 409 -- both sides know the licence and disagree about its terms
    /// (`types.rs:12601`). Same payload shape, a different sentence.
    public static let termsMismatch = "LICENSE_TERMS_MISMATCH"
}
