import Foundation

// Third-party licences: reading the gate and recording consent. Consent and
// acquisition are separate acts -- neither route moves a model's bytes.
public extension HTTPBackend {
    /// Every gated licence and whether THIS machine has accepted it.
    func licenses() async throws -> [ThirdPartyLicense] {
        let listing: LicenseListing = try await get("/api/licenses")
        return listing.licenses
    }

    /// Records consent on this machine and answers with the refreshed state,
    /// so nothing has to re-read (`routes.rs:11506-11515`).
    @discardableResult
    func acceptLicenses(_ acceptances: [LicenseAcceptance]) async throws -> [ThirdPartyLicense] {
        let listing: LicenseListing = try await post(
            "/api/licenses/accept", body: AcceptLicensesBody(acceptLicenses: acceptances))
        return listing.licenses
    }
}

/// `routes.rs:11477-11481`. `MoldJSON.encoder`'s snake-case conversion turns
/// `acceptLicenses` into the wire's `accept_licenses`.
struct AcceptLicensesBody: Encodable {
    let acceptLicenses: [LicenseAcceptance]
}
