import Foundation

/// Third-party licences: reading the gate and recording consent.
public protocol MoldLicencesBackend: Sendable {
    /// Every gated licence and whether THIS machine has accepted it.
    func licenses() async throws -> [ThirdPartyLicense]
    /// Records consent on this machine and answers with the refreshed state.
    @discardableResult func acceptLicenses(_ accept: [LicenseAcceptance]) async throws -> [ThirdPartyLicense]
}
