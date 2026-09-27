import Foundation

/// mold's error envelope. `code` is the part to branch on.
///
/// Every field is optional because half the routes this app calls do not send
/// all of them: `create_download`'s 400 is `{"error": …}` with no code
/// (`routes.rs:11563-11569`), and the catalog routes answer plain text
/// (`catalog_api.rs:586-590`). Requiring both threw the machine's own
/// sentence away and left "the machine answered with an error (400)".
struct APIError: Decodable, Sendable {
    let error: String?
    let code: String?
    /// Present only on a licence refusal (`routes.rs:38-49`).
    let license: LicenseRefusal?
}
