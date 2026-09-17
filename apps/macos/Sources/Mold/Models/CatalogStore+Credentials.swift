import Foundation
import MoldClient

/// A machine's own Hugging Face and Civitai tokens (Settings ▸ Accounts, S7).
///
/// Credentials live on the MACHINE, not on this Mac (design fact 9), so this
/// is the same per-host state Discover reads through `CatalogStore`, split
/// into its own file purely for size.
extension CatalogStore {
    /// Read once per host -- a hint in Discover that a provider token would
    /// surface more results, and the state Settings ▸ Accounts shows before
    /// anyone has touched a field.
    func loadCredentials(on host: MoldHost.ID) async {
        guard byHost[host]?.credentials == nil, let client = hosts.backend(for: host) else { return }
        guard let status = try? await client.catalogCredentials() else { return }
        byHost[host, default: HostState()].credentials = status
    }

    /// Writes a token on the named machine. The response IS the refreshed
    /// status (`catalog_credentials.rs:171-207`) -- never a second fetch, and
    /// never the token itself read back.
    @discardableResult
    func saveCredential(_ provider: String, token: String, on host: MoldHost.ID) async -> Bool {
        guard let client = hosts.backend(for: host) else { return false }
        do {
            let status = try await client.setCatalogCredential(provider, token: token)
            byHost[host, default: HostState()].credentials = status
            hosts.succeeded(on: host, doing: "save the \(providerName(provider)) token")
            return true
        } catch {
            hosts.report(error, on: host, doing: "save the \(providerName(provider)) token")
            return false
        }
    }

    /// A STORED token wins over the environment on that machine (fact 9), so
    /// clearing one falls back to whatever `HF_TOKEN`/`CIVITAI_TOKEN` says
    /// there -- which is exactly the status this answers with.
    func clearCredential(_ provider: String, on host: MoldHost.ID) async {
        guard let client = hosts.backend(for: host) else { return }
        do {
            let status = try await client.clearCatalogCredential(provider)
            byHost[host, default: HostState()].credentials = status
            hosts.succeeded(on: host, doing: "clear the \(providerName(provider)) token")
        } catch {
            hosts.report(error, on: host, doing: "clear the \(providerName(provider)) token")
        }
    }

    private func providerName(_ provider: String) -> String {
        provider == "hf" ? "Hugging Face" : "Civitai"
    }
}
